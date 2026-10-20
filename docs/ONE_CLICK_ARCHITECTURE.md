# 一键翻译流水线 — 架构说明

> 覆盖 `one_click_translate.py`（stage1 采集 + stage2 处理）与 `photos_capture.py`（窗口采集）。
> 本流水线以 **Photos 蓝框为 OCR 最小单位**：蓝框 = 用户框选的一次选区，不可分割；
> 蓝框之外的一切（轮廓岛）保持原样，不 OCR、不翻译、不擦字、不嵌字。

## 一、总览

```
原图目录 (--images)
   │
   ▼  stage1 (默认 / --capture-only)        ─── photos_capture.py
Photos 并行窗口 (上3下3, 行距=窗口高150%)
   └ 窗口固定为页面尺寸×25% (--display-scale, 原图文件不变)
   └ 扫描文本 → Ctrl+A 抓取蓝框 (重试 0.4/0.8/1.2s) → selection_blocks / selection_groups
   └ 生成 {page}.bluebox_check.png (蓝框蓝色 / 整页选中红色)
   │
   ▼  stage2 (--process)                    ─── one_click_translate.py
   1. build_bluebox_dataset → islands.json (纯蓝框聚合组, 无高亮岛识别)
   2. build_bluebox_masked → bluebox_masked.png (整页填黑图)
   3. manga-ocr 只读蓝框 (从填黑图裁剪, 蓝框扩6px), 读空2次跳过
   4. 蓝框按组拼接整句 → 方向判定 (宽≥高=横排, 否则竖排右→左)
   5. 本地 Sakura 翻译 (泄漏/重复/超长/短符号/3次上限 防护)
   6. 擦字 (只擦有蓝框的岛) → 嵌字 (两级边界 + 字号约束) → typeset.png
   7. 合并 {书名}.pdf
```

## 二、核心概念

| 概念 | 定义 |
|---|---|
| 蓝框 (blue box) | Photos 选区，`selection_blocks[]`，`page_scale=false` 的才是正文单位 |
| 整页选中 (page_scale) | Photos 误选了整页（封面/跨页），标记红色，不驱动 OCR 与嵌字 |
| 组 (group) | `selection_groups[]`，同一句子的蓝框集合（按句聚合） |
| 岛 (island) | `islands.json` 中的**蓝框聚合组**（纯蓝框，无高亮岛识别；蓝框外内容不产生岛） |
| 填黑图 | 整页一张，蓝框扩 6px 框内原像素、框外填充色（`--bluebox-fill` 默认黑） |
| 有框岛 | 岛 bbox 内含任一蓝框中心 → 擦字 + 嵌字 |
| 无蓝框页 | 无任何可用蓝框 → 整页保持原样（不 OCR、不翻译、不擦字、不嵌字） |

## 三、stage1 — 采集（框来源二选一）

> **两条路径产出同一份 `capture.json` 契约**（`source` + `capture.selection_blocks`
> / `selection_groups` / 状态字段），因此核对图、dataset 构建、填黑图、20%/40%
> 分组、stage2 全部**零改动共用**。切换开关 `--box-source {paddle,photos}`
> （GUI 设置页「框来源（采集）」），**默认 paddle**。

### 3.0 PaddleOCR 全页检测（`--box-source paddle`，默认）

- 对每页原图跑 PP-OCRv6_medium_det（`paddle_ocr_worker.py`，GPU `auto`），
  检测框 → 过滤（`min_side=6px`；面积 > 20% 整页的框丢弃，同 Photos 规则）
  → 按阅读序（上→下、右→左）编号 `sel_001…` → 20%/40% 分组写 `selection_groups`
- **无窗口自动化**：不开 Photos、无焦点/时序/漂移问题，离线确定性；
  GPU 下 32 页约 7 秒（Photos 路径约 20s/页）
- 检不到任何框 → `selection_review_required: true`（沿用人工审核门硬规则）；
  整页全黑 → `black_page: true` 跳过
- **大拟音/效果字过滤**（`_filter_large_sfx`，默认开，`--keep-large-boxes` 关）：
  判据 = 框**短边** min(w,h) > 2×**同方向**基准 且 > 页短边 ×10%。
  - 用短边而非单边长度：文字行无论横竖，短边≈字号、逐页稳定；大拟声词两个
    方向都大。**按宽度筛必然误伤横排行**（实测本册最宽对白框 0.11 页宽，
    而竖排短边中位仅 0.038 页宽）。
  - 基准**按方向分组**（横比横、竖比竖）；某方向样本 <3 时回落全页基准——
    漫画以竖排为主，待删的大拟声词往往就是该页**仅有**的横框，严格"横比横"
    会让基准等于离群值本身而全部漏删（实测页 32：横框仅 1 个且就是那个
    385px 大字，基准 385 → 阈值 962 → 一个都删不掉）。
  - 实测（图源样张 全册）：删 9/541，与蓝框对白重叠 0；9 个全是大号拟音
    效果字（ドッ/ズズッ/グッ/フッ/ギュッ…）。与"整页大框 >20% 面积"规则叠加。
- **粒度**：输出"文本行"级框。对白页与 Photos 的按句切分接近；**大字/疏排页面
  Photos 会退化成逐字切分，Paddle 仍是整行**（实测见 开发记录，注意页 45 框 vs 13 框）

### 3.0.0.1 擦字缓存与并行（性能）

- **擦字是全链最重的一步**：实测页 1（10 个岛）**12~13s**，其中 **OpenCV FSR
  修复占 ~87%**（4.7s）；字形分割 0.5s、风格判定 0.1s、掩膜/审计 <0.1s。
  对比：**嵌字排版+绘制仅 0.048s**（"嵌字慢"其实是"擦字慢"）
- **擦字缓存** `dataset/{page}/erased.png` + `erased.key`：
  - 键 = 原图 sha256 + 每岛擦字矩形 + 最小岛面积 + 擦字配置 + `ERASE_CACHE_VERSION`
  - 任一输入变化即失效（改框/换图/改配置都不会串用旧结果，有断言覆盖）
  - **验证过输出逐字节一致**（命中与不命中两份 typeset.png 的 sha256 相同）
  - 收益：改排版/编组/重合并 PDF 时 **32 页 86s → 3s**
  - 属于可删的中间产物：删了就重擦，结果一致
- **页级并行**：`--max-workers`（GUI「并行页数」）默认 **3**（上限 3）。
  擦字/嵌字按页并行，冷启 32 页 86s → 50s
- 另：`translations.json` 改为**按页合并**写入（原为整文件替换，`--pages 1`
  这类子集运行会把其余页译文整本抹掉——实测踩到过）

### 3.0.0 目录布局（两来源的产物分开存放）

```
outputs/oneclick/{书名}/
├── capture/paddle/{page}/capture.json     Paddle 检测的框
├── capture/photos/{page}/capture.json     Photos 采的框（+ photos_base/ocr/selected 帧）
├── check/paddle/{page}.bluebox_check.png  红框
├── check/photos/{page}.bluebox_check.png  蓝框
├── dataset/{page}/…                       单套，由本次运行选用的来源生成
├── typeset/{page}.typeset.png             单套，同上
├── translations.json / ocr_readings.json  单套，同上
└── run.json / run.log
```

- **capture / check 按来源分层**：切来源不会覆盖对方的采集结果，两条路径可共存
- **dataset / typeset / 缓存是单套**：由本次 `--box-source` 生成；换来源重跑即覆盖
- stage2 读取时**优先本次指定的来源**，该来源没有这页则回退另一来源（并在日志里说明），
  所以"用 paddle 跑 stage2、只采过 photos"也能继续
- `BLUEBOX_REPORT.md` 的条目带来源标记（`[paddle] 第 N 页：…`）
- 读写的路径解析集中在 `_capture_source_dir / _find_capture_page / _iter_capture_pages`
  几个 helper 里，避免各处硬拼路径

### 3.0.1 两条框来源的规则对照

**共用同一套阈值**（集中在 `manga_localizer/thresholds.py`，两边 import 同一份，
避免"规则应当相同、实际各写一个数"的漂移）：

| 规则 | 值 | 作用 |
|---|---|---|
| 整页大框过滤 `WHOLE_PAGE_AREA_RATIO` | 面积 > **20%** 页 → 丢 | 不是原子文字单位 |
| 大拟音/效果字过滤 `SFX_MIN_SIDE_*` | 短边 > **2.0×** 同方向基准 **且** > 页短边 × **10%** → 丢 | 风格化大字 |
| 手绘字过滤 `ART_TEXT_*`（**默认关**，`--drop-art-boxes` 开） | 岛级**严格联合版裁决**（见下文「严格联合版岛级裁决」）：留日文 = (P<0.6 ∨ 框级软标记) ∧ 字数<7；权重缺失才退回几何判据（取样矩形平坦占比 < 本书闸门 且 原文 ≤10 字） | 画在画面上的效果字/拟声 |

三条过滤对 **Photos 与 Paddle 都生效**：整页大框在采集期丢，大拟音在**共用的
数据集构建阶段**（`build_bluebox_dataset`）丢——放在共用阶段，两条路径才会
得到同一结果。

**手绘字过滤默认关**，三个理由别混：

- 在它适用的书上它很准：甲书 闸门 0.70，324 个岛里丢对 163 / 误伤 3（岛级
  F1 0.985）。
- 但它**不是每本书都适用**：乙书 上精确率只有 30.9%（强跑等于把一半对白
  当拟声丢掉）。所以先**按书自检**，自检不过就整本禁用。
- 自检本身也会出错（判据是几何的，而"拟声还是台词"是语义的），所以决定权留给
  用户：确认这本书的手绘字以拟声为主再开。

**按书自检**（`art_text_gate`）用这本书自己的**长文本岛**（≥12 字，几乎必然是对白）
的平坦占比自证"气泡基准"：中位 × 0.9 当闸门。基准必须**又高又稳**——**p25 ≥ 0.65**
才放行，否则整本禁用（一个岛都不丢）。实测：甲书 p25 0.76 → 闸门 0.69（岛级
F1 0.985）；乙书 p25 0.56 → 禁用。**用 p25 而不是 p10**：长文本岛里必然混着少数
横排印刷长文（封面警示语 0.17 / 后记段落 0.34）与**偶然被读出 ≥12 字的手绘字**
（`ぬるっと♡じゅっ必ずい` 0.15），p10 那个位置正好由它们决定——同一本书换个编组就
从 0.67 掉到 0.37（误禁用）。

手绘字判据横跨两个阶段，是因为它的两个条件来源不同：**几何**在数据集构建期算
一次、存进 `islands.json`（`bg_flat_ratio` + 判据用的取样矩形 `members_bbox`）；
**读数**（原文 ≤10 字，用来放过压在画面上的印刷正文——封面警示语、后记段落）
要等 OCR，判定在 `build_page_texts`（`art_text_island`）。命中即把该岛文本判空，
走既有的"读不出字的岛保留原图"出口：**不擦、不译、不嵌**。存的是比例而不是
布尔，所以调阈值只需重建 dataset，不必重跑翻译。

两个实测逼出来的细节：

- **取样矩形用成员框原样并集 ±2px，不用岛的 `image_bbox`**：后者带编组扩边
  6px，而扩边是给填黑/擦字/嵌字的版面余量，不是文字范围——拿它取样会把邻居的
  墨迹吃进来（实测页 6「えーと」被邻居压掉 30% 面积）。
- **同一处字被切成几个岛时同判**：重叠框读数归一化后互为前缀（`_same_lettering`）
  就并成一组，占比取组内最小、原文取最长（取最短会把印刷长句的碎片误杀）。
  不这么做，同一个字会拿到两个相反的判定（实测「かわいい♥」0.46 留 /
  「かわい♡」0.37 丢）。
- 另外 AI 的**说明性输出**（"图片中没有任何日文文本…"）在 `build_page_texts` 里
  按空读数处理——它不算读数，但字数很多，会把 ≤10 字的护栏撑爆（实测页 4 一个
  本该被判手绘字的岛就是这样被放行的）。


自检不住的书（`art_text_gate` 返回 `None`）**改用视觉判据**：若配了视觉接口
（`ai_ocr_url/model/api_key`），逐岛问模型 `S`（拟声→不译）/ `D`（台词→译）/
`U`（图里没有可读的字→不译），送图带 ±60px 上下文，跳过长文本岛与无读数岛，
结果缓存进 `art_text_judge.json`。**没配接口、或数据集没算过 `bg_flat_ratio`
（功能没开）时一律不调**，保持"全部保留原图"。

详见 开发记录「手绘字过滤」一节。

**严格联合版岛级裁决（2026-09-30 用户定稿，取代"分类器只看 P"的旧裁决）**：

```
留日文 = (P(台词) < ART_TEXT_CLASSIFIER_THRESHOLD(0.6) ∨ 框级软标记) ∧ 字数 < ART_TEXT_STRICT_MAX_CHARS(7)
```

- **框级软标记**（`soft_flag_large_sfx`）：现行大拟音硬过滤（2.0×/10% 硬丢）的
  放宽档（1.2×/5%），**只标记不丢框**（岛上的 `sfx_soft_flag` + islands.json 的
  `soft_flag_version` 标记）——框和文本都留在数据集里，成岛后岛级裁决，被救回的
  岛全文翻译、擦字覆盖全部成员框（无残墨、无断句）。实测软标记贡献了严格版
  453 个拦截里的 424 个。
- **字数护栏**：≥7 字的岛**结构上**不因拟声判据留日文——成句台词的安全是逻辑
  保证，不依赖模型可靠度。依据是成本重定义：「丢」= 该岛保留日文原样（不擦不译
  不嵌），误杀台词的真实代价是"这句以日文留在画面"，实测成句台词（≥12 字）到
  P<0.7 才开始出现这种情况（三本书共 2 个）。
- **复核加厚带**：被标记 ∧ 字数 5~6 的岛照 k=7 丢，但**全部**写进
  `art_text_review.csv`——这是"救门槛 k=5（多保短台词）"与"k=7（多拦拟声）"
  两个方案优点的结合落点（该字数段台词与拟声所有本地信号同形，实测交换率
  2.6:1 即墙，交给人工定夺，判错写 keep 即救回）。
- 人工 `art_text_force_drop.txt` 优先于护栏；`keep` 清单照常救回。
- **与三层路由的关系**：分类器权重可用时，书级自检 / 逐页路由 / 视觉判据都不参与
  （分类器裁决书无关、对所有页生效）；权重缺失退回几何时才走原路由（§3.0.1 上文）。
  软标记与 `bg_flat_ratio` 同一门控（`--drop-art-boxes`），功能没开的书绝不因此丢岛。
- 旧 dataset 自动重建：缺 `soft_flag_version` 标记即视为旧规则产物。
- 全部实测数字与被否决路线见 开发记录「六之前四」；探针七个在 `var/probe_*.py`。

**按来源不同的只有两个补偿量**（框松紧不同，规则本身不分叉）：

| 补偿量 | `paddle`（默认） | `photos` | 含义 |
|---|---|---|---|
| `BBOX_PAD` | **4** | **6** | 布局边界 = 擦字范围 = 成员框并集 ± 该值 |
| `GLYPH_MARGIN` | **5** | **12** | 字号上限 = 组内最窄框 + 该值 |

依据：Photos 蓝框是**文本选区**（紧贴笔画，宽中位 29px），Paddle 检测框是
**文本行 + 余量**（宽中位 37px，约松 2px/边）。两个补偿量把观感拉平。

**仍然只有一条路径有的行为**（属"框从哪来"的固有差异，不是规则差异）：

| | Photos | Paddle |
|---|---|---|
| 定位方式 | 窗口自动化 + 蓝色蒙版像素公式 | PP-OCRv6_medium_det（GPU） |
| 框粒度 | 大字/疏排页会退化成**逐字** | 一律**整行** |
| 微框过滤（边 < 6px 丢） | ✗ | ✓ |
| 检出内容 | 只选"认得"的文字 → 漏拟音/小字 | 会检出拟音/效果字/小字 |

### 3.0.1.1 check 图颜色按框来源（红=Paddle / 蓝=Photos）
- check 图（`check/{page}.bluebox_check.png`，文件名保持不变，工具链到处引用）
  由 stage1 共用尾段绘制，画的是 capture.json 的 `selection_blocks`（扩 6px）
- **颜色按 `capture.box_source`**：Paddle 检测框画**红**（`#e6194b`）、Photos
  蓝框画**蓝**（`#4363d8`）—— 一眼看出这页的框来自哪条路径
- 注意：管线**只产这一张可视化**（原图 + 框）。"编组对比 / 填黑对比 / 检测对比"
  是分析工具 `ocr_box_compare.py` 的产物，不是管线输出；dataset 目录里没有可视化
- 审核页预览的回退链因此包含 check 图（否则 Paddle 页预览全空，因为 Paddle 的
  `capture/{page}/` 只有 capture.json，没有 `photos_*` 帧）：
  选中帧 → `photos_base` → **check 图** → 原始源图

### 3.0.2 筛框后无框 → 进审核页
- 采集期有框、但在**下游被过滤光**的页（微框 / 整页大框 / 大拟音 / 微小岛
  <800px²），下游 `_stage2_dataset` 会调 `_mark_review_required` 往 capture.json
  写 `selection_review_required` + `selection_review_reason` +
  `selection_empty_after_filter`
- 为什么必须补标记：这类页在采集期是有框的，采集期的审核标记覆盖不到；
  不补就会静默跳过，让人误以为"这页没问题"
- 审核页据此列出（原因显示「筛框后无框」），人工可二选一：
  - **重新识别（Paddle）** → `--box-source paddle --overwrite`：离线重检测，不开窗口、快
  - **重采所选（Photos）** → `--box-source photos --overwrite`：开 Photos 窗口重采（原流程）
- 两个按钮各自**显式覆盖框来源**，互不干扰（`_recapture_pages(box_source=...)`）

### 3.1 Photos 蓝框采集（`--box-source photos`）

- 并行 6 窗口：上 3 下 3，紧贴四边对齐（上贴顶、下贴底、左贴左、右贴右），窗口最小间距 32px（漂移容差）
- **显示缩放（上限 + auto-fit）**：`--display-scale <n>` 时每窗尺寸 ≤ 页面 × n%（宽高比一致），整页完整可见；
  行/列放不下时按比例自动缩小（下限 12%）保持并行数，仍放不下才减窗（线程数同步，绝不重叠）；
  原图文件不缩放、分辨率不变，坐标经 fit 变换还原回原图原分辨率；`--display-scale 100` 恢复固定平铺
- 窗口重锁 `_ensure_window_rect`：每次操作前核对窗口矩形，被挤跑自动回位
- 蓝框抓取重试：0.4s / 0.8s / 1.2s 三次
- 黑页静默：`_is_black_page`（dark 占比 >0.90）→ 标记 `black_page`；无文字页（Photos 不出现
  OCR 控件/蒙版）→ 标记 `no_text`；两者 stage2 均跳过
- **蓝框检测单路径**（`detect_selection_boxes`，详见 `BLUEBOX_CAPTURE.md`）：
  严格公式 `|selected - (keep×base + (0,38,67))| ≤ 5`，keep=174/255，
  paint-delta>2 守卫（排除页面固有纯蓝）；缺 base 帧或三次 Ctrl+A
  仍无蓝色像素 → `selection_review_required: true`，进审核门
- 整页大框过滤：连通域面积 > 帧 20% 的整页选中框**直接丢弃**（非原子文字框）
- **硬规则（人工审核门）**：三次 Ctrl+A 仍无蓝色像素 → `selection_review_required: true`，
  该页不进 stage2；stage1/2 提示待审核，`--confirm-no-boxes <页>` 确认无框后才可关闭
- 结果：`capture/{page}/capture.json`（选区 + 元数据）、`check/{page}.bluebox_check.png`
  （**原图直接叠加蓝色框**，无填黑、无高亮岛金框；蓝框按 OCR 实际使用的扩 6px 版本绘制）

## 四、stage2 — 后台处理 (one_click_translate.py)

### 4.1 数据集构建
- `build_bluebox_dataset` → `dataset/{page}/islands.json`：**纯蓝框聚合组**（每个组 = 一个岛，
  蓝框扩 2px），不做差分高亮岛识别；`selection_blocks.json` 同步写入（嵌字列位来源）
- 无蓝框页 → 不生成 dataset，整页保持原样（PDF 中为该页原图）
- 失败页 fallback：`island_extract.build_fallback_dataset`（fallback_manga 标记页）

### 4.2 填黑图 (build_bluebox_masked, pad=6)
```
masked = 全页填充色 (BLUEBOX_FILL)
每个非 page_scale 蓝框扩 6px → 框内粘贴原像素
```
- 作用：manga-ocr 裁剪输入。蓝框内只含本框文字 → 邻框文本永不泄漏进裁剪

### 4.3 OCR (run_manga)
- 子进程加载 manga-ocr 一次，遍历全部 crops（JSON 行协议输出）
- **只读蓝框**：`{page}:box:{block_id}`，source = 填黑图，bbox = 蓝框扩 6px
- 无框岛**不进 OCR**（填黑图填充区是纯色，manga-ocr 会幻觉输出，如 "それでも、"）
- 读空 2 遍判误选（同一轮内两遍读，非跨轮）：第一遍全量读 → 读空的框以
  `{page}:box:{id}:reread` 的独立 id **再读一遍** → 仍读空才判定为误选区，记入
  `dropped_boxes`，此后**全线不参与**（不进句子拼接、不擦字、不嵌字）
  - 因为缓存只存非空读数（见 4.8），恒空的框每次运行都会重演"读→复读"，
    不会被缓存跳过
- **AI OCR 全量优先（可选）**：`--stage2-engine ai-ocr` 时**全部框**走 OpenAI 兼容视觉模型
  （`manga_localizer/ai_ocr.py`，默认 DeepSeek `deepseek-flash`，配置在 `config/api_keys.json`
  的 `ai_ocr_url/model/api_key`）。视觉模型**读空的框回退 manga-ocr**
  （`_run_manga_subprocess`）——不是 Sakura，Sakura 是翻译层。手绘字/符号（♡）上视觉模型
  明显强于字符级 manga-ocr
- **说明性输出不是读数**（`text_rules.is_ocr_commentary`）：视觉模型读不到字时会回一句
  "图片中没有任何日文文本。它看起来像是…"，若当成读数就会一路翻成中文塞进原框
  （实测页 8 / 页 14 各一个岛：70 字塞进 96×57 的框 → 12px 兜底、几乎看不见）。
  命中即视为**空读数** → 回退 manga-ocr；仍读不出就按假选框丢弃、保留原图
- **回退读数可疑 → 不翻译（保留原文）**：回退框的读数若命中 `text_rules.suspicious_short_kana`
  （纯假名/符号且 ≤6 字符），判为**假选框**写入 `dropped_boxes` → 不翻译、不擦字、不嵌字，
  页面保持原样。实书实测依据：`ぎゅっ♡` 被 manga-ocr 读成 `どうし/どんな/どうやっ`，
  再被翻成"为什么/什么样/怎么会"，语义全错还擦掉了原文，代价远大于保留日文
  - 只作用于**回退**读数：视觉模型的读数不判（它读对了就是对的），正常 manga-ocr 读数也不判
    （否则 `どうして` 这类合法短台词会被误杀）
  - 智能体视角提示：这条与"`--stage2-engine ai-ocr`"配合才有意义——回退是低置信信号，
    低置信 + 短假名 = 大概率是风格化拟声词
- **并发（`--ai-ocr-workers`，默认 12、护栏 32）**：视觉 OCR 是纯网络等待型负载，
  逐框串行时每框一次请求（实测 1.91s/框，一本 586 框要 18+ 分钟）。改并发后
  实测 8 路 0.20s/框（9×）、16 路 0.10s/框（19×）、24 路反而退化（排队/限流）。
  每个框独立重试，单个失败不影响其它框（读空的框照旧回退 manga-ocr）
- **编码降采样（`MAX_SIDE=1280`）**：长边超上限才缩（漫画单框通常 50~300px，
  不触发），大框缩后传输与推理更快；字远大于该尺度，视觉模型仍读得清

### 4.4 组拼接与方向 (page_texts / page_directions)
- 组内蓝框按方向排序拼接为整句：
  - 横排（宽≥高的框占多数）：y 升序 → x 升序
  - 竖排：x 降序（右→左）
- 岛文本 = 组拼接结果（一个岛可能含多组，按组序连接）
- 方向判定结果存入 `directions`，供嵌字用

### 4.5 翻译 (island_translate.py)
- 引擎双级：**AI 整页批量优先**（`translate_page_batch`，OpenAI 兼容，默认 DeepSeek
  `https://api.deepseek.com/v1` + `deepseek-chat`，需 `--ai-api-key`/`config/api_keys.json`），
  失败/未配置 → **本地 Sakura 兜底**（`translate_island_page`，`http://127.0.0.1:8080/v1`，模型 `sukinishiro`）
- AI 批量缓存 `translation_batch_cache/`（按页批量签名 = 页 + 岛文本 + 模型 + 温度 +
  **术语表 sha + 提示词版本**）；本地缓存 `translation_cache/`
  （signature = 源文本 + 上下文 + 术语表 + 提示词版本）
  - 术语表 sha 必须进签名：术语表是提示词的正文（拟声词自动收录会改它），不进签名就会出现
    "改了术语表却命中旧缓存、改动毫无效果"（FORMAT_VERSION=2 起修正）
- **提示词约束**：译文用于嵌进**大小固定**的文字框（框由原文字数/框尺寸决定），提示词要求
  "长度以原文为准，日译中通常压到七成，宁短勿长，但不要为了压字数丢原意"。
  注意**不设硬性字数上限**：实测 1222 对真实译文/原文的字数比 p50=0.75、p95=1.00、仅 4.6% 超 1.0×
  （且全是 2~6 字的小感叹句），说明模型本身就压缩，硬卡上限只会伤到原意
  - 改任一处翻译提示词 → 必须升 `PROMPT_VERSION`（translator.py）/ `ISLAND_PROMPT_VERSION`
    （island_translate.py），否则旧缓存静默命中
- **拟声词自动进术语库**（`manga_localizer/glossary_auto.py`）：译好且**译法稳定**的拟声词回灌
  `config/glossary.json`，此后全书（含重跑）统一译法。三道护栏决定"只有准确的才进库"：
  1. 原文必须是 `is_sfx_like`（纯假名/符号 ≤6 字）；2. 译文 ≤6 字且不在疑问词黑名单
  （为什么/什么样/怎么会…——就是上面那类错读数的产物）；3. 同一原文 + 同一译法至少出现
  `--sfx-glossary-min-count`（默认 2）次
  - **冲突即撤出**：同一原文出现第二种译法 → 说明 AI 自己都没把握，自动条目撤出，
    宁可每次现译也不锁死一个可能错的译法
  - **手写条目永不覆盖**：note 不以「自动收录」开头的条目视为用户手写，只读不写
  - 入库会改术语表 sha → 页级翻译缓存失效 → 下次运行按新术语表统一重译（收敛，不会每轮都失效）

- 防护（`_validate_island_translation`）：
  - 空输出 / 原文照抄 → 拒绝
  - 泄漏提示词字段（本页前文/术语表/人名表/翻译成…）→ 拒绝
  - 单行重复 >4 次 → 拒绝
  - 长度 > max(24, 源×2.5) → 拒绝
  - 假名残留 > max(2, 译文×15%) → 拒绝
  - 泄漏上下文文本 / 无关术语表目标词 → 拒绝
- 短符号岛（≤2 字符标点/符号）→ 零引擎调用，直接保留
- 每岛硬上限 3 次引擎调用 → 仍失败则 kept-source（保留源文本，绝不留空）
- 每页译文同时落盘到 `translations.json`（原子写），使 `--stage typeset` 可独立运行于已翻译页

### 4.6 擦字 (erase_and_typeset_page_with)
- `dataclasses.replace` 构造**只含有框岛的 dataset 子集** → `erase_island_image` 只擦有框岛
- 无框岛像素**保留原样**（原始内容不动）
- 每岛先求**统一框 B**（该岛蓝框组并集 ±6px，`build_union_overrides["bbox"]`）；擦字 dataset 的岛 bbox 替换为 B，`extra_regions` 传整框 → **擦字范围 = 整个 B**（原文墨水 ∪ 全框矩形），擦除/合成/审计都以 B 为几何

### 4.7 嵌字 (island_typeset.py)
见下一节。

### 4.8 OCR 读数缓存 (ocr_readings.json)
- **为什么**：manga-ocr 无检测、每框一次推理；历史版本每次 `--process` 都全量重识别（全书 ~40 分钟）
- **机制**（`run_manga` / `_load_readings_cache` / `_store_readings_cache` / `_select_crop_misses`）：
  - 缓存文件 `outputs/oneclick/{书名}/ocr_readings.json`：`{"{page}:box:{box_id}": "识别文本"}`
  - 每次 OCR 前加载缓存，**只对缺失框跑 OCR**（manga-ocr 子进程 / AI-OCR 同一逻辑），结果合并后**原子写回**（tmp + replace；并行段之间 merge-on-store 不互丢）
  - **只缓存非空读数**：读空的框不入缓存 → 二次复读判误选逻辑完全保留
  - 缓存损坏/缺失 → 忽略并重建；重新采集（框变）→ 旧键自然失效
- **收益**：改分组/改嵌字/重合并 PDF 等任何重跑，OCR 阶段 ≈ 0 秒
- `--stage ocr` 独立运行时按同一缓存逻辑只补缺失框；`--overwrite` 会先剔除选中页的读数/译文键再全量重读
  - **清空前先备份**：`translations.json` / `ocr_readings.json` 会被复制成 `*.bak`
    （`_backup_before_purge`）。`--overwrite` 是"先删后重建"，而重建要跑完 OCR+翻译，
    中途失败（进程被杀/断网/额度不足）就成了"只丢不进"——实测踩到过一次，24 页译文
    被清成 `{}` 等于白花钱重翻。有了 `.bak`，最坏也只是回滚

### 4.9 输出
- `typeset/{page}.typeset.png`、合并 PDF `outputs/oneclick/{书名}/{书名}.pdf`

### 4.10 备份与恢复
- **代码/规则版本**：git 提交即备份（分组规则、嵌字规则、文档都随代码版本化）
- **书级产物**：`outputs/oneclick/{书名}/` 可整体复制/压缩即完整备份（含 capture、dataset、两套翻译缓存、OCR 读数缓存、typeset、PDF）；重新铺到同样路径即可恢复/继续
- **整库快照**：`robocopy outputs\oneclick backups\<日期>\oneclick /E` + `robocopy config backups\<日期>\config /E`
  （`backups/` 已被 .gitignore 忽略；`config/api_keys.json` 含密钥，勿提交 git）
- **恢复重跑**：删除 `dataset/*/islands.json` → 用新分组规则重建；翻译/OCR 缓存命中则零重算；`--resume` 跳过已有 typeset 页
- **单页重采**：删 `capture/<页>/`（+ `dataset/<页>/`）→ `--pages <页>` 重跑 stage1 → 核对 `check/<页>.bluebox_check.png` → `--pages <页> --process --no-pdf` 重嵌该页

### 4.11 可清理的冗余（详见 ONE_CLICK_USAGE.md 五.7）
- `crops.*.json`（每次运行生成的 OCR 临时任务文件）、`_logs/`（旧分段日志）
- `check/` 下旧管线调试图（`*.erased.png` / `*.erasure_mask.png` / `*.ranges_overlay.png` / `*.islands_grouped.png`）、同页旧格式 `*.bluebox_check.jpg`
- 项目根 `build/`（PyInstaller 中间产物，成品在 `dist/`）
- 不可删：`dataset/`、`typeset/`、`ocr_readings.json`、两套翻译缓存

### 4.12 阶段拆分与覆盖
- **阶段拆分（`--stage {ocr,translate,typeset}`，与 `--process` 互斥）**：只跑 stage2 的某一个阶段，
  `--pages` 选出该阶段页；各阶段有前置依赖，缺前置的页跳过并提示：
  - `--stage ocr`：需 `capture/{page}/`（已采集页），自动补建 `dataset/{page}/`，按缓存只补缺失框
  - `--stage translate`：需该页已有 OCR 结果（无结果页跳过并提示），AI 批量优先、Sakura 兜底，
    结果落盘 `outputs/oneclick/{书名}/translations.json`
  - `--stage typeset`：需该页已有 `translations.json` 译文（无译文页保持原图），擦字+嵌字+合并 PDF
- **`--overwrite`（覆盖旧结果，与 `--resume` 互斥）**：级联清除选中页旧产物后全量重做：
  - 采集覆盖 → 清 `capture/{page}/`、check 图、`dataset/{page}/` + 剔除该页 readings/translations 键
    → 下游阶段自然全量重做（杜绝新旧蓝框串用旧译文）
  - `--process`/`--stage ocr` 带 `--overwrite` → 清除选中页 dataset + 缓存键后全量重做/重读；
    `--stage translate` 带 `--overwrite` → 剔除该页译文键后重翻（签名缓存命中时结果一致、不花引擎费）
- 不带 `--overwrite` 时沿用缓存：采集跳过已采页、OCR 只补缺失、翻译重写文件、嵌字覆盖 png；
  `--resume` 与 `--overwrite` 互斥

## 五、嵌字标准（定稿）

### 5.0 蓝框分组（`_group_boxes_by_overlap`）

**顺序：筛框 → 编组（用原始框几何判）→ 再取扩框后的外接矩形。**
编组回答"这些框是不是同一个文本单位"，只看文字排布，不含给 OCR 的扩边量
（扩边是工程补偿，不是排版事实）；`pad` 只用于**组的外接矩形**（成员框扩边后的
并集），下游据此决定填黑/擦字/嵌字的范围。

- 竖排框对合并 **当且仅当**：**横向间距 < 30% × 较小框宽** 且 **纵向重叠 > 40% × 较大框高**
- 横排框对合并 **当且仅当**：**纵向间距 < 30% × 较小框高** 且 **横向重叠 > 40% × 较大框宽**
- 混合方向对**永不合并**；沿阅读方向堆叠（同列竖排堆叠 / 同行横排并排）**永不合并**（蓝框是可信原子单位）
- 重叠基准取**较大**框维度：避免"矮框只搭住高框一小截"也被合并（较小基准在顶端对齐时恒满足，形同虚设）
- **30% 是替换"含扩边 20%"的重标定**：旧规则在扩框后几何上判，扩边给每个间距
  加了约 2×pad 的**常量**，对小框的相对影响过大；改为原始框的纯比例后是**尺度无关**的。
  比例扫描（图源样张 532 框）：0.6/0.5/0.4/0.3/0.2 → 261/265/266/271/274 组；
  比例越低越多"多列一句"被拆开（逐页复核过页 3/6/19/23/35 的原文），
  由人工看图后定为 **0.3**
- 每组 → 独立岛，岛 bbox = **组外接矩形 ±2px**；**每组的翻译嵌字只在组内进行**

#### 5.0.1 字号上限与布局边界（按框来源补偿）
- **布局边界 / 擦字范围 = canonical box = 成员框并集 ±bbox_pad**（stage2 用
  `build_union_overrides` 覆盖岛 bbox；岛 bbox 本身不是布局边界）
- **字号上限 = 组内最窄框 + glyph_margin**（竖排取宽、横排取高）
- 因 Paddle 检测框自带约 2px/边 余量，而 Photos 蓝框紧贴笔画，两者用不同补偿：

| 框来源 | bbox_pad（布局/擦字） | glyph_margin（字号上限） |
|---|---|---|
| `paddle`（默认） | **4** | **5** |
| `photos` | 6 | 12 |
- 两者也刻意解耦：改布局边距不会牵动字号（字号上限只取框尺寸，不取裁剪后的值）

#### 5.0.1.1 描边溢出余量（`STROKE_OVERFLOW_MARGIN = 8`）
- 文字描边是**向外**画的（PIL `stroke_width = 字号//10`，最大约 6px），而块是按
  字身框贴边排的 → **最外圈描边必然伸到布局框之外**。不加处理就被"岛外零改动"
  的硬裁剪切掉，表现是**边缘的字缺一笔**（实测 223/268 个岛中招）
- 处理：**擦字范围与裁剪掩膜都按该余量放宽**（`extra_regions` 与
  `typeset_island_page` 的掩膜同步 +8px）。描边落在已擦除区域内，**不会覆盖
  原图像素**；审计相应按放宽后的掩膜判定，仍能抓住真正的越界改动
- 试过但否决的方案：把描边算进"块必须放得下"的判定 → 描边吃掉框的全部边距，
  字号中位 35→29px、pending_review 从 17% 涨到 76%，代价过大

#### 5.0.1.2 省略号统一 3 点（`text_rules.normalize_ellipsis`）
- 日漫常见 `……`（6 点）与 `．．．` 写法，中文排版按**3 点**。任何 3 个以上
  点状字符连写（含两个及以上 U+2026）收敛成单个 `…`；**2 点不改**
- **两边路径、两套翻译引擎共用一份规则**：翻译清洗（`translator._clean_translation`）
  与嵌字绘制前（`typeset_island_page`）各归一一次 —— 后者覆盖"保留源文本"的
  符号岛与历史译文，**重嵌字即生效**

#### 5.0.2 小组让位（避免嵌字互相干扰）
- 组矩形就是嵌字布局边界，**两组矩形相交 = 两段译文会打架**，故在**编组阶段**
  就解决：面积小的组让位，沿"移动量最小"的轴推开到不相交（四周留 1px）
- 位移上限 = 0.5 × 自身较短边：推不开就保持原位（宁可轻微重叠，不把文字甩远）
- 位移只记进数据集 `typeset_shift`，**不动成员框**——OCR 裁剪与擦字仍以原框为准
- stage2 应用时：布局框按位移平移，**擦字范围同时覆盖原位与让位处**（否则译文
  会压在没擦掉的原图笔画上）
- 实测 图源样张（Paddle 框 32 页）：24 个岛触发让位，位移 2~25px

### 5.1 统一框与布局边界（唯一，`_fit_size`）
- 每岛 **统一框 B = 蓝框组并集 ±6px**（`build_union_overrides["bbox"]`，image 坐标、不夹紧）；无蓝框岛 B = 岛 bbox
- 擦字范围、布局边界、合成裁剪、面积比基准**全部 = B**（overlay 中一个框）
- 译文块以 B 为布局边界，块中心 = B 中心

### 5.2 面积比（唯一算法）
- `ratio = A_text / A_B`（**唯一基准 = 统一框 B 面积**），目标 **60% – 110%**
- 达标 → `typeset`；不达标 → `typeset_pending_review`（仅质量信号）
- 字号从大到小搜索，取首个"块 ⊆ B + 比例达标"的字号；无达标字号取比例最大者

### 5.3 硬上限（块必须完整在 B 内）
- 块宽/高 **≤ B 宽/高** 且块四边必须在 B 内，否则该字号不选 —— 文字永远完整显示，不截断
- **放不下的岛保留原图**：嵌字前先做一次布局预排（`_unfittable_island_ids`），
  判定"连 12px 也放不下、只能被裁"的岛从擦字与嵌字集合里剔除 —— 擦白了框却写不出
  可见的字，比保留原文更差（实测 丙书 有 17 个这种岛，框只有 8~41px）
- 字号可降到 **12px 绝对下限**（`MIN_FONT_SIZE_MIN`）以适配 B；仍放不下 → 12px 渲染，允许超出到 **150% 包络**（`MAX_GROUP_BBOX_OVERFLOW_RATIO=0.5`，标记 pending_review），再不行则渲染并裁剪（绝不把擦掉的岛留白）
- **列间距/行距永不为 0**（`max(1, 字号//4)`），不靠压扁间距硬塞

### 5.3.1 过大单框不参与合并（编组规则的一部分，见 5.0）
- 判据 = 短边 > `--sfx-min-side-ratio`（默认 1.5）× 同方向基准 **且 > 页短边 × 8%**
  （绝对下限与采集期剔除大拟音同形）；命中的框**各成一岛、绝不与相邻列合并**
  （`oversized_box_ids` + `_group_boxes_by_overlap(no_merge_ids=...)`）
  - **下限是必需的**：只看倍数时，小字页面（实测 丙书 页 97 常规字号 22px → 门槛 33px）
    会把页上几乎所有框判成过大 → 全压到 30px、还互相不合并（封顶框 23% → 8%）
- 原因：这种框在原图里就是独立的超大文字（拟声/效果字），并进竖排组会把组的外接矩形
  与字号一起撑大（实测页 7：3 框并成一组 → 131px 巨型字号；拆开后降到上限内）
- 不合并之后它天然满足"单框岛"条件 → 字号被 `--sfx-max-font-size` 封顶
- **改这条规则后必须重建 dataset**：`_stage2_dataset` 在 `islands.json` 比 `capture.json`
  新时会跳过重建（防止无意义重算）。做法：删 `dataset/*/islands.json` + `selection_blocks.json`
  再跑 `--process`（不用 `--overwrite`，那样会把译文/读数一起清掉）

### 5.4 字号约束（glyph 限制，扩框版）
- **首选下限 28px**：优先使用 ≥28px；28px 放不下时继续降到 12px 绝对下限
- **竖排**：字号（=字宽）≤ 蓝框**扩 6px 后**的最窄宽（`glyph_width_limit` = min 扩框宽）
- **横排**：字号（=字高）≤ 蓝框**扩 6px 后**的最矮高（`glyph_height_limit` = min 扩框高）
- **整页字号上限 `--page-font-ceiling-ratio`（默认 1.5）**：译文字号 ≤ N × 该页常规字号
  （= 该页蓝框短边中位数）。每个岛各自"尽可能撑满"会让同一页出现 2 倍级差
  （实测页 7：14 字组 131px vs 该页基准 66px）；只封顶不放大
- **拟声词/过大单框/短原文额外封顶 `--sfx-max-font-size`（默认 30px）**：判定依据是**日文原文**命中
  `text_rules.is_sfx_like`（纯假名/符号 ≤6 字），判定集合由上游 `page_texts` 传入
  （`typeset_island_page(..., sfx_islands=...)`），**不靠译文猜**。只封顶不放大
  （生效值 = min(框自适应值, 上限)）：拟声框原图里又大又粗，字号跟框走会嵌出一个
  60~120px 的中文拟声盖过对白；正常台词不受影响

### 5.5 对齐与列位
- **竖排**：列内字自上而下，**同一句（同一岛内）各列顶端对齐**（共享顶边）；列从右到左、以 B 中心整组居中均分（**不跟随各蓝框位置**）
- **横排**：行内字自左而右，**各行左对齐**（短行不居中偏移），块整体以 B 中心水平居中
- 中心容差：横向偏移 ≤ B 对角线 5%（竖排只查横向，因列已顶端对齐）

### 5.6 样式（TypesetConfig）
- 黑字白描边：stroke = max(1, 字号//10)
- 字距 = 字号//16；列距/行距 = 字号//4
- 字体 msyh.ttc；可选 `background_color`（每字后画纯色底块）
- **符号字体回退（`_FontChain`）**：msyh 缺字形时按字符回退
  `seguisym.ttf`（Segoe UI Symbol，覆盖 ♡♥♪♫❤∀）→ `seguiemj.ttf`。
  缺字形判定 = 把字符渲染成位图与私用区字符 `\ue000`（必然是豆腐块 □）的位图比较，
  **逐字符**取字体（`draw.text` 一次一个字符）——不是整串一个字体，
  所以"夹紧♡"会是 msyh 画中文、seguisym 画 ♡。纯中文字体下 ♡ 会渲染成 □，
  这是"原文有 ♡ 但成品是方框"的根因（不是 OCR 也不是翻译问题）


## 六、数据契约

```
outputs/oneclick/{书名}/
├── capture/{page}/capture.json          # 选区 + 元数据 (bluebox_masked 前身)
├── check/{page}.bluebox_check.png       # 采集核对图
├── dataset/{page}/
│   ├── islands.json                     # 岛（蓝框分组聚合；删除即按新规则重建）
│   ├── selection_blocks.json            # 蓝框 + 组
│   ├── original.jpg                     # 原图
│   └── bluebox_masked.png               # 填黑图 (OCR 输入)
├── translation_cache/                   # 本地翻译缓存 (signature 校验)
├── translation_batch_cache/             # AI 整页批量翻译缓存
├── ocr_readings.json                    # OCR 读数缓存（非空读数；重跑免 OCR）
├── translations.json                # 书级译文落盘 {page: {island_id: 译文}}（翻译/全链写，嵌字读）
├── typeset/{page}.typeset.png           # 嵌字图
└── {书名}.pdf                           # 最终产物
```

## 七、已知坑（LESSONS 补充）

1. **manga-ocr 无检测能力**：无自带文本检测，必须靠蓝框定位裁剪。整页/纯色图会幻觉（如纯黑区读出 "それでも、"）——所以无框岛绝不进 OCR
2. **union 面积基准**：掩膜值是 255，`mask.sum()` 会放大 255 倍 → 必须 `np.count_nonzero`
3. **无框岛擦除泄漏**：`erase_island_image` 默认擦全部岛 → 嵌字前必须构造"有框岛子集" dataset
4. **填黑图裁剪**：蓝框裁剪从填黑图取，但**列位/字号约束用原始蓝框**（不扩 pad）——两者不要混用
5. **窗口遮挡**：固定平铺（display-scale 100）行距必须 > 窗口高，否则 Photos 弹出菜单会遮挡相邻窗口；
   缩放布局由 auto-fit 保证矩形互不重叠、不越界，线程数 = 实际窗口数

## 八、OCR 引擎切换与扩展

> 引擎分两个层面，互不替代：**采集期**引擎用文字位置"量"出列结构（骨架），
> **stage2** 引擎读蓝框文字（内容，直接进翻译）。

### 8.1 采集期引擎（`photos_capture.py`）
- 参数 `--ocr-engine {auto,windows-ocr,paddle}`（GUI 设置页「OCR 引擎（采集）」）
- `auto`（默认）= Windows OCR 优先，读不出/不可用时 Paddle 兜底
- 强制指定某引擎：不可用或读不出 → **抛 RuntimeError 报错停止**（不回退）
- 统一入口 `_run_masked_ocr(masked_path, engine, paddle_device) -> (predictions, engine_used)`；
  逐列重试链 `_retry_columns_with_crops(..., ocr_engine)` 跟随所选引擎
- **接入新引擎**：写一个 `_run_<engine>_masked(path) -> list[dict]`，返回统一
  预测 schema `{polygon, text, confidence}`，在 `_run_masked_ocr` 注册 +
  batch parser 的 `--ocr-engine` choices + GUI 下拉框加一项即可；
  下游分组/排版流水线（`_group_paddle_columns` → `_finalize_paddle_column`）零改动
- 注：GUI 一键流程当前走 `--frames-only`（只拍帧+蓝框，不做遮罩 OCR）；
  `--ocr-engine` 在完整 fast 路径（不带 `--frames-only`）生效

### 8.2 stage2 读字引擎（`one_click_translate.py`）
- 参数 `--stage2-engine {manga-ocr,ai-ocr}`（GUI 设置页「读字引擎（嵌字前）」），默认 manga-ocr
- `manga-ocr`：本地子进程读蓝框（配置了 AI OCR 时 AI 仍补读 misses，历史行为不变）
- `ai-ocr`：**全部框**走 AI 视觉模型；未配置 `--ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key`
  → 抛 RuntimeError 报错停止；视觉模型读空的框**回退 manga-ocr**（本地子进程，一次加载模型）
  并记入 `fallback_ids`，交由 §4.3 的"可疑读数不翻译"规则判定
- 接入点 `run_manga(..., stage2_engine=...) -> (readings, fallback_ids)`；新引擎在此分支注册即可
  - 返回 `fallback_ids` 是**契约的一部分**：调用方靠它区分"视觉模型读的"与"回退读的"
