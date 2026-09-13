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
- **AI OCR 补强（可选）**：manga-ocr 未读出的框（misses）若有配置 `ai_ocr_url/model/key`，
  走 OpenAI 兼容视觉模型（`manga_localizer/ai_ocr.py`）复读，结果同样入缓存

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
- AI 批量缓存 `translation_batch_cache/`（按页批量签名）；本地缓存 `translation_cache/`
  （signature = 源文本 + 上下文 + 术语表 + 提示词版本）
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

- 竖排框对合并 **当且仅当**：**横向间距 < 60% × 较小框宽** 且 **纵向重叠 > 40% × 较大框高**
- 横排框对合并 **当且仅当**：**纵向间距 < 60% × 较小框高** 且 **横向重叠 > 40% × 较大框宽**
- 混合方向对**永不合并**；沿阅读方向堆叠（同列竖排堆叠 / 同行横排并排）**永不合并**（蓝框是可信原子单位）
- 重叠基准取**较大**框维度：避免"矮框只搭住高框一小截"也被合并（较小基准在顶端对齐时恒满足，形同虚设）
- **60% 是替换"含扩边 20%"的重标定**：旧规则在扩框后几何上判，扩边给每个间距
  加了约 2×pad 的**常量**，对小框的相对影响过大；改为原始框的纯比例后是**尺度无关**的。
  实测扫描（图源样张）：0.4~0.6 的编组结果几乎一致，取值不敏感
- 每组 → 独立岛，岛 bbox = **组外接矩形 ±2px**；**每组的翻译嵌字只在组内进行**

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
- 字号可降到 **12px 绝对下限**（`MIN_FONT_SIZE_MIN`）以适配 B；仍放不下 → 12px 渲染，允许超出到 **150% 包络**（`MAX_GROUP_BBOX_OVERFLOW_RATIO=0.5`，标记 pending_review），再不行则渲染并裁剪（绝不把擦掉的岛留白）
- **列间距/行距永不为 0**（`max(1, 字号//4)`），不靠压扁间距硬塞

### 5.4 字号约束（glyph 限制，扩框版）
- **首选下限 28px**：优先使用 ≥28px；28px 放不下时继续降到 12px 绝对下限
- **竖排**：字号（=字宽）≤ 蓝框**扩 6px 后**的最窄宽（`glyph_width_limit` = min 扩框宽）
- **横排**：字号（=字高）≤ 蓝框**扩 6px 后**的最矮高（`glyph_height_limit` = min 扩框高）

### 5.5 对齐与列位
- **竖排**：列内字自上而下，**同一句（同一岛内）各列顶端对齐**（共享顶边）；列从右到左、以 B 中心整组居中均分（**不跟随各蓝框位置**）
- **横排**：行内字自左而右，**各行左对齐**（短行不居中偏移），块整体以 B 中心水平居中
- 中心容差：横向偏移 ≤ B 对角线 5%（竖排只查横向，因列已顶端对齐）

### 5.6 样式（TypesetConfig）
- 黑字白描边：stroke = max(1, 字号//10)
- 字距 = 字号//16；列距/行距 = 字号//4
- 字体 msyh.ttc；可选 `background_color`（每字后画纯色底块）

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
- `ai-ocr`：全部框走 AI 视觉模型；未配置 `--ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key`
  → 抛 RuntimeError 报错停止
- 接入点 `run_manga(..., stage2_engine=...)`；新引擎在此分支注册即可
