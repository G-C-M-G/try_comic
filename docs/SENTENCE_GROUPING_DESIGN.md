# 高亮岛内分句与翻译单元设计（AI 可读规范）

版本：2026-08-05
适用项目：manga_localizer（高亮岛内翻译嵌字，几何权威 = `islands.json`）
上游数据：`outputs\exam\dataset33\` 等，只读输入。

> **状态标注（2026-08-07）**：本文档描述的是旧"高亮岛 + Paddle 行"流水线（M1-M4）。
> 现行 one-click 流水线的分组合并/嵌字规则见
> [`ONE_CLICK_ARCHITECTURE.md` §4.6–§5.6](ONE_CLICK_ARCHITECTURE.md)：
> 蓝框分组 = 扩框(±6px)间距 < 20%×较小扩框宽/高 且 重叠 > 40%×较大扩框高/宽；
> 嵌字统一框 B = 蓝框组并集 ±6px；面积比基准 = A_B；块必须完整在 B 内
> （字号可缩至 12px 绝对下限）。本文下述 120% 面积硬上限等旧规则不再生效。

## 0. 目的

给定高亮岛数据集 + 行级 OCR 结果，确定：
- 哪些 OCR 行属于同一个"句"（翻译单元）；
- 句的阅读顺序；
- 无法自动判定时如何进入人工核对。

## 1. 术语

| 术语 | 定义 |
|---|---|
| 岛 (island) | `islands.json` 中的高亮区域，几何权威 = `image_bbox`（可选 `image_polygon`）。一个岛 = 一个台词/文字容器。 |
| OCR 行 (line) | 引擎输出的最小文本单元，带多边形/bbox 与文本（Windows.Media.Ocr 无置信度；Paddle 有）。 |
| 列片段 (column) | 岛内按几何聚类合并的纵向/横向文本行集合，是"句"的下级单位。 |
| 句 (sentence) | 岛内按 2.3 拆出的语义完整片段，是翻译单元内的**子结构**（含顺序、句尾标点、`column_ids` 映射），不是独立翻译请求。默认整个岛 = 1 句；出现强句边界信号时拆为多句。 |
| 翻译单元 (translation unit) | 一次 Sakura 请求的原文；**与岛一一对应**（整岛一段总文本），内含 `sentences` 子结构，记录 `unit_id -> island_id + column_ids` 映射。 |

## 2. 分层判定模型

"哪些文本是一句"分为三层，逐层进行，任何一层失败都进人工核对而不是静默猜测。

### 2.1 第一层：行 → 岛归属（容器判定）

判据（按优先级）：

1. OCR 行中心点 `(cx, cy)` 落在岛 `image_bbox` 内（含 2px 容差）；
2. 有 `image_polygon` 时，中心点必须同时满足 point-in-polygon；
3. 一行同时落入多个岛（重叠岛）→ 取中心点最近者，其余记 `conflict`。

规则：
- 无岛归属的行（中心点在任何岛外）→ `unassigned`，进人工核对队列，不参与翻译；
- 岛内无任何 OCR 行的保留岛 → `island_empty`，进人工核对（可能是 SFX 或识别失败）。

### 2.2 第二层：行 → 列片段（几何聚类，竖排为主）

判定标准（复用 `try_comic/photos_capture.py:_group_paddle_columns` 已验收阈值）：

- 竖排（主方向，漫画默认）：两行并入同一列，当且仅当
  - x 带重叠率 `overlap / min(w1, w2) >= 0.45`；
  - 垂直间隙 `gap <= max(30, 0.45 * min(h1, h2))`；
  - 高度比 `max(h1,h2) / min(h1,h2) <= 1.6`。
- 横排对称版本：y 带重叠率 >= 0.45、水平间隙同式、宽度比 <= 1.6。
- 列方向判定：bbox 高 >= 宽 → 竖排；否则横排。
- 墨迹投影空隙切分（`_split_columns_by_vertical_gaps`）：列 bbox 内竖投影存在高度 >= `max(30, 页宽/30)` 的全空带 → 拆成两列。防止不同文本块被合并。

列内行序：竖排按 y 升序（上→下），横排按 x 升序（左→右）。

### 2.3 第三层：列片段 → 句边界（语言信号）

句边界信号按强度排序：

| 信号 | 判据 | 强度 |
|---|---|---|
| 强（标点） | 行文本以 `。！？…」】！?…` 或闭合引号（`」』」』`）结尾 | 直接切句 |
| 强（标点） | 行内出现上述句末标点后跟内容 | 在该标点处切句 |
| 中（语法） | 无标点时，行尾匹配助词/结句表：`か・ね・よ・な・ぞ・わ・だ・です・ます・だろう・だろ・の・もん・から` 等；或动词终止形（分词器可判） | 视为句尾，置信 medium |
| 中（引号） | 引号开闭配对计数闭合 | 视为一段结束 |
| 低（无信号） | 无标点、无语法命中 | 整岛视为 1 句，但 `needs_review=true` |

规则：
- 默认**整岛 = 1 个翻译单元**（岛内文本按一段总文本翻译，一次请求）；2.3 的拆句只生成单元内
  `sentences` 子结构与映射，**不产生独立翻译请求**；不做"跨岛合句"；
- 拆句时保留 `sentence -> [column_ids]` 映射，禁止把一列文本拆给两句；
- 一句被拆在多列（列片段连续但无句末信号）→ 必须合并为一句，不得逐列直译；
- 句边界判定结果为 low 的岛进入人工核对队列（阻断翻译）。

### 2.4 阅读顺序

- 岛内：竖排列序从右到左（x 降序），列内上到下；
- 岛间：按格（panel）阅读顺序；无格信息时按"右上角距离"启发式：从页右上角开始，每次取距当前锚点最近的下一个岛（论文 Hinami 2021 同法，报告 91.9% 准确）；
- 全部顺序写入 state，供人工核对调整后重排。

## 3. 翻译单元确定（最终规则）

依据：岛 = 台词容器，整岛文本按一段总文本一次翻译（与 Hinami 2021 Scene-based
translation 结论一致：合理分组比逐句上下文更重要；一句被拆多列/多岛时整段翻译最稳）。

1. 每个保留岛恰好产生 **1 个翻译单元**（整岛一段总文本，一次 Sakura 请求）；
2. 2.3 的拆句结果作为单元内 `sentences` 子结构（每句含顺序、句尾标点、`column_ids` 映射），
   用途仅为审计与译后校验，**不生成独立翻译请求**；
3. 单元原文 = 岛内全部列片段按 2.4 阅读顺序拼接，句间保留句尾标点与分隔标记；
4. 译后校验：译文与原文 `sentences` 数量/句尾标点分布不一致（吞句、合并、重排）→
   记 `needs_review`（沿用现有"未翻译/异常扩写"拒绝逻辑），不写入缓存；
5. 请求 Sakura 时附带整页上下文（其他岛已译/未译文本），**跨岛不合并翻译**；
6. 输出 `unit_id -> island_id -> [column_id]` 可审计映射（对齐现有 `translation_units` 契约）。
7. **上下文增强（可选联网）**：翻译一张图片时，上下文联系（跨页人物/专名消歧、省略主语
   补全、视觉语义等）**允许调用联网 AI** 辅助。该规则与 OCR 引擎无关，对 Windows OCR、
   Paddle、manga-ocr 任一引擎的结果一律生效（OCR 本身始终无网络，联网只发生在翻译阶段）。
   约束：
   - **默认不启用**（保持"全部离线可跑"底线），显式配置后启用；
   - 默认翻译器仍为本地 Sakura，联网 AI 仅作上下文增强，不得替代整段翻译；
   - 每次联网调用在审计中记录模型、用途、原文指纹与返回内容；
   - 缓存签名绑定"联网开关 + 模型版本 + 提示版本"，离线/联网结果不混写；
   - 联网不可用或失败时静默回退到纯本地上下文，不得阻断翻译。

## 4. 人工核对触发条件（任一命中即 pending 阻断）

- 行无岛归属 / 岛内无行（2.1）；
- OCR 低置信：Paddle 平均置信 < 0.6；Windows OCR（无置信度）→ 用跨引擎分歧代替；
- 跨引擎文本分歧：两引擎文本归一化编辑距离 > 阈值（建议 0.3）或内容不一致；
- 句边界信号 = low（2.3）；
- 列聚类方向冲突（同一岛内同时出现竖排与横排且无法判定主方向）。

关闭人工核对（`--no-manual-review`）时全部记为 `bypassed`，不删除风险记录。

## 5. 输出契约（state 扩展）

```json
{
  "island_id": "island_003",
  "image_bbox": [132, 31, 193, 156],
  "direction": "vertical",
  "area_base": 9225.0,
  "units": [
    {
      "unit_id": "u_003_1",
      "source_text": "それは……違うんだ。",
      "sentences": ["それは……違うんだ。"],
      "column_ids": ["c_003_1", "c_003_2"],
      "boundary_signal": "strong",
      "needs_review": false
    }
  ]
}
```

审计字段至少包含：归属结果、聚类阈值命中明细、句边界信号强度、阅读顺序、核对状态、失败原因。
排版审计额外记录：每岛 `area_base`（原文范围面积）、`area_ratio`（排版框面积/基准面积）、
是否使用 5% 放宽及其原因、是否使用 (110%, 120%] 区间及其原因、越界像素数（应为 0）、岛中心偏移。

## 6. 排版约束（岛内嵌字，防文字乱跑）

### 6.1 硬边界

- 译文与擦除区域必须全部落在岛几何内：默认 `image_bbox`（有 `image_polygon` 时按其多边形）。
- 默认每侧 **0% 扩张**；如需放宽最多 **5%**，且必须逐项记录原因并置 `typeset_pending_review`。
- 任何模式不得越岛；排版失败时**不得静默裁切到框外**，记 `translation_failed` 后重试/换策略/人工复核。
- 最终图相对擦字图的所有像素变化必须落在岛的 0% 边界并集内（逐像素审计，沿用现有 pixel_audit）。

### 6.2 文本总大小区间（相对岛内原文文字范围，按面积）

- 基准面积 `A_base` = **岛内原文文字范围面积**：该岛所有 OCR 行 bbox 并集的外接矩形面积（岛内原文占的"范围"）。
- 译文排版框面积 `A_text` = 所有译文字形外接矩形（排版框）的面积。
- 面积比 `r = A_text / A_base`：
  - 正常区间：**60% ≤ r ≤ 110%**（min_ratio 默认 0.6，可配置）；
  - 放宽区间：(110%, **120%**] 允许使用，但必须记录原因并置 `typeset_pending_review`；
  - 硬上限：**>120% 禁止**，直接判 `translation_failed`；
  - 下限：`r < min_ratio`（译文太稀疏/太小）判 `typeset_pending_review`，先放大字号再试。
- 调整优先级：先改字号 → 再调字距/行距 → 最后才考虑放宽区间。
- **硬边界不变**：译文排版框必须全部落在岛 bbox 内（6.1：默认 0% 扩张，≤5% 放宽需记录+核对）。`r` 上限与岛 bbox 冲突时（原文范围边距不足 10%），优先缩小字号；仍不满足 → `translation_failed`，不得越岛。
- 岛中心锁定（`preserve_unit_center` 的岛版）：原文列并集中心与译文列并集中心必须一致，防止文字整体漂移。
- 同一翻译单元的所有正文列使用相同基础字号；句末连续 `……`、`！？`、`！！` 可较小字号，但最多占一个排版字位（沿用现有规则）。

### 6.2a 排版双模式（M4 落地）

- **模式 1（列位跟随）**：译文列占用原 Paddle 列 x 中心（右→左），列内垂直居中于原列 y 范围；
  `r = A_text / A_base_union ∈ [min_ratio, 110%]`（A_base_union = 岛内 Paddle 行框并集面积，掩膜统计）。
- **模式 2（外接矩形，r < min_ratio 触发）**：译文为居中竖排块（多列到岛宽），基准换成行集
  外接矩形面积 `A_rect`，加大字号使 `A_text/A_rect ∈ [min_ratio, 110%]`；中心锁定不变。
- 纯竖排（右→左）；字间距 `tracking=字号//16`、列间距 `column_gap=字号//4`（可配）。
- 全部嵌字白描边（`stroke_width=max(1, 字号//10)`，颜色可配默认纯白）。
- 无 Paddle 行岛：跳过面积比（审计 n/a），只查不越岛 + 中心锁定。

### 6.3 状态语义

| 面积比 r | 状态 |
|---|---|
| [min_ratio(默认0.6), 110%] | 正常，不标记 |
| (110%, 120%] | `typeset_pending_review`（命令返回非零等待人工确认） |
| r < min_ratio 或 >120% | `typeset_pending_review`（先按模式 2 加大字号再判；>120% 判 `translation_failed`） |
| 越出岛 bbox（含 5% 放宽未记录） | `translation_failed` |

### 6.4 验收标准

- 所有保留岛：译文全部在岛内；岛外像素变化 = 0。
- 所有岛面积比 `r ∈ [min_ratio, 110%]`；出现 (110%, 120%] 的岛均有原因记录与核对标记；
  无 `>120%` 或越岛项。
- 重复运行面积比与位置可复现；只改译文重嵌时不改变岛几何与面积基准。

## 7. 与现有代码的复用映射

| 本文档 | 已有实现 |
|---|---|
| 2.2 列聚类 | `try_comic/photos_capture.py:_group_paddle_columns`（阈值已验证）、`_split_columns_by_vertical_gaps` |
| 2.1 行输入 | `try_comic/windows_ocr.py:windows_ocr_predictions`（Windows OCR）、Paddle 子进程接口 |
| 3 整段（整岛）翻译单元 | `manga_localizer` 现有 `translation_units` 整句契约与 `preserve_unit_center` |
| 3.7 上下文增强（可选联网） | 现有 translator 整页上下文 + 缓存签名机制（扩展"联网开关 + 模型版本"绑定） |
| 4 人工核对 | `manga_localizer` 现有 `pending/bypassed` 队列机制 |
| 6 排版约束 | `manga_localizer` 现有 typesetter（字号/字距/方向/描边）、pixel_audit（越界并集审计）、`layout_bbox` 边界体系（需把几何源从列换成岛） |

---

# 附录 A：引擎能力核查与出入（PaddleOCR / manga-ocr）

核查方式：官方文档（PaddleOCR PP-StructureV3、kha-white/manga-ocr README）+ 本机安装代码
（`paddle_venv/Lib/site-packages/paddlex` 3.7.2、`manga-image-translator` 本地 fork）。

## A.1 PaddleOCR（paddleocr 3.7.0 / paddlex 3.7.2）

### 能力清单

| 能力 | 有/无 | 说明 |
|---|---|---|
| 行级检测+识别+置信度 | 有 | PP-OCR 标准流水线输出（bbox/text/score） |
| 句子（句边界）判断 | **无** | 识别模型不输出任何句级信息 |
| 段落级合并 | 有（文档场景） | PP-StructureV3 Markdown 转换器的段落连续性检测 |
| 阅读顺序 | 有（文档场景） | XY-Cut 增强布局解析（`layout_parsing/xycut_enhanced/`）+ PP-DocLayoutV2 学习式 reading-order 头 |
| 版面元素分类 | 有 | 20+ 类（text / paragraph_title / table …） |

### 段落合并的实际判据（本机源码 `layout_parsing/utils.py:get_seg_flag`）

相邻两个 text 块判为"同段连续"（不换段）当且仅当同时满足：

1. 前块末尾边距小：`|end_coordinate - seg_end_coordinate| < 10`；
2. 当前块起始边距小：`start_coordinate - context_left < 10`（左边距对齐）；
3. 前块行数 > 1；
4. 阅读方向边缘距离 `edge_distance < max(prev_width, cur_width)`。

### 与本文档的出入

| 出入点 | Paddle 行为 | 本文档要求 | 结论 |
|---|---|---|---|
| 句边界 | 无 | 2.3 标点+语法判据 | Paddle 不提供，2.3 必须自研 |
| 段落=句子 | Paddle 段落是**版面**概念（缩进/边距/行数），与语法无关 | 句 = 语法完整单元 | 概念不等价，不能直接替换 2.3 |
| 段落判据适用性 | 面向文档：统一左边距、多行块 | 漫画竖排岛内：无统一边距、字距≈字符距 | 判据在岛内基本不成立，不可直接复用 |
| 阅读顺序 | 全局版面 XY-Cut（块→列→行） | 岛内右→左列序 + 右上角距离 | 岛内列序 Paddle 不保证；可参考其 XY-Cut 思想做岛间排序 |
| 可复用部分 | 行级几何 + 置信度、块级归属 | 2.1/2.2 输入、4 低置信信号 | 直接复用 |

## A.2 manga-ocr（kha-white）与本地 MITS 集成

### A.2.1 独立 manga-ocr

| 能力 | 有/无 |
|---|---|
| 端到端识别（输入=裁剪后的文本区图） | 有，输出纯字符串 |
| 多行一次识别 | 有（整气泡一次 forward） |
| 检测 / 行几何 / 置信度 / 句信息 | **全无** |

官方 README 明确：模型"总是尝试识别"，无文本时也可能"dream up"逼真的句子。
⇒ manga-ocr 自身对"哪些文本是一句"**零判断能力**；它假定调用方已经把一句/一段裁好。

### A.2.2 本地 MITS（zyddnys/manga-image-translator fork，`try_comic/manga-image-translator`）

分组能力在管线**合并阶段**（非模型内）：

- `utils/generic.py:quadrilateral_can_merge_region`：两行并入同一区域，当且仅当
  - 多边形距离 `dist <= 2 * char_size`（discard_connection_gap=2）；
  - 字号比 `max/min <= 1.5`；长宽比方向兼容（ratio=1.9）；
  - 角度差 < 15°；轴对齐时：字符间距 < 0.6×字号 且中心对齐 < 1.5×字号。
- `textline_merge/split_text_region`：两框合并当距离 `< (1+γ)×字号`（γ=0.5）且角度差 `< 0.2π`；多框用 MST 距离标准差阈值 `max(0.3×字号+5, 5)` 切分。
- 本地定制（commit `269588d`）：`render_segments`/`translation_segments` 把"句段"贯穿 OCR→翻译→渲染，保留段级映射。

### 与本文档的出入

| 出入点 | MITS 行为 | 本文档要求 | 结论 |
|---|---|---|---|
| 句判断 | 无（合并是纯几何，无标点/语义） | 2.3 语言信号 | 2.3 必须自研，双方一致地缺失 |
| 合并阈值 | 距离 ≤ 2×字高、中心对齐 ≤1.5×字高 | gap ≤ max(30, 0.45×行高)、x 带重叠 ≥45% | 阈值体系不同但同属几何聚类；竖排漫画建议以本项目已验证阈值为准，MITS 阈值可作对照 |
| 阅读顺序 | 帧分割 + 区域排序（`utils/sort.py`） | 2.4 右上角距离 | 思想一致，实现可参考 |
| 置信度/核对 | manga-ocr 无置信度 | 4 低置信触发核对 | 用 manga-ocr 时核对只能靠跨引擎分歧或人工抽样 |
| 段级映射贯穿 | 已有 `render_segments`（本地定制） | 第 5 章 state 契约 | 目标一致，实现可参考 |

## A.3 结论（2026-08-05 实测更新：双引擎架构）

1. **Windows OCR 已从默认路径移除**（识别率差：`ョドーシ〕が`、`、た、か。らこそ` 等误读；
   代码保留在 `island_ocr.py` 但 CLI 不再启用）。
2. **文本权威 = manga-ocr**（整岛一次识别，无几何/置信度）；**几何权威 = Paddle 岛级裁剪**
   （`paddle_ocr_worker.py --crops`，raw/raw2x/clahe/otsu 四变体择优，提供行/列框与置信度）。
3. **交叉验证**（`cross_validate.py`，10 页实测 44 通过/18 REVIEW/5 微岛舍弃）：
   - 一致性 = 归一化（去标点）后 Paddle 片段**包含于** manga 文本（顺序以 manga 为准，
     Paddle 片段乱序不算分歧）；片段长度 ≥2 才算证据，全是短片段时回退编辑距离；
   - 分歧判定：归一化编辑距离 > 0.3 → REVIEW（manga 文本为准）；
   - Paddle 行置信 < 0.6 → 低置信 REVIEW 佐证；
   - manga 空结果岛 → Paddle 兜底文本 + 必 REVIEW。
4. **strong 切句置信度**：manga 文本（权威）句末标点直接切（无置信度概念）；
   Paddle 兜底路径标点切句要求所在行平均置信 ≥0.90，否则降级 REVIEW；
   连续标点串（`！！`）只在一串末尾切一次。
5. **微岛标准**：`image_area < 800px²`（可配置 `min_island_area`）→ 舍弃
   （不擦字/不翻译/不嵌字，报告记录 `dropped_tiny`；10 页实测 5 岛被舍弃）。
   实现：`island_contract.filter_tiny_islands()` 返回过滤后的数据集（effective hash 重算，
   缓存签名绑定保留岛集）；`island_erasure.erase_island_image(min_island_area=...)` 跳过
   微岛并记录。
6. **精确度开关**（`--precision local|online`，默认 local）：两种模式下 OCR 架构相同
   （manga 文本 + Paddle 几何）；local = 整岛原文交本地 Sakura；online = 整岛原文交
   OpenAI 兼容接口（base_url/api_key/model 可配，口语化/上下文优先，生僻词交 AI 自由发挥）；
   实现：`manga_localizer/island_translate.py`——整岛=1 翻译单元、复用 `TranslationCache`
   （签名绑定 precision+模型+提示版本）、online 失败静默回退 local 并记录、拒绝空译/
   原样输出（沿用 `_validate_translation` 校验）。
7. 分句 2.3 v1（标点表 + 助词/结句词表）已实现并实测（33 页：strong 4/particle 3/low 11，
   连续标点修复后 strong 边界正确）；后续统计无标点句命中率后再决定是否引入分词器。
8. **strong 切句置信度**：manga 文本（权威，无置信度）句末标点直接切；Paddle 兜底路径
   标点切句要求所在列平均置信 ≥0.90（`split_sentences(strong_confidence=0.9)`），
   不满足则不切并 `needs_review`；连续标点串（`！！`）只在一串末尾切一次。
9. **kept-source 约定（M3 生效）**：翻译时每个岛最多尝试两轮（contextual → concise 精简提示）；
   两轮都失败（空/原样输出/提示泄漏/假名残留）即判定为**符号或不可译拟声**（如 `ス`、`の．．．`、
   `．．．？`、`！！？`）→ **译文=原文保留**，绝不让岛留空；该判定写入决策缓存
   （`{signature}.kept.json`），后续运行直接命中不再请求模型。输出契约：
   `status=kept-source` + `failure_reason` + `kind=symbol`（纯标点/符号时）。M4 排版时
   kept-source 岛**仍须嵌字回原符号**（擦字阶段已擦掉字形），按 SFX 样式（黑字白描边）绘制。
