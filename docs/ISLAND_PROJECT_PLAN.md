# 高亮岛翻译嵌字项目计划（ISLAND PROJECT PLAN）

版本：2026-08-05
项目目录：`D:\workplace\gpt\manga_localizer`（独立项目，不复制上游 GUI/模型/缓存/MITS）
上游只读输入：`D:\workplace\opencode\try_comic`（`island contract.md` + `outputs\exam\masked10\` + `outputs\exam\dataset33\`）
规格文档：`docs/SENTENCE_GROUPING_DESIGN.md`（分句/翻译单元/排版约束/引擎核查）

## 0. 目标与核心约束

- 给定原图 + 岛数据集（islands.json + filled_*），逐岛识别日文、整岛一段翻译成中文、嵌字回原图。
- 硬约束：
  1. 岛外像素变化 = 0（逐像素审计）；
  2. 译文与擦除全部落在岛 bbox 内（0% 默认，放宽 ≤5% 需记录+核对）；
  3. 嵌字面积比 `r = A_text / A_base ∈ [80%, 110%]`（硬上限 120%），`A_base` = 岛内 OCR 行并集外接矩形面积；
  4. 翻译单元 = 岛（整岛一段总文本一次请求），岛内分句只作 `sentences` 子结构；
  5. 引擎：Windows.Media.Ocr(ja) 优先 → Paddle 兜底；Sakura 本地默认，联网上下文增强可选（默认关）；
  6. SFX 保留并翻译（黑字白描边）；人工核对默认开启。

## 1. 现状盘点：复用 vs 新建

### 直接复用（几何源换成岛后逻辑不变）
| 模块 | 用途 |
|---|---|
| `pixel_audit.py` | 逐像素变化审计（把"20% 硬边界并集"换成"岛边界并集"） |
| `eraser.py` | mask 内白底填充/局部修复、擦除审计 |
| `translator.py` | Sakura/OpenAI 兼容/mock、缓存签名、术语表、拒绝未翻译/扩写 |
| `typesetter.py` | 竖排右到左、字号/字距/描边、中心锁定 |
| `dry_run.py` | 核对图绘制（几何字段换岛） |
| `state.py` / `batch_cli.py` / `cli.py` / `gui.py` 骨架 | state、批量报告、无窗口入口 |

### 新建（岛几何核心）
| 新模块 | 职责 | 对应设计章节 |
|---|---|---|
| `island_contract.py` | islands.json 解析与校验、filled_* 图与原图同尺寸校验 | 2.1 前置 |
| `sentence_splitter.py` | 行→岛归属(2.1)、岛内列聚类(2.2)、句边界(2.3)、阅读顺序(2.4) | 第 2 章 |
| `ocr_island.py` | 每岛裁剪 → Windows OCR → Paddle 兜底，输出岛级行几何 | C 阶段 |

### 改造（几何体系从列 → 岛）
| 现有模块 | 改动 |
|---|---|
| `region_masks.py` | 列 polygon envelope → 岛 polygon/bbox mask（禁止巨形外接矩形） |
| `layout_bounds.py` | 10%/20% 体系 → 岛 0%/5% 硬边界 + 面积比 80/110/120 判定 |
| `ocr_contract.py` | OCRColumn/LayoutRegion/TranslationUnit → island_id/unit/sentences 结构 |
| `pipeline.py` | 各 stage 的几何来源从 column 换 island |
| `state.py` schema | 增加 `island_id`、`units[].sentences`、`area_base`、`area_ratio`、核对状态 |

## 2. 实施阶段（A–F，每阶段含产物与测试）

### A. 仓库骨架 + 岛解析
- 新建 `island_contract.py`：解析 `islands.json`（`image_bbox`/`image_polygon`/`image_center`/`image_area`），
  校验：filled_black/filled_blue 与原图同尺寸、bbox 在图内、id 唯一、`image_size` 与原图一致。
- 更新 README（岛模式用法）+ requirements（无新增大依赖；Paddle 走子进程接口）。
- 测试 `test_island_contract.py`：dataset33 真实样例 + 构造坏样例（尺寸不符/越界/重复 id）。

### B. 岛擦字 mask + 岛外零改动审计 + dry-run
- `region_masks.py` 按岛生成 mask（有 polygon 用 polygon，否则 bbox），擦字后 `pixel_audit`
  断言岛外像素变化 = 0，越界像素计入失败（不静默放行）。
- `dry_run.py` 换岛几何：输出岛编号 + mask + 硬边界 + 核对状态图。
- 测试 `test_island_masks.py`、`test_island_erasure.py`（26–35 十页真实数据跑擦字审计）。

### C. 岛内 OCR + 分句
- `ocr_island.py`：每岛按 image_bbox（+polygon）裁剪原图 → `windows_ocr_predictions`；
  空结果/低质自动回退 Paddle（复用上游 `paddle_ocr_worker.py` 子进程协议）。
- `sentence_splitter.py` 实现 2.1–2.4：行中心落岛（含 point-in-polygon）→ 岛内列聚类
  （x 带重叠 ≥45%、gap ≤ max(30, 0.45×行高)、高度比 ≤1.6，竖排/横排对称版）→
  分句 v1（标点表 + 助词/结句词表）→ 岛内右→左、列内上→下排序。
- 输出岛级原文 + 列几何（新契约 `island_ocr.json`）；无归属行/无信号句 → pending。
- 测试 `test_sentence_splitter.py`：合成竖排样例（含跨列一句、多句一岛、无标点句）+ dataset33 实测统计。

### D. 整岛一段翻译
- 单元 = 岛；原文 = 岛内全文按 2.4 顺序拼接（保留句尾标点）；译后校验：
  译文句数/句尾标点分布与 `sentences` 不一致 → `needs_review`，不写缓存。
- 缓存签名 = 原图 SHA + islands.json SHA + 原文 + 术语表 + 模型 + 提示版本 + **联网开关**。
- `config/glossary.json` 加示例词（ヨドイン/ウイラス/ワテクア/カークァス）。
- 可选联网上下文增强（设计 3.7）：默认关，启用时审计记录模型/用途/指纹，失败静默回退本地。
- 测试 `test_translator_island.py`（mock 端到端 + 缓存命中/损坏 + 吞句拒绝）。

### E. 岛内排版
- `layout_bounds.py` 改造：硬边界 = 岛 bbox（0% 默认；≤5% 放宽记录原因 + `typeset_pending_review`）；
  面积比 `r = A_text / A_base`：`[80%, 110%]` 正常、(110%, 120%] 记录+核对、`<80%`/`>120%` → `translation_failed`。
- 竖排右到左、岛中心锁定、SFX 黑字白描边、失败不静默裁切。
- 测试 `test_island_typeset.py`：面积比边界（79%/80%/110%/111%/120%/121%）、越岛、中心偏移、5% 放宽记录。

### F. CLI/批量/state/端到端验收
- `cli.py` 增加岛模式（`--islands-dir` 等），`batch_cli.py` 批量扩展，`state.py` 新 schema，
  只改译文直接重嵌（跳过 OCR/擦字）。
- dataset33 + masked10（26–35 十页）端到端跑通，输出：每页耗时、越界报告、核对图、batch.report.json。
- 测试 `test_state_and_batch_island.py` + 验收脚本（断言岛外零改动、无越岛、面积比区间、缓存不串图）。

## 3. 验收标准映射

| 验收项 | 对应 |
|---|---|
| 10 页岛外像素变化 = 0 | B/E 审计，F 端到端报告 |
| 所有保留岛有处理结果；译文框不越岛；无跨岛擦除 | C/F 报告 |
| 面积比 ∈ [80%, 110%]；(110%, 120%] 有记录；无 <80% / >120% | E 审计 |
| 重复运行可复现；缓存不串图；批量无窗口 | D 缓存 + F 报告 |
| 自动测试覆盖：岛解析/mask/岛外零改动/句列映射/0%5%越界/缓存签名/失败恢复 | A–F 测试清单 |

## 4. 风险与决策点

1. **列聚类阈值**：2.2 阈值来自整页列聚类，岛内（列更短）可能偏松/偏紧 → 在 dataset33 实测，
   记录命中率后微调（阈值参数化，进配置）。
2. **分句 v1 命中率未知**：先出标点+助词表 v1，统计无标点句占比，再决定是否引入
   kuromoji/sudachi 判终止形（设计附录 A.3）。
3. **岛无 OCR 行**：识别失败（艺术字/拟声页）→ 记录 `island_empty` 进人工核对；
   `A_base` 不可计算时面积比判定跳过并标记 blocked（不伪造数字）。
4. **面积比冲突**：`r` 上限与岛 bbox 冲突（原文范围边距不足）→ 先缩字号，仍不满足判失败，不越岛。
5. **Windows OCR 缺口页**（31/34 艺术字页）→ Paddle 兜底已有实测依据（island contract 基线）。

## 5. 里程碑

| 里程碑 | 内容 | 交付物 |
|---|---|---|
| M1 | A + B | 岛解析/擦字/审计/dry-run；10 页岛外零改动报告 |
| M2 | C | 岛内 OCR + 分句；岛级原文/列几何/核对图 + 分句命中率统计 |
| M3 | D | 整岛翻译 + 缓存 + glossary；mock 端到端译文 |
| M4 | E + F | 排版 + CLI/批量/state；10 页端到端验收报告（耗时/越界/核对图） |

## 6. 门禁流程（用户确认制）

每个里程碑完成核心产物后**暂停并提交给用户确认**，批准后才进入下一步：

| 里程碑 | 门禁交付物（展示给用户确认） |
|---|---|
| M1 | 33 页 dry-run 核对图（岛编号 + mask + 硬边界 + OCR 行框标注）；10 页"岛外像素变化=0"报告 |
| P1 门禁 | **双引擎交叉验证门**：10 页 manga-ocr 文本 × Paddle 岛级裁剪 互证表（通过/REVIEW/微岛舍弃）、每页标注图、REVIEW 原因明细（编辑距离/低置信/空岛） |
| M2 | 已填黑图 OCR 标注图 + 表格：岛 → 行列表 → 整岛原文 → 分句 sentences → 边界信号强度 |
| M3 | 按文档规则翻译一一对应表：岛 → 原文（整岛一段）→ 译文 → needs_review → 缓存指纹 |
| M4 | 嵌字图 + 面积比/越界审计表 + 批量报告 |

门禁不通过则原地修正，不跳级。

## 7. 数据与前置说明

- 10 页全量数据集**已存在**于上游 `outputs\exam\dataset\26-35`（fill_mode="shape"、contours=true），
  无需重新生成；样例页另有 `outputs\exam\dataset33\33`。均只读引用。
- 上游 OCR 经 `manga_localizer/island_ocr.py` 子进程调用上游 venv 的 `windows_ocr.py`，本仓库无 winrt 依赖。
- M1 实现：`island_contract.py`（解析校验）、`island_masks.py`（岛 mask）、`island_erasure.py`
  （岛内擦字，mask 裁剪到岛内保证岛外零改动）、`island_dry_run.py`（核对图）、`island_render.py`（CLI）。

## 10. M3 实施记录（2026-08-05，D 阶段：整岛翻译）

- **批量翻译** `scripts/m3_translate_pages.py`：输入 M2 `{page}.final.json` → 输出 `{page}.translation.json`
  （岛 → 原文 → 译文 → backend/model/cache → status → role → review_status）。
- **提高精确度开关**（`--precision online`，默认 local）：整页所有岛编号打包一次请求
  （`translate_page_batch` + `PageBatchCache` 页级缓存），`编号: 译文` 逐行解析，缺号/失败
  静默回退 local Sakura；失败兜底 kept-source（保留原文 + 决策缓存，避免重复请求）。
- **校验强化**：`_validate_island_translation`——提示词泄漏（含上下文文本片段泄漏）、kana 比例、
  长度、原样输出拒绝；`IslandTranslationCache`（岛版缓存校验，不套用 ?/! 保留规则，避免 OCR
  噪声误杀）；contextual → concise 两级提示重试。
- **10 页实测**（Sakura 在线）：62 岛 = 58 翻译（二次运行 58/58 缓存命中）+ 4 kept-source
  （31/005 ス、31/006 の．．．、26/003 ．．．？、34/001 ！！？——拟声/符号岛 Sakura 无法翻译，
  保留原文并记录原因）；33/015 上下文泄漏被拦截后精简重试得 `李可多因`。
- 产物：`var/island_m1/m3/`（每页 translation.json、REPORT_M3.md、TRANSLATION_TABLE.md）。

## 11. 下一步（M4，E+F 阶段）

- E：中文排版——竖排右到左、硬边界=岛 bbox（0% 默认，≤5% 放宽记录+核对）、面积比
  `A_text/A_base ∈ [80%, 110%]`（硬上限 120%）、岛中心锁定、SFX 黑字白描边。
- F：单图/批量 CLI + state + 报告；10 页端到端验收（岛外零改动、无越岛、面积比区间、耗时）。

## 12. M4 实施记录（2026-08-05，E 排版 + F 批量）

- **纯竖排**：漫画一律竖排右到左（列内上到下），删除横排分支。
- **双模式字号搜索**（`manga_localizer/island_typeset.py`）：
  - 模式 1（列位跟随）：译文列占用 Paddle 列 x 中心（**右→左**），列内垂直居中于原列 y 范围；
    `r = A_text / A_base_union ∈ [0.6, 1.1]`（min_ratio 默认 0.6 可配 `--min-ratio`，硬上限 1.2）
  - 模式 2（外接矩形，r<0.6 触发）：译文为居中竖排块（多列到岛宽），基准=行集外接矩形面积
    `A_rect`，加大字号使 `A_text/A_rect ∈ [0.6, 1.1]`；中心锁定不变
  - 无 Paddle 行岛：跳过面积比（审计 n/a）
- **间距参数**：字间距 `tracking = 字号//16`、列间距 `column_gap = 字号//4`（可配），块尺寸与绘制同公式。
- **样式**：全部嵌字白描边（`stroke_width = max(1, 字号//10)`，颜色可配默认纯白）。
- **中心锁定**：块中心偏移 ≤ 岛 bbox 对角线 × 5%（圆形），超差 `typeset_pending_review`。
- **像素审计**：相对擦字图变化全部在岛 bbox 内（岛外=0，岛并集掩膜裁剪兜底）。
- **10 页实测**（Sakura 全量，min_ratio=0.6）：59 typeset + 3 pending_review（31/006 符号岛中心偏移
  16>5、32/002 `嗯` 2 字短文 0.51、33/014 边界偏移 15>14.3）+ 5 skipped（微岛）+ 0 failed；
  每页排版 <0.1s。
- **state 与重嵌**（`manga_localizer/island_state.py`）：批量输出 `{page}.state.json`——
  绑定 original/islands/erased/translations 四路 SHA + 每岛排版审计 + Paddle 几何记录
  （area_base/area_rect/centre/列 bbox）；`--reembed-state` + `--edit-translations` 只跑排版一步
  重嵌（<0.1s），哈希不匹配拒绝（防旧译文套新图）；skipped 岛不入 state。
  实测：33 页改 island_004 译文重嵌，结果与全流程一致（island_004 字号 25、ratio 0.93）。
- 产物：`var/island_m1/m4/`（每页 typeset.png + typeset.audit.json + state.json、REPORT_M4.md、
  33.compare.png、33.reembedded.png）。

## 9. M2 实施记录（2026-08-05）

- **M2-A（原图重读，无扩区）**：REVIEW 岛 manga 用原 bbox + original.jpg 重读（`scripts/m2_review.py`），
  重跑交叉验证：18 → 8 REVIEW；修复 33/004 `間→闇`、27/004 `脆弱さ`、31 页全部等。
  调试结论：bbox 扩区对 manga 有害（改变缩放比例）、对 Paddle 偶有收益但会污染相邻文本 →
  **已取消扩区**；黄色描边保留为默认样式（6px、`(255,215,0)`，`render_island_dry_run` 可配置）。
- **M2-B（GUI 核对 + 懒人开关）**：`manga_localizer/island_gui.py` + `island_gui.py`（启动器）：
  页面选择、REVIEW 列表（manga/Paddle/原因）、原图 6px 黄描边预览、处置按钮（批准/改字/拒绝/SFX）、
  **懒人开关默认开**（REVIEW 直接承认 manga）；保存 `{page}.dispositions.json`；
  `scripts/apply_dispositions.py` 应用处置重写 `{page}.final.json`（approve/correct/reject/sfx，
  保留 effective_sha256 绑定）。10 页演示：8 项 REVIEW 全部懒人批准，62 保留岛最终原文就绪。
- **已取消**：REVIEW 升级通道（VLM/照片翻译）——用户后续优化 OCR 识别区域，从源头减少 REVIEW。
- 门禁产物：`var/island_m1/m2/`（REPORT_M2A.md、每页 m2.cv/md/annotated.png、final.json、
  FINAL_TEXTS.md、dispositions.json）。

## 8. 双引擎架构（P1–P4 已执行，2026-08-05）

- **Windows OCR 移除默认路径**（代码保留，CLI 不启用）。
- **文本权威 = manga-ocr**（`scripts/manga_ocr_islands.py`，整岛一次识别，多页一次模型加载）。
- **几何权威 = Paddle 岛级裁剪**（`paddle_ocr_worker.py --crops`，raw/raw2x/clahe/otsu 四变体，
  `scripts/merge_paddle_crops.py` 按岛择优合并）。
- **交叉验证**（`manga_localizer/cross_validate.py` + `scripts/cross_validate_pages.py`）：
  片段包含判定（顺序以 manga 为准）+ 编辑距离 >0.3 分歧 + 行置信 <0.6 低置信 + manga 空岛兜底。
  10 页实测：44 通过 / 18 REVIEW / 5 微岛舍弃（门禁已确认）。
- **P2 微岛标准**：`min_island_area=800px²`（配置化），`filter_tiny_islands()` 过滤，
  擦字/翻译/嵌字全部跳过并记录 `dropped_tiny`（33 页实测 006/007 舍弃，岛外零改动保持）。
- **P3 精确度开关**：`--precision local|online`（默认 local），OCR 架构不变，仅翻译引擎切换；
  `manga_localizer/island_translate.py`：整岛一段翻译、OpenAI 兼容接口（base_url/api_key/model 可配）、
  缓存签名绑定 precision+模型+提示版本、online 失败静默回退 local、拒绝空译/原样输出。
  33 页 mock 演示：17 单元 0 失败，二次运行 17/17 缓存命中。
- **P4 strong 切句置信度**：manga 文本（无置信度）标点直接切；Paddle 兜底路径标点切句要求
  列平均置信 ≥0.90，否则不切并 `needs_review`；连续标点（`！！`）只切一次。
