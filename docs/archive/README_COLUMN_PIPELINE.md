# Manga Localizer

独立的日文漫画擦字、整句翻译和中文嵌字项目。上游 schema 4 OCR JSON 是稳定、只读输入；
本仓库不包含旧 GUI、缓存、模型、MITS 或 OCR 实现。

## 当前状态

B、C、D、E、F、G 阶段及首版独立 GUI 已实现：

- 使用交接包 `ocr_schema_v4.json` 做 Draft 2020-12 结构校验，再做 ID、句列映射、几何、
  阅读顺序、人工核对计数等跨字段校验。
- 图片路径按 `--image`、OCR JSON 同目录、显式/约定样本目录依次解析，不使用旧机器绝对路径。
- 人工核对默认开启。`pending` 会保持页面阻断；`--no-manual-review` 只把风险记为
  `bypassed`，不会删除队列。
- 可加载哈希绑定的 `*.ocr.review.json`，应用批准、改字、改框或拒绝决定，并重建完整句与逐列
  布局映射。原 OCR JSON 不会被改写。
- dry-run 为每个 `column_id` 独立生成 polygon mask，并在原尺寸图片上画出 mask、原框、
  10% 排版硬边界、句/列编号和人工核对状态；不擦字、不翻译。
- 20% 边界只能从人工核对后的完整 `image_bbox` 直接计算，不能从 10% 框继续扩张。
- 擦字始终保留独立 `column_id` mask。白底气泡通过深色字形阈值和受限环带中值填充；画面
  文字通过黑字/白描边接触关系生成 mask，再使用无窗口 OpenCV FSR 快速局部修复。
- 最终图只从擦字后结果向组合 mask 内回贴，审计会逐像素确认 mask 外变化数为 0，并记录每列
  mask 哈希、策略、后端、变化像素和失败状态。
- 后续嵌字验收已提供快速抽样基础：中央 20% 优先，整框固定步长兜底；两者均无变化时必须记为
  `translation_failed`，然后重试、切换翻译/渲染策略或进入人工复核。像素检查预算为 60 秒；擦除和嵌字耗时只记录，不作为失败条件。
- 翻译按 `translation_units.source_text` 整句请求，完整 `translation_query` 只作为列完整性参考；同一
  单元不会拆成多列分别直译。请求附带整页前后文，结果再按 `unit_id -> column_ids` 建立可审计映射。
- `Translator` 支持本地 Sakura、任意 OpenAI 兼容接口和离线 mock。Sakura 默认先用上下文提示，发生
  提示泄漏、未翻译或异常扩写时重试并切换精简提示；无关术语目标也会被拒绝，不会写入缓存。
- 术语表/人名表位于 `config/glossary.json`。缓存签名绑定 OCR 有效哈希、完整原文、列映射、整页上下文、
  术语表、模型和提示版本；缓存命中不会调用模型，损坏缓存自动视为未命中。
- Sakura/OpenAI 翻译优先保留完整列标记。标记完整时记录精确语义片段；标记缺失或只保留连续前缀时，
  只接受整句译文，不伪造列级语义映射，由排版器按完整句和 OCR 列容量重新分配。
- 中文排版默认先在 `layout_bbox_10pct` 内缩小字号、调整列内字数和字距。只有显式允许 20% 且 10% 失败时，
  才从人工核对后的完整 `image_bbox` 直接计算 `layout_bbox_20pct`；结果状态为待人工核对。
- 气泡正文使用黑字，画面文字使用黑字白描边；竖排按 OCR 列从右到左、列内从上到下。每列先检查中央
  20%，再固定步长抽样整框；没有实际像素变化时记为 `translation_failed`。
- 最终嵌字图保持原尺寸，并逐像素确认相对擦字图的所有变化均在 20% 硬边界并集内。像素审计限时 60 秒，
  排版耗时不受此限制。
- 每列实际排版框的宽和高均不得小于完整 OCR `image_bbox` 的 80%。同一个完整翻译单元的所有正文列使用
  相同基础字号；句末连续 `……`、`！？`、`！！` 可使用较小字号，但最多只占一个排版字位。
- `preserve_unit_center` 默认开启：完整句原 OCR 列并集与最终排版列并集的几何中心必须完全一致。关闭时
  仍记录中心偏移，不删除审计。
- 完整流水线输出可编辑 `*.state.json`，记录输入哈希、完整句译文、逐列 mask/擦字策略、排版框、字体、
  字号、方向、颜色、描边、溢出、处理状态和失败原因。只改译文时会作废不匹配的旧列片段并直接重嵌。
- `batch.py` 按目录逐页串行运行，不打开窗口、不控制鼠标，并始终输出 `batch.report.json`；串行策略避免
  同时请求本地 Sakura。
- `gui.py` 提供基于 PySide6 的人工核对、单页运行和“OCR订正”页面。订正页通过适配器调用上游
  `photos_capture.py`/`background_ocr.py`，OCR、mask、擦字、翻译和排版算法不在 GUI 事件中重复实现。

四页端到端验收见 [`docs/ACCEPTANCE_2026-08-04.md`](docs/ACCEPTANCE_2026-08-04.md)。C 阶段必须显式
使用 `--erase-only`，避免把未翻译的擦字图误认为最终成品。

## 安装

```powershell
D:\tools\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dry-run

```powershell
.venv\Scripts\python.exe render.py `
  --image samples\images\1.jpg `
  --ocr samples\ocr\1.ocr.json `
  --review samples\ocr\1.ocr.review.json `
  --output artifacts\dry_run\1.dry-run.png `
  --dry-run
```

模板仍含 `pending` 时，dry-run 会正常输出核对图和 `*.audit.json`，但状态为 `blocked`；正式
翻译阶段必须停止。显式绕过仅用于风险可接受的批处理：

```powershell
.venv\Scripts\python.exe render.py --ocr samples\ocr\1.ocr.json `
  --output artifacts\dry_run\1.bypassed.png --dry-run --no-manual-review
```

图例：半透明色块是逐列 mask（颜色区分翻译单元）；彩色实线是 polygon；白色实线是完整
`image_bbox`；黄色虚线是 `layout_bbox_10pct`；橙色框/`P01` 标签表示 pending。图内数字是阅读
序号，完整 `unit_id/column_id` 在同名 audit JSON 中。指定 `--max-layout-overflow 20` 时额外显示
红色点划线 20% 绝对上限，但 dry-run 不代表已经获准使用该边界。

## 擦字

```powershell
.venv\Scripts\python.exe render.py `
  --image samples\images\1.jpg `
  --ocr samples\ocr\1.ocr.json `
  --review samples\ocr\1.ocr.review.json `
  --output artifacts\erase\1.erased.png `
  --mask-output artifacts\erase\1.mask.png `
  --diff-output artifacts\erase\1.diff.png `
  --audit-json artifacts\erase\1.erase.audit.json `
  --erase-only
```

`--erase-strategy auto|solid-fill|inpaint` 可覆盖自动样式策略。输出均为无损 PNG 且与原图同尺寸；
命令不打开窗口、不控制鼠标。存在 `pending` 时默认拒绝擦字，显式 `--no-manual-review` 会继续，
但所有风险仍以 `bypassed` 保留在审计中。

## 整句翻译

默认使用已运行的本地 Sakura（`sukinishiro`）。输出是可审计的翻译候选 JSON，不是最终图片：

```powershell
.venv\Scripts\python.exe render.py `
  --ocr samples\ocr\8.ocr.json `
  --review samples\ocr\8.ocr.review.json `
  --output artifacts\translation\8.translation.json `
  --translate-only
```

离线测试可使用确定性 mock：

```powershell
.venv\Scripts\python.exe render.py `
  --ocr samples\ocr\8.ocr.json `
  --review samples\ocr\8.ocr.review.json `
  --output artifacts\translation\8.mock.json `
  --translate-only --translator mock `
  --mock-translations config\sample_translations\8.json `
  --no-translation-cache
```

`--translation-prompt-style auto`（默认）先使用整页上下文，只有失败才切换精简提示；`contextual` 或
`concise` 可用于对比。若服务不可用或所有尝试失败，命令返回 2，JSON 保留每个失败单元和尝试原因。

`config/sample_translations/` 是四页验收样本的确定性整句备用译文。它们不会替代默认 Sakura；只有显式
选择 mock 时使用，产物会如实记录 backend。每个值仍是一条完整译文，`<n>` 仅用于对齐回 OCR 列。

生产核对译文可使用 `source-bound-translations` 格式，同时绑定 OCR SHA-256、`unit_id` 和完整原文。
OCR 框、分组或原文发生变化时文件会被拒绝，避免把旧译文静默套到新 OCR 上。

## 中文嵌字

```powershell
.venv\Scripts\python.exe render.py `
  --image samples\images\8.jpg `
  --ocr samples\ocr\8.ocr.json `
  --review samples\ocr\8.ocr.review.json `
  --erased-image artifacts\erase\8.erased.png `
  --translation-json artifacts\translation\8.translation.json `
  --erasure-audit artifacts\erase\8.erase.audit.json `
  --output artifacts\typeset\8.typeset.png `
  --audit-json artifacts\typeset\8.typeset.audit.json `
  --typeset-only --max-layout-overflow 10
```

默认只使用 10% 软边界。`--max-layout-overflow 20` 表示允许在 10% 无法容纳后尝试直接 20% 硬边界；
任何实际使用 20% 的列都会记录原因，整页状态为 `typeset_pending_review`，命令返回非零等待人工确认。
排版失败时不会把文字静默裁切到框外。

## 完整流水线与 State

不选择 `--dry-run/--erase-only/--translate-only/--typeset-only` 时，`render.py` 直接执行完整流水线：

```powershell
.venv\Scripts\python.exe render.py `
  --image samples\images\8.jpg `
  --ocr samples\ocr\8.ocr.json `
  --review samples\ocr\8.ocr.review.json `
  --output output\8.translated.png
```

默认生成 `output/8.translated.state.json` 和 `output/8.translated.stages/`。可分别用 `--state-output`、
`--stage-dir` 指定位置。中心锁定默认开启，`--no-preserve-unit-center` 可关闭，但偏移仍写入审计。

编辑 state 中的 `units[].translated_text` 后直接重嵌：

```powershell
.venv\Scripts\python.exe render.py `
  --ocr samples\ocr\8.ocr.json `
  --review samples\ocr\8.ocr.review.json `
  --state output\8.translated.state.json `
  --output output\8.reembedded.png
```

state 会校验 OCR、原图和擦字图哈希。若只改完整译文，旧列片段不能拼回新译文时自动失效；该路径不调用
OCR、擦字或翻译器。

## 批量处理

```powershell
.venv\Scripts\python.exe batch.py `
  --images samples\images `
  --ocr-dir samples\ocr `
  --output-dir output
```

OCR 和 review 默认按 `<图片 stem>.ocr.json`、`<图片 stem>.ocr.review.json` 匹配。每页输出最终 PNG、
state 和阶段目录；成功与失败都保留在 `batch.report.json`。离线 mock 批量验收可增加
`--translator mock --mock-translations-dir config/sample_translations`。

## GUI 与人工核对

```powershell
.venv\Scripts\python.exe gui.py
```

也可以启动时直接打开一页：

```powershell
.venv\Scripts\python.exe gui.py --ocr samples\ocr\1.ocr.json
```

界面加载 schema 4 OCR 后会解析匹配原图，并加载同名 `*.ocr.review.json`；没有 review 时创建一份绑定
OCR SHA-256 的工作副本。人工核对页显示所有风险列、原 polygon/完整框、当前文字、MangaOCR 候选、
PaddleOCR 候选、置信度和原因，支持：

- 批准当前文字和几何；
- 修正文字、坐标输入完整框，或在画布上重新框选；
- 拒绝误检；
- 保持 pending，或经确认后批准全部待定项。

保存使用同目录临时文件校验后原子替换。跨页哈希不匹配、框越界、空修正或决定集合不完整时不会覆盖原
review。运行页可执行 dry-run、完整流水线和已有 state 重嵌；这些任务在后台启动 `render.py`，窗口保持
响应。人工核对默认开启，pending 会阻止正式翻译；关闭时界面明确显示 bypass 数量，风险记录仍写入产物。
10%/20% 边界和整句中心锁定都有独立控件。

GUI 仅在显式运行 `gui.py` 时打开。`render.py` 和 `batch.py` 的无窗口行为不变。界面遵循
`D:\tools\gui-standard\GUI设计规范.md`：220px 侧边栏、页面卡片、teal 设计令牌、原生图标按钮及持久化
亮暗主题；项目内 [`manga_localizer/theme.py`](manga_localizer/theme.py) 是该标准的独立 QSS 落地。

Photos 辅助批量订正的边界、命令和验收见
[`docs/OCR_CORRECTION_ARCHITECTURE.md`](docs/OCR_CORRECTION_ARCHITECTURE.md)、
[`docs/OCR_CORRECTION_USAGE.md`](docs/OCR_CORRECTION_USAGE.md) 和
[`docs/ACCEPTANCE_OCR_CORRECTION_2026-08-05.md`](docs/ACCEPTANCE_OCR_CORRECTION_2026-08-05.md)。
面向下一轮“减少前台时间、保持翻译质量”的任务提示词见
[`docs/OCR_CORRECTION_OPTIMIZATION_PROMPT.md`](docs/OCR_CORRECTION_OPTIMIZATION_PROMPT.md)。

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 目录

```text
manga_localizer/       契约、核对、mask、擦字、翻译、排版、state、批量、GUI 与像素审计核心
schema/                 上游 schema 4 原始结构约束
samples/                交接包四页原图、OCR、review 模板、基准和参考核对图
docs/upstream/          交接包四份冻结文档与 manifest
tests/                  B/C/D/E/F/G 与 GUI 自动测试
render.py               无窗口命令行入口
batch.py                无窗口目录批量入口
gui.py                  独立人工核对与单页运行界面
```
