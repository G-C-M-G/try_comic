# 下一个项目的启动提示词

把下方“提示词正文”完整发送给新项目中的 Codex。新项目应使用独立目录，不要直接在
`try_comic` 中继续堆叠嵌字实现。

## 提示词正文

```text
我要创建一个独立的日文漫画自动擦字、翻译与中文嵌字项目。

上游 OCR 项目位于：
D:\workplace\opencode\try_comic

OCR 交接包位于：
D:\workplace\opencode\try_comic\backups\2026-08-04_ocr-handoff\ocr-handoff-final.zip

请先完整阅读交接包中的 OCR_交接文档.md、架构说明.md、使用说明.md、OCR_BASELINE.md，
再检查 samples 下 1、8、19、22 四页的原图、最终 *.ocr.json、照片基准和 OCR 核对图。
不要重新开发 OCR，也不要把旧项目的 GUI、缓存和整个 MITS 目录复制进新项目。把 schema 4
OCR JSON 当作稳定只读输入；如果未来需要新 OCR，只通过旧项目 background_ocr.py 的 CLI
或清晰的适配器调用。

目标：给定一张原图和对应 OCR JSON，自动把 OCR 区域逐块擦除，以完整句子翻译为自然、
口语化、符合人物语气的中文，再把中文精准排回对应区域，输出最终翻译图片。支持单图和
目录批量，运行期间不打开窗口、不控制鼠标，模型完整时可以离线运行。

必须遵守的核心规则：

1. 翻译上下文和图像几何分离。翻译必须使用 translation_units.source_text 或带列标记的
   translation_query，不能把一条完整台词拆成若干列各自直译。
2. 擦字必须按 translation_units.layout_regions/columns 中的 polygon 逐块生成 mask，不能用
   完整句子的巨大外接矩形盖住画面。需要合并 mask 时仍要保留 column_id 级审计记录。
3. layout_bbox_10pct 是最终文字允许使用的最大范围，已经包含总宽高 10% 的余量，禁止二次
   扩张。译文放不下时缩小字号、合理换列或进入人工复核，不能静默越界。
4. 人工核对默认开启。若 manual_review.translation_blocked=true，必须暂停该页，把
   manual_review_queue 中的框、两模型候选文字和原因显示给我。我可以批准、改字、改框或
   拒绝误检。只有全部解决后才能翻译。提供“人工核对”开关；关闭时允许继续，但界面和状态
   必须显示 bypassed 数量，不能删除风险记录。
5. 气泡纯色背景和画面线稿采用不同擦除策略。白色气泡可受限填充；线稿、网点和画面文字
   使用局部 inpainting。任何策略都应尽量保持 mask 外像素不变，并输出像素差核对结果。
6. 对带白色描边的画面文字，mask 要包含原字和描边；中文默认用黑字加白描边覆盖。气泡
   正文默认黑字，竖排从右到左。样式判断要做成可替换策略。
7. 翻译默认使用现有本地 Sakura OpenAI 兼容接口 http://127.0.0.1:8080/v1，但实现独立的
   Translator 接口，支持 mock、其他在线模型和缓存。加入术语表/人名表；中文避免日语语序、
   生硬书面语和无意义扩写，保持台词简洁。
8. 输出原图同尺寸的最终图片，以及可重新编辑的 *.state.json。状态至少记录 OCR 文件哈希、
   原图哈希、column_id/unit_id、原文、译文、mask、擦除策略、排版框、字体、字号、方向、
   字色、描边、处理状态和失败原因。只修改译文后应能跳过 OCR/擦字直接重嵌。
9. 图片路径不能依赖 OCR JSON 中旧机器的绝对路径。CLI 显式 --image 优先，其次解析 JSON
   同目录或样本目录的同名图片，并验证 width/height。
10. 先做可审计的命令行核心，再做 GUI。首个可用版本至少提供：
   - render.py --image <图> --ocr <json> --output <图>
   - batch.py --images <目录> --ocr-dir <目录> --output-dir <目录>
   - --dry-run：只画 region/mask/layout 编号核对图，不擦字、不翻译
   - --state：从已有状态重新嵌字
11. 代码按职责拆分，建议模块为 ocr_contract、review、region_masks、eraser、translator、layout、
    renderer、state、pipeline、cli。核心算法不能埋在 GUI 事件里。

实施顺序：

A. 建立新 Git 仓库、README、依赖文件、配置样例和测试框架。
B. 实现 schema 4 解析/校验、人工核对开关/阻断和 dry-run 核对图，先在四页样本上证明区域
   映射与人工决定回填正确。
C. 实现逐列 mask 与局部擦字，并做 mask 外像素不变测试；先不接翻译。
D. 实现完整句翻译、术语表、缓存和 column_ids 映射。
E. 实现中文自动排版、10% 边界约束和白描边艺术字策略。
F. 串联单图/批量 CLI，输出 state、阶段图和错误报告。
G. 在 1、8、19、22 上端到端运行，逐页给出原图、mask、擦字图、嵌字图和越界报告。

最低验收标准：

- 四页所有保留 OCR 列均有明确处理结果，不漏掉多字符正文。
- 不跨气泡、分镜或相邻台词擦除；mask 外没有无关改图。
- 最终文字不超过 layout_bbox_10pct。
- 翻译按完整句子生成，中文可直接用于漫画阅读。
- 同一输入重复运行结果可复现，缓存不会用错原图/OCR/译文。
- 批量运行不打开窗口、不控制鼠标。
- 人工核对默认开启；任何 `pending` 框阻止翻译。开关关闭时明确记录 `bypassed`。
- 自动测试覆盖 schema 校验、路径解析、mask、句列映射、排版边界、缓存签名和失败恢复。

开始时先梳理交接数据并给出短计划，然后直接实现 B 阶段。不要停在只写方案；完成 dry-run
实现、测试，并生成四页核对图后再汇报。
```
