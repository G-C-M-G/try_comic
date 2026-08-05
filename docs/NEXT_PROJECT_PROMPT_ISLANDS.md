# 下一个项目启动提示词：高亮岛内翻译嵌字

把下方"提示词正文"完整发送给新项目中的 AI。新项目使用独立目录，输入数据来自
`D:\workplace\opencode\try_comic\outputs\exam\dataset33`（示例页）与交接文档
`D:\workplace\opencode\try_comic\island contract.md`。

## 提示词正文

```text
我要创建一个独立的日文漫画翻译嵌字项目，核心约束：翻译文字必须严格限制在高亮岛内。

上游数据位于：
D:\workplace\opencode\try_comic

请先完整阅读 island contract.md，并查看 outputs\exam\dataset33\33\ 下的样例数据集
（filled_black.png / filled_blue.png / islands.json / islands_annotated.png / original.jpg）。
数据集是稳定、只读输入：原图、填黑/填蓝岛掩膜、每岛 bbox/中心/面积（可选多边形轮廓）。
不要重新采集照片，不要复制上游的 GUI、模型、缓存或 MITS。

目标：给定原图与岛数据集，自动把岛内的日文文字逐岛识别、整句翻译成中文，并嵌字回原图，
且满足：岛外像素零改动、译文文字与擦除区域全部落在岛边界内、翻译按整句不逐字直译。

引擎规则（重要）：
1. 文字识别优先 Windows.Media.Ocr(ja)（已装语言包，winrt 调用，无窗口、无模型下载）；
   它返回行级框与文本（无置信度），可直接与岛匹配。
2. 空结果或低质量时回退 PaddleOCR（上游 paddle_ocr_worker.py 子进程接口，或自行封装
   PP-OCRv6 medium）。
3. Windows Photos 扫描文本采集默认关闭：只在需要人工补采某岛时作为可选开关使用，
   默认不打开任何窗口、不控制鼠标。
4. 高亮岛轮廓标注功能默认关闭（`--contours`）：仅验证/调试时开启，输出"形状保真填图
   （非背景像素=岛精确轮廓）+ 岛多边形 + 标注图"，不是主流程依赖。

必须遵守的核心规则：
1. 几何权威 = islands.json 的 image_bbox（及可选 image_polygon）。擦字 mask 按岛/列生成，
   不得用巨大外接矩形覆盖岛外；岛外像素变化必须为 0（逐像素审计）。
2. 嵌字硬边界 = 岛 bbox（默认每侧 0% 扩张；如需放宽最多 5%，且逐项记录原因并入人工核对）。
   译文放不下时先缩小字号、调行列字距；任何模式不得越岛。
3. 翻译只按完整句子请求（岛内列按阅读顺序合并成句，句末标点切分），禁止逐列直译。
   同一岛可含多句。
4. 人工核对默认开启：OCR 低置信/多引擎分歧/清单不匹配项进入队列并阻断翻译；关闭时只记录
   bypassed，不删除风险。
5. 拟声词（SFX）保留并翻译（默认开启），样式可单独配置（黑字白描边）。
6. 输出原图同尺寸最终图 + 可重编辑 *.state.json（记录岛 id、原文、译文、mask、擦除策略、
   排版框、字号、越界与失败原因）。只改译文应能跳过 OCR/擦字直接重嵌。
7. 全部离线可跑（Sakura 本地接口 http://127.0.0.1:8080/v1 为默认翻译器，需可替换接口），
   批量不打开窗口、不控鼠标、可断点恢复、输出 batch.report.json。
8. 术语表/人名表（如 ヨドイン/ウイラス/ワテクア/カークァス）做成可配置 glossary，
   译文避免日语语序、生硬书面语与无意义扩写。

实施顺序：
A. 建仓库/README/依赖/测试框架；解析 islands.json 与 filled_* 图，校验与原图尺寸一致。
B. 实现岛内擦字 mask 与"岛外像素变化=0"审计；dry-run 输出岛/列编号核对图。
C. 接入 Windows.Media.Ocr(ja) 岛内识别 + Paddle 回退，输出岛级原文与列几何。
D. 整句翻译（Sakura）+ 术语表 + 缓存（签名含原图/岛/原文/模型/提示版本）。
E. 中文排版：竖排右到左，硬边界=岛 bbox（0% 默认），越岛即失败并记录。
F. 单图/批量 CLI + state + 报告；在 dataset33 样例上端到端跑通并给出核对图。

验收标准：
- 10 页样例（26-35，上游 masked10 可生成全量数据集）岛外像素变化 = 0。
- 所有保留岛均有处理结果；译文框不越岛；无跨岛擦除。
- 重复运行可复现；缓存不串图；批量无窗口。
- 自动测试覆盖：岛解析、mask、岛外零改动、句列映射、0%/5% 越界、缓存签名、失败恢复。
先实现和测试，再给出每页耗时与越界报告。
```

## 交接清单（上游已就绪）

- 10 页采集与填黑/填蓝数据：`outputs\exam\masked10\`（含每页 photos_base/photos_ocr/capture.json）
- 样例数据集：`outputs\exam\dataset33\33\`（原图/填黑/填蓝/islands.json/标注图）
- 岛提取：`island_extract.py --contours`（默认关）
- 翻译验证：`island_translate.py --engine windows-ocr`（Windows OCR 优先，Paddle 兜底）
- 文档：`island contract.md`
