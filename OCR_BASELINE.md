# OCR 基准与改造路线

更新日期：2026-08-04

## 1. 首个真实样本

- 原图：`C:\Users\YLP\Desktop\软件\漫画翻译\原文\54话生肉福利连\5.jpg`
- 对照成品：`C:\Users\YLP\Desktop\软件\漫画翻译\翻译\葬送的芙莉莲漫画_第54话 对比\5.jpg`
- Windows“照片”扫描文本：已在原图上实际运行，并逐列选择、复制和标定坐标。

目标气泡正文三方结果：

```text
探知できている複製体の総数が受験者よりも僅かに少ないです。
特に一番の脅威であるゼンゼさんの複製体の反応が見つかりません。
```

该气泡由七个列框组成，并在第四列句号处分为两个完整句子。现有 MangaOCR 对第一句正文正确，但页面下方存在真实 OCR 错误：现有结果把 `ゼンゼ` 识别成了 `ゼノゼ`，Windows“照片”识别正确。当前成品的另一项问题是原文子框被合并为大中文框，原有分块和留白被破坏。

## 2. Windows“照片”扫描的实际特性

优点：

- 竖排日文正文识别准确。
- 能给出比当前状态文件更细的文字选择区域。
- 本页对 `ゼンゼ` 的识别优于现有 MangaOCR 结果。
- 每次选中一列时会显示可测量的高亮区域，可映射回原图坐标。

限制：

- 整页“全选/复制”会把振假名单独混入正文，例如 `ふくせいホいたち`、`えせいたい`。
- 跨气泡复制顺序不是漫画阅读顺序，不能直接作为翻译输入。
- “照片”应用没有公开、稳定的批量 OCR API，界面自动化只能作为可替换的采集适配器。
- 本机 `Windows.Media.Ocr` 只报告 `zh-Hans-CN` 语言包，不能复用“照片”的日文效果。
- 新版 Windows AI `TextRecognizer` 需要 NPU/Copilot+ PC，不适合作为当前机器的主方案。

结论：Windows“照片”适合作为高精度基准、人工校正入口和小批量列框采集器，不适合作为无人值守的主批处理后端。当前原型将不稳定的界面采集与稳定的列分组、句子重建和坐标映射分开。

## 3. 候选 OCR

### 首选验证：PP-OCRv6 medium

- 官方模型统一支持日文，并返回检测框、识别文本和置信度。
- 官方日文检测基准明显高于 PP-OCRv5；medium/small 支持日文，tiny 不支持日文。
- 应放进独立 OCR 环境或独立进程，避免 Paddle 依赖污染现有 Torch/MangaOCR 环境。

### 保留：MangaOCR

- 对漫画竖排、振假名、复杂字体和低质量图像有专门训练。
- 能整块识别，但不提供字符/词级坐标；继续作为气泡级复核器，而不是唯一几何来源。

### 不作为主方案：Tesseract `jpn_vert`

- 本机已经安装并在本页实测。
- 整页及目标气泡都出现大量错字、空格和版面噪声，明显弱于 Windows“照片”和现有 MangaOCR。

## 4. 目标管线

```text
文本检测（CTD / PP-OCR）
  -> 保留每个多边形与置信度
  -> 区分正文和小号振假名
  -> 按竖排右到左重建列顺序
  -> MangaOCR 对整个气泡做语境复核
  -> 同一气泡内按句号/人工断点切成完整句子
  -> 按完整句子翻译，同时保留组成它的原文列框
  -> 按列框联合擦字，并在对应气泡范围内排版译文
```

本次已先完成底层准备：CTD 会真正采用配置中的 `box_threshold`；文本行只合并一次；状态 JSON 保存 `source_segments`、`translation_segments`、`ocr_backend` 和置信度来源；工作台保存时按比例保留多框几何；缓存签名包含完整检测/OCR/渲染配置。`photos_ocr.py` 另行实现了列级清单、窗口坐标到原图坐标的映射、气泡聚合和气泡内句子切分。MangaOCR 本身不返回置信度，旧系统的高分来自辅助 48px 模型，现在只记录为 `ocr_score_proxy`，不再冒充最终 OCR 置信度。

主翻译管线现已加入离线句子重建：MITS 按画格排好 OCR 区域后，连续累积到 `。！？` 或省略号；整句附带临时分段标记送给本地 Sakura，以完整语境翻译；译文再按标记或标点回填原始文本框。画格检测失败时只允许几何相邻框合并。Windows“照片”不参与运行时，只用于准确率基准。

## 5. 后续验收方法

从 18 页样本中固定一组人工真值，不再只凭肉眼看单页：

- 检出率：正文气泡漏检数、误检数。
- 框质量：每个原文块的 IoU，振假名是否混入正文。
- 字符准确率：正文 CER，并单独统计人名。
- 阅读顺序：竖排列和跨气泡顺序错误数。
- 嵌字：译文是否落回对应原框、是否越界、是否破坏气泡留白。

PP-OCRv6 已按这组样本与 CTD + MangaOCR 并排验收；它作为补漏和交叉验证后端默认启用，
但不无条件覆盖稳定的 MangaOCR 文字结果。

## 6. 无窗口 OCR 实现与当前基准

`background_ocr.py` 已实现可离线批量运行的双模型列级 OCR：

```text
CTD 文本检测/掩膜 + PP-OCRv6 medium 独立整页检测
  -> 重叠框交叉验证 + Paddle 独有框补漏 -> 宽区域拆成竖排列/去重
  -> 每列原图/CLAHE/Otsu 三次 MangaOCR -> 双模型文字仲裁
  -> 低置信单字符可放弃 -> 漫画阅读顺序 -> 按句末标点组成翻译单元
```

运行时不启动 Windows“照片”，不打开其他窗口，也不发送鼠标或键盘事件。Windows“照片”
清单只在指定 `--photos-baseline` 时读取，用作离线验收真值，不参与检测或识别。

单图及目录批量命令：

```powershell
venv\Scripts\python.exe background_ocr.py "D:\漫画\1.jpg" "outputs\1.ocr.json" --device cuda
venv\Scripts\python.exe background_ocr.py "D:\漫画\整话" "outputs\整话OCR" --device cuda
```

带“照片”对照的验收命令：

```powershell
venv\Scripts\python.exe background_ocr.py "D:\漫画\1.jpg" "outputs\1.ocr.json" `
  --photos-baseline "outputs\1.photos.json" --device cuda
```

PaddleOCR 在独立 `paddle_venv` 子进程中运行，避免 Paddle 与现有 Torch/Transformers 依赖互相
污染。PP-OCRv6 检测和识别模型保存在 `models\paddle`，当前共约 133 MB。模型首次下载后，
子进程显式传入本地模型目录并关闭模型源探测，批量运行无需联网。复建独立环境：

```powershell
D:\tools\Python311\python.exe -m venv paddle_venv
paddle_venv\Scripts\python.exe -m pip install -r requirements-paddle-ocr.txt
```

JSON 中 `columns` 是每个可单独嵌字的原文列，包含三次读数、识别一致性、四点多边形、
原框和总宽高放宽 10% 的 `layout_bbox_10pct`；`translation_units` 保存完整句上下文和组成它的
列 ID，翻译后仍可逐列回填。单字符只有在三次读数一致性低于 0.67 时才放弃，多字符正文列
始终严格保留。每列还记录 `detection_sources`、`paddle_validation`、`cross_validation` 和
`text_source`，可审计两个模型的框重叠、文字分歧及最终取值原因。

仲裁规则是 MangaOCR 稳定结果优先，PaddleOCR 不因单次高置信就覆盖漫画专用模型；只有
MangaOCR 三次明显分歧且 Paddle 给出置信度不低于 0.90 的多字符日文时，Paddle 才接管文字。
Paddle 独有高置信多字符框会进入 MangaOCR 三次复核后补入。Paddle 独有单字符不能靠一次
结果直接放行。

验收门槛固定为：多字符正文零遗漏、相对“照片”的字符错误率不超过 10%、文本框精确率与
召回率的 F1 不低于 90%。框匹配不再只看召回率，额外误框同样会降低分数。全角/半角标点及
省略号形式先做等价归一，再计算字符编辑距离。

2026-08-03 实测：

| 页面 | 框精确率 | 框召回率 | 框 F1 | 字符错误率 | 多字符漏列 |
|---|---:|---:|---:|---:|---:|
| 1.jpg | 94.44% | 100% | 97.14% | 5.48% | 0 |
| 8.jpg | 100% | 100% | 100% | 4.55% | 0 |
| 19.jpg | 100% | 96.55% | 98.25% | 1.48% | 0 |

19.jpg 唯一未匹配项是“照片”自身不稳定的一列单字符，符合允许放弃的规则。22.jpg 没有
“照片”基准，后台扫描得到 18 个有效列、7 个翻译单元，未用于计算上述准确率。

四页批量共 70 个有效列都获得 PP-OCRv6 框交叉确认。当前四页没有出现需要补入的
Paddle-only 正文框；补漏分支通过自动测试覆盖。1.jpg 中 MangaOCR 对 `ダメだ…` 的三次读数
互相冲突，PaddleOCR 高置信结果与“照片”一致，仲裁后把本页字符错误率由 8.22% 降到 5.48%。

双模型不一致不再直接进入下游翻译。默认策略会把文字分歧、单模型定位、弱框重叠、低重复
一致性与 Paddle 接管框写入 `manual_review_queue`，并设置翻译阻断。人工核对可显式关闭，
但候选仍以 `bypassed` 状态保留，方便之后追溯或补审。

## 7. 交接冻结

本基准对应 schema 4 OCR 里程碑。最终样本、JSON、照片基准和核对图已经标准化到
`outputs\paddle_batch_input`、`outputs\hybrid_batch`、`outputs\ocr_baselines` 和
`outputs\ocr_review`。后续擦字/翻译/嵌字作为独立项目消费这些数据，具体契约见
`OCR_交接文档.md`，新项目启动提示词见 `NEXT_PROJECT_PROMPT.md`。
