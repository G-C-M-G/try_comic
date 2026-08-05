# 下一个项目启动提示词：优化上游 OCR 识别区域（高亮岛质量）

把下方"提示词正文"完整发送给新项目中的 AI。项目在 `D:\workplace\opencode\try_comic`（已合并两仓），
输入数据只读引用 `outputs\exam\dataset\`（26-35 十页）与交接文档。

## 提示词正文

```text
我要优化漫画"高亮岛"（Windows Photos 扫描文本）的识别区域质量，并重新生成交接数据集。
本仓库是合并后的漫画翻译流水线：上游（photos_capture/island_extract/make_island_dataset）
产出的 islands.json + filled_* 被下游（manga_localizer/）消费做擦字、OCR 互证、翻译、排版。

请先完整阅读：docs/island contract.md、docs/OCR_交接文档.md、docs/ARCHITECTURE.md、
docs/LESSONS.md（第 2/3 节与"遗留问题"）。样例数据 outputs\exam\dataset33\33\，
全量 outputs\exam\dataset\26-35\。数据集是稳定只读输入，不要改旧数据；工作在新采集产物中。

目标：提升高亮岛几何质量，使下游 OCR（manga-ocr 文本权威 + Paddle 几何佐证）的
REVIEW 数量从 8/62 进一步下降，并消除已知噪声：

1. 岛 bbox 精度：Photos 高亮区常截断文字（下游已证实：截断文本在填黑图里是黑背景，
   扩区救不回，只能从原图重读）。优化 photos_capture.py 的高亮框/岛提取，
   使 image_bbox 完整包住文字（可结合原图二值化膨胀或 Paddle 框反馈）。
2. 微小岛噪声：面积 <800px² 的岛（现 5 个）直接过滤或合并。
3. 相邻文本污染：下游 33/009 的 Paddle 佐证混入相邻岛片段（纯然たる?）——
   岛间判定需排除 bbox 重叠/过近的相邻文字。
4. 识别稳定性：29/004 类多引擎分歧（間/閣/闇）应在采集端用更高分辨率或原图裁剪解决。
5. 保持契约：islands.json 结构不变（schema_version 1，image_bbox/image_center/image_area/
   image_polygon 可选），filled_black/filled_blue 原图分辨率，contours 默认关。

验收标准：
- 重新采集并生成 26-35 十页新数据集，与原数据集并列（outputs\exam\dataset_v2\）。
- 用下游流水线重跑：scripts/cross_validate_pages.py + scripts/m2_review.py +
  scripts/m3_translate_pages.py + scripts/m4_typeset_pages.py，全部可跑通。
- REVIEW 数量 ≤ 3/62；无"文本截断"类 REVIEW；微岛（<800px²）为 0。
- 岛外像素零改动、面积比/中心/缓存规则与现流水线完全兼容（不破坏下游契约）。
- 自动测试覆盖新算法，回归测试保证 26-35 原数据集解析不变。

先实现和测试，再给出新旧数据集对比报告（每页岛数/bbox 覆盖率/REVIEW 前后/识别文本差异）。
```

## 交接清单（本仓库已就绪）

- 全量数据集：`outputs\exam\dataset\26-35\`（islands.json + filled_black/blue + original.jpg）
- 样例：`outputs\exam\dataset33\33\`
- 上游采集：`photos_capture.py`（batch --fast）、`island_extract.py`、`make_island_dataset.py`
- 下游流水线：`scripts\cross_validate_pages.py` / `m2_review.py` / `m3_translate_pages.py` /
  `m4_typeset_pages.py`、`island_render.py`、`island_gui.py`
- 验收基线：`var\island_m1\`（M1-M4 全量产物，REVIEW 8/62、59 typeset）
- 经验：`docs\LESSONS.md`（引擎选型/数据几何/擦字/翻译/排版 26 条 + 遗留问题）
