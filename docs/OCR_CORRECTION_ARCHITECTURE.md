# OCR 辅助订正架构

## 目标与边界

`try_comic` 是 OCR producer；`manga_localizer` 是 schema 4 consumer。producer 可以调用 CTD、
PP-OCRv6、MangaOCR 和 Windows“照片”，consumer 只读取经过核对的 schema 4 OCR JSON。
不复制旧 GUI、模型、缓存或 MITS；任何未来 OCR 仍通过 producer CLI 或明确适配器调用。

## 权威规则

Windows“照片”是订正阶段的权威证据，优先用于：

- 判断页面上是否存在可见文字；
- 保存 Photos 全选文字清单及阅读顺序；
- 发现漏框、异常窄框和候选冲突；
- 在 Photos 与其他 OCR 结果一致时确认文字内容。

原图坐标是几何和像素的权威来源：Photos 的蓝色高亮必须先通过特征匹配校准到原图坐标，
不能直接把视口截图或高亮层当成最终擦字 mask。Photos 与 CTD/Paddle/MangaOCR 冲突时，保留
双方候选并进入核对；不能静默选择较短框。

## 两段式流程

```text
短暂前台 Photos 采集
  -> base.png / ocr-overlay.png / inventory.txt / capture.json
  -> 坐标校准与哈希校验
  -> 后台 CTD + PP-OCRv6 + MangaOCR 融合
  -> Photos 权威覆盖检查
  -> 仅冲突、漏框、异常窄框和角色不确定项进入核对
  -> schema 2 决定回填
  -> 兼容 schema 4 OCR JSON
  -> manga_localizer: 逐列 mask -> 擦字 -> 整句翻译 -> 排版 -> state / QA
```

### P0 输入快照

为每个运行生成唯一 `run_id`，记录页号、原图 SHA-256、尺寸和 OCR 版本。所有阶段目录只写入
该运行，禁止把新旧批次混在同一目录。

### P1 Photos 快速采集

一次前台批量采集，开启 `--quick --resume`。每页只保存四类必要资产：原图快照、Photos 高亮层、
全选文字清单和 `capture.json`。`capture.json` 必须包含 viewport、仿射矩阵、特征匹配数、inlier
比例、重投影误差、输入哈希和 `complete` 状态。采集完成后关闭前台控制；后续不打开窗口。

### P2 后台融合

Photos 文字和高亮作为主证据，其他 OCR 只用于补充几何和检测冲突。自动通过条件应同时满足：

1. Photos 坐标校准有效，哈希和尺寸匹配；
2. Photos 清单与候选按阅读顺序单调匹配；
3. 没有唯一候选冲突、漏框、异常窄框或对白/旁白角色歧义。

不满足任一条件就写入 `manual_review_queue`，不进入翻译。

### P3 核对与回填

核对文件使用 schema 2，支持批准、改字、改完整框、调整顺序、设置 `panel_index`、
`container_id`、`role`、断句、恢复候选和拒绝误检。所有可见对白/旁白必须成为
`accepted/corrected/rejected`；`pending` 阻断翻译，关闭核对只记录 `bypassed`，不能删除风险。
回填后只输出新的 schema 4 JSON，原始 OCR 保持只读。

### P4 下游本地化

翻译只请求 `translation_units.source_text`；`translation_query` 的列标记只用于可审计映射，
禁止逐列直译。擦字按每个 `column_id` 的 polygon 建 mask：白色气泡用受限填充，线稿/网点/描边
文字用局部 inpainting。排版以核对后的完整 `image_bbox` 为基准，10% 是默认软约束，20% 只能
从完整框直接计算且是硬上限；实际框宽高不得低于原框 80%。

## 性能策略

- 前台只做 Photos 采集；CTD/Paddle/MangaOCR、融合、核对图、翻译、擦字和嵌字均后台运行。
- 默认整页采集，不逐列点击；只有后台冲突项才做一次定向补采。
- `--resume` 按原图哈希复用已完成采集；哈希、尺寸或坐标校准变化时强制失效。
- OCR 可用 `--device cuda`；Paddle 在有 GPU 模型时用 `--paddle-device gpu:0`，否则用 CPU，
  不影响 Photos 采集。
- 下游批量保持串行，翻译缓存签名绑定原图、OCR、完整原文、列映射、术语表、模型和提示版本。
- 像素审计只检查中央 20% 加固定步长整框样本，预算 60 秒；擦字和排版时间只记录，不把它们误判为
  像素审计超时。

## 当前耗时基线

本次 11 页 Photos 订正从首个 `base.png` 到最后一个 `inventory.txt` 的墙钟跨度约 59 分 50 秒；
按每页 `base.png -> inventory.txt` 相加的有效采集跨度约 36 分 14 秒，`base.png -> ocr-overlay.png`
相加的 Photos 扫描跨度约 5 分 07 秒。这些数字来自文件时间戳，只能作为下一轮对比基线。

现有 `capture.json` 没有记录窗口真实激活区间。下一轮必须增加：

```text
timing.foreground_started_at
timing.foreground_ended_at
timing.foreground_active_seconds
timing.photos_wait_seconds
timing.input_control_seconds
timing.background_processing_seconds
```

`batch.capture.json` 还应汇总总前台秒数、每页 P50/P95、复用页数、冲突列数和定向补采列数。
没有这些字段时，不得把文件写入跨度当作精确窗口占用时间。

## 运行资产

建议目录如下，`capture`、`fusion`、`review` 和 `final` 均位于同一个 `run_id` 下：

```text
output/ocr-correction/<run_id>/
  capture/<page>/capture.json
  fusion/<page>.ocr.json
  review/<page>.review.json
  reviewed-ocr/<page>.ocr.json
  final/<page>.translated.png
  final/<page>.state.json
  final/<page>.stages/{mask,erased,translated,*.audit.json}
  batch.report.json
```

可删除的临时截图、失败重试和旧批次应移入外部备份，不进入源码仓库。正式输出只复制 `final` 和
必要审计，不复制模型、缓存或旧项目目录。

## 失败恢复

采集失败：保留失败 manifest，重试该页；校准失败时不生成猜测框。融合冲突：只补采冲突列并重新
生成该页 OCR。翻译失败：重试当前模型、切换 Translator，再人工处理；整列嵌字前后无像素变化时
必须标记失败。排版失败：先缩字号、调列和字距，超过 10% 才能申请直接 20%，并进入人工核对。
所有失败原因和输入哈希写入 state/report，已完成页可复用。
