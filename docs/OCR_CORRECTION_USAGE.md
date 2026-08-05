# OCR 辅助订正使用说明

以下命令按“前台一次采集，后台全部处理”执行。路径使用新运行目录，避免覆盖旧批次。

## 0. 快速路径（fast-mask，推荐）

使用照片“扫描文本”高亮描边岛 + 掩膜后 PaddleOCR 逐岛识别（`--cross-check` 默认关，
开启时叠加 CTD+MangaOCR 仲裁证据）。同一批次复用常驻 Paddle 会话，前台占用约
20-40 秒/页：

```powershell
cd D:\workplace\opencode\try_comic
venv\Scripts\python.exe photos_capture.py batch --fast `
  --images "<原图目录>" --pages $pages --output-dir "$run\capture" --resume --paddle-device cpu
```

`capture.json` 记录 `timing.foreground_active_seconds`（真实前台秒数）与完整照片
清单 `inventory`。冲突列定向补采（只重开照片、只对融合队列列点击复制）：

```powershell
venv\Scripts\python.exe photos_capture.py recapture `
  --capture-dir "$run\capture" --fusion-dir "$run\fusion" --pages $pages
```

后台融合（fast 清单 → schema 4，覆盖检查先行，低置信/兜底/清单不匹配入核对队列）：

```powershell
venv\Scripts\python.exe background_ocr.py "<原图目录>" "$run\fusion" `
  --pages $pages --photos-fast-dir "$run\capture" --dialogue-only `
  --device cpu --paddle-device cpu [--cross-check]
```

## 1. 创建运行目录

```powershell
$run = "D:\workplace\gpt\manga_localizer\output\ocr-correction\20260805-fast-review"
$pages = "1,2,6,9,12,23,28,29,31,33,35"
New-Item -ItemType Directory -Force "$run\capture","$run\fusion","$run\review","$run\reviewed-ocr","$run\final" | Out-Null
```

## 2. Photos 一次采集

只在这一段占用前台“照片”、键盘和鼠标；采集完即可离开窗口。已有完整 artifact 时使用
`ingest`，不会重新打开 Photos：

```powershell
cd D:\workplace\opencode\try_comic
venv\Scripts\python.exe photos_capture.py batch `
  --images "<原图目录>" --pages $pages --output-dir "$run\capture" --quick --resume
```

每页必须有 `capture.json`，且 `capture.complete=true`、坐标校准 `valid=true`、输入哈希和尺寸一致。
只要一页不满足，停止该页，不输出猜测框。

已经通过其他方式保存好每页 `base.png`、`ocr-overlay.png` 和 `inventory.txt` 时，只做后台 ingest：

```powershell
venv\Scripts\python.exe photos_capture.py ingest `
  --images "<原图目录>" --pages $pages --output-dir "$run\capture" --resume
```

## 3. 后台融合 OCR

```powershell
venv\Scripts\python.exe background_ocr.py "<原图目录>" "$run\fusion" `
  --pages $pages --photos-capture-dir "$run\capture" --dialogue-only `
  --device cuda --paddle-device gpu:0
```

没有可用 Paddle GPU 时将最后一项改为 `--paddle-device cpu`。该阶段不打开窗口。
Photos 是文字清单、覆盖和漏框检查的权威；CTD/Paddle/MangaOCR 只用于交叉验证和恢复完整框。

## 4. 只核对冲突项

```powershell
foreach ($p in ($pages -split ',')) {
  venv\Scripts\python.exe ocr_review.py prepare "$run\fusion\$p.ocr.json" `
    --output-ocr "$run\reviewed-ocr\$p.ocr.json" `
    --review "$run\review\$p.review.json" --schema-version 2
}
```

优先处理 `manual_review_queue` 中的 `photos_unmatched_overlay`、Photos 文本冲突、异常窄框、
漏框和 `role=uncertain`。每个可见对白/旁白必须明确为 `approved/corrected/rejected`；仍有
`pending` 时禁止翻译。

已有审计过的 compact patch 可直接应用：

```powershell
venv\Scripts\python.exe ocr_review.py apply-patch `
  "$run\fusion\$p.ocr.json" "<review_patch.json>" "$run\reviewed-ocr\$p.ocr.json" `
  --review "$run\review\$p.review.json"
```

## 5. 下游批量翻译、擦字和嵌字

```powershell
cd D:\workplace\gpt\manga_localizer
.venv\Scripts\python.exe batch.py `
  --images "<原图目录>" --ocr-dir "$run\reviewed-ocr" --output-dir "$run\final" `
  --pages 1,2,6,9,10,12,23,28,29,30,31,33,35 `
  --translator sakura --glossary config\glossary.json `
  --max-layout-overflow 10 --preserve-unit-center
```

只有人工确认允许时才使用 `--max-layout-overflow 20`；实际使用 20% 必须逐项记录原因并进入复核。
若只改译文，用已有 state 重嵌，不重新 OCR、擦字或翻译：

```powershell
.venv\Scripts\python.exe render.py --ocr "<同页 OCR JSON>" `
  --state "$run\final\<页>.state.json" --output "$run\final\<页>.reembedded.png"
```

## 6. 最小验收

检查 `batch.report.json` 和各页 `*.stages/*.audit.json`：状态均为 `completed/typeset`，无 pending、
无失败列、无 20% 外像素、mask 外变化为 0、所有列像素实际变化、排版框不低于原框 80%。确认第 10、30
页不跨气泡后，才覆盖正式输出目录。

## 7. 时间优化开关

- `--resume`：复用同一原图哈希的 Photos artifact。
- `--pages`：只处理问题页，不扫描整目录。
- `--dialogue-only`：拟声词、标题和画面艺术字进入忽略清单，不进入翻译。
- `--no-manual-review`：仅在明确接受风险时使用；必须保留 `bypassed` 记录。
- `--translator mock`：测试管线，不用于成品。

核心原则：把前台时间花在一次 Photos 批采集；把等待、融合、冲突判断和翻译全部留在后台。
本次 11 页基线约为 59 分 50 秒墙钟跨度；下一轮以 `capture.json.timing` 的真实前台秒数为准，
不能继续用文件时间戳估算。
