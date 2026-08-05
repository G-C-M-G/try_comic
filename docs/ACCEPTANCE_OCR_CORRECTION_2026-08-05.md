# 13 页 OCR 订正验收

验收页面：`1、2、6、9、10、12、23、28、29、30、31、33、35`。其中 11 页经 Photos 辅助
重新融合，10、30 页只重建气泡归属、顺序和翻译单元。

## 结果

- 13/13 页流水线状态为 `typeset`，共 280 个保留列，失败列 0。
- 所有人工队列 `pending=0`、`bypassed=0`。
- 所有擦字结果 mask 外变化像素为 0。
- 所有嵌字结果 20% 硬边界外变化像素为 0。
- 所有列均有中央 20% 或整框低采样像素变化；最慢单页像素审计低于 1 秒。
- 所有排版框宽高均不小于完整 OCR 框的 80%，整句基础字号一致，中心锁定全部通过。
- 13 页均在 10% 软边界内完成，实际使用 20% 的列为 0。

已验证回归：第 1 页包含 `じゃなく`、`している` 和 `やられた！！` 完整框；第 10 页拒绝
`そういえば` 并拆开 `…あれは`/`ただの猟師だ`；第 30 页两段长句不跨气泡；第 31 页为
`あの剣はアイツの…`；第 35 页 `馬鹿な！` 使用完整框。第 9、28、33 页的异常窄框在视觉抽查
中进一步按 Paddle 完整候选订正，重跑后不再残留可读日文。

候选产物位于 `output/ocr-correction/final-candidate/`。每个 `<页>.stages/` 包含：

```text
original.jpg
ocr-review.png
ocr-review.audit.json
mask.png
erased.png
erase.diff.png
erase.audit.json
translation.json
translated.png
typeset.audit.json
```

修改前备份：

- `D:\workplace\gpt\backups\2026-08-05_ocr-correction-prechange`
- `测试\_backups\ocr-correction-20260805-012353`
