# OCR 交接审计

审计包 SHA-256：`10f2074a00502fbd5c9ef5ae3e13c08e7d5023c12657372d58df30dcf8d25d3b`

已完整阅读 `OCR_交接文档.md`、`架构说明.md`、`使用说明.md`、`OCR_BASELINE.md`，并逐页核对
原图、最终 OCR JSON、review 模板、照片基准和 OCR 核对图。交接输入为 schema 4，四页尺寸均
为 985x1400，共 70 个保留列。

| 页 | 保留列 | 翻译单元 | ignored | pending | 照片框 F1 | CER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 6 | 0 | 5 | 97.14% | 5.48% |
| 8 | 6 | 1 | 0 | 0 | 100% | 4.55% |
| 19 | 28 | 6 | 1 | 1 | 98.25% | 1.48% |
| 22 | 18 | 7 | 1 | 10 | 无 | 无 |

1、19、22 当前 `translation_blocked=true`；8 可直接通过 review gate。19 的照片基准唯一漏配
是低置信单字 `し`，不是多字符正文漏列。22 没有照片基准，画面字带白描边的比例明显更高，
后续必须走局部 inpainting/艺术字样式策略，不能套用气泡白底填充。

最新版 review 模板与各自 OCR 文件 SHA-256 完全匹配。待核对项的 MangaOCR 三次读数、
PaddleOCR 候选、重叠率和原因均保留；在用户完成决定前，本项目只生成核对图，不进入翻译。

B 阶段 dry-run 已为 70 个保留列生成 70 个独立 polygon envelope mask。相邻列共享边界导致
少量像素重叠，但四页 `cross_unit_overlap_pixels` 均为 0，没有跨翻译单元/相邻台词的 mask
重叠。集中人工队列见 `artifacts/dry_run/manual_review_queue.md`。
