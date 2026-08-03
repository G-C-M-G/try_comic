# OCR 里程碑备份与清理报告

执行日期：2026-08-04

## 已创建的可恢复备份

备份目录：`backups\2026-08-04_ocr-handoff\`

| 文件 | 内容 |
|---|---|
| `try-comic-pre-clean.bundle` | 清理前根仓库完整 Git 历史 |
| `try-comic-final.bundle` | 文档和清理完成后的根仓库完整 Git 历史 |
| `manga-image-translator.bundle` | MITS 嵌套仓库完整 Git 历史及 `local/try-comic` 分支 |
| `legacy-generated-before-clean.zip` | 清理前的完整 `outputs` 与 `result`，111 项、解压约 28.71 MB |
| `ocr-handoff-final.zip` | 新项目最小交接包：文档、提示词、样本、最终 JSON、基准、核对图和 producer 参考代码 |
| `SHA256SUMS.txt` | 上述备份文件的 SHA-256 校验值 |

`legacy-generated-before-clean.zip` 的归档后大小约 27.21 MB；归档创建并成功读取目录后才开始
清理旧输出。

## 已清理

- `outputs` 下 9 个旧调试目录。
- `outputs` 根部 63 个实验截图、中间 JSON、多代翻译图和状态文件。
- 旧 `result` 目录。
- 根目录和测试目录的 Python `__pycache__`。
- 空 `.cache` 目录。
- `sfx_report.txt` 与 `sakura\gui_server.log` 运行日志。

清理后的 `outputs` 只保留：

```text
outputs/
├── hybrid_batch/        # 4 个最终 OCR JSON
├── paddle_batch_input/  # 4 张对应原图
├── ocr_baselines/       # 1、8、19 页照片基准
└── ocr_review/          # 4 张核对图和 OCR_result.md
```

这些输出也包含在 `legacy-generated-before-clean.zip` 与最终交接包中，所以本地目录损坏时仍可
恢复。

## 有意保留

以下内容未视作冗余删除：

- `venv`、`paddle_venv`：当前可运行的两个隔离 Python 环境。
- `models`、`sakura`：OCR 与翻译模型，离线运行所需。
- `manga-image-translator`：独立 Git 仓库和上游引擎。
- `译文缓存`、`config` 中的词典/翻译记忆：用户运行数据。
- `依赖备份`：环境重建所需的本地 wheel。
- 根目录既有 `backup_*.zip`：合计体积很大，但代表多个历史阶段，未证明内容等价，因此没有
  擅自删除。

若以后需要释放磁盘，应该先分别列出旧 ZIP 的内容、校验关键版本能否从 Git bundle 恢复，
再由用户明确决定删除哪些历史阶段。

## 验证结论

- 根仓库与 MITS 仓库工作树在交接前均为干净状态。
- 根仓库全部 38 项单元测试通过。
- 四个最终 JSON 均为 schema 4，原图存在且尺寸匹配。
- 1/8/19 页继续满足框 F1 不低于 90%、CER 不高于 10%、多字符正文零遗漏。
- OCR 核对图可由 `ocr_preview.py` 从最终 JSON 重新生成。
