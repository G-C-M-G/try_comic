# 漫画翻译助手（Manga Localizer）

日文漫画 → 中文嵌字全流程：**高亮岛采集 → 岛数据集 → 双引擎 OCR 互证 → 整岛翻译 → 岛内竖排嵌字 → state 重嵌**。

本仓库由两个项目合并而成：
- **上游（try_comic 原仓）**：Windows Photos"扫描文本"高亮岛采集、差分岛提取、PaddleOCR 子进程、OCR 人工核对；
- **应用（manga_localizer 原仓）**：岛契约校验、岛内擦字（岛外零改动审计）、manga-ocr × Paddle 交叉验证、
  人工核对 GUI（懒人开关）、整岛一段翻译（本地 Sakura / 在线 AI 精确度开关）、纯竖排双模式排版、state 重嵌。

## 快速上手

```powershell
# 环境（root venv 已含 PySide6/PIL/numpy/winrt/manga_ocr/jsonschema）
venv\Scripts\python.exe -m unittest discover -s tests        # 92 项测试

# 单页 dry-run（33 页样例，原图 + 岛 + 6px 黄描边）
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset33\33 --dry-run --output var\smoke\dry-run.png

# 单页完整流水线：擦字 -> 翻译 -> 排版 -> state
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --image outputs\exam\dataset\33\original.jpg --erase-only --min-island-area 800 --output var\run\33.erased.png
venv\Scripts\python.exe scripts\m3_translate_pages.py --m2-dir var\island_m1\m2 --dataset-root outputs\exam\dataset --pages 33 --output-dir var\run --precision local
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --typeset-only --erased-image var\run\33.erased.png --translation-json var\run\33.translation.json --ocr-lines var\island_m1\m2\33.paddle.lines.json --output var\run\33.typeset.png --audit-json var\run\33.typeset.audit.json

# 只改译文重嵌（跳过 OCR/擦字/翻译，<0.1s）
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --reembed-state var\run\33.state.json --erased-image var\run\33.erased.png --edit-translations edits.json --output var\run\33.reembedded.png

# 人工核对 GUI（懒人开关默认开=承认 manga 结果）
venv\Scripts\python.exe island_gui.py --m2-dir var\island_m1\m2 --dataset-root outputs\exam\dataset
```

## 目录

```
manga_localizer/      应用核心包（岛契约/擦字/交叉验证/翻译/排版/state/GUI）
scripts/              流程脚本（采集→数据集→M1~M4）
island_render.py      无窗口 CLI（dry-run/erase/translate/typeset/reembed）
island_gui.py         人工核对 GUI 启动器
schema/               OCR 结构契约（schema 4 + review v2）
config/               glossary（ヨドイン/ウイラス/ワテクア/カークァス 等）、ui.json
tests/                岛版测试（tests/ml）+ 上游模块测试
outputs/exam/         上游数据集（masked10 采集 / dataset 全量 / dataset33 样例）
var/island_m1/        M1-M4 验收证据（标注图/报告/state/缓存）
models/paddle/        PaddleOCR 模型缓存（paddle_ocr_worker 依赖）
sakura/               本地翻译模型（Sakura LLM）
legacy/               旧版 GUI/列版流水线归档（不参与构建）
docs/                 架构、使用、经验教训、交接文档
photos_capture.py     上游：Photos 高亮采集（batch --fast）
island_extract.py     上游：差分岛提取
make_island_dataset.py 上游：交接数据集生成
paddle_ocr_worker.py  上游：PaddleOCR 隔离子进程（--crops 四变体）
windows_ocr.py        上游：Windows.Media.Ocr(ja) 无窗口识别
ocr_review.py         上游：OCR 人工核对
background_ocr.py     上游：背景融合 OCR
venv/ + paddle_venv/  两套 Python 环境
```

## 文档
- `docs/ARCHITECTURE.md` 架构与数据流
- `docs/USAGE.md` 全命令手册
- `docs/LESSONS.md` 经验教训
- `docs/island contract.md` / `docs/OCR_交接文档.md` 上游交接
- `NEXT_PROJECT_PROMPT_OCR_OPTIMIZE.md` 下一个项目（优化高亮岛识别区域）提示词

## 环境
- `venv/`：应用 + Windows OCR + manga-ocr（root venv，PySide6/PIL/numpy/jsonschema/winrt/manga_ocr/opencv-contrib）
- `paddle_venv/`：PaddleOCR（paddleocr 3.7 + paddle 3.3，模型在 `models/paddle`）
- `sakura/`：翻译模型（本地接口 http://127.0.0.1:8080/v1）
