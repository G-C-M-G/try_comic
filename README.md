# 漫画翻译助手（Manga Localizer）

日文漫画 → 中文嵌字全流程：**Photos 蓝框采集 → 蓝框 OCR → 整页翻译（AI 优先 / 本地 Sakura 兜底）→ 擦字 → 竖排/横排嵌字 → PDF**。

本仓库由两个项目合并而成：
- **上游（try_comic 原仓）**：Windows Photos"扫描文本"高亮岛采集、差分岛提取、PaddleOCR 子进程、OCR 人工核对；
- **应用（manga_localizer 原仓）**：岛契约校验、岛内擦字（岛外零改动审计）、manga-ocr × Paddle 交叉验证、
  人工核对 GUI（懒人开关）、整岛一段翻译（本地 Sakura / 在线 AI 精确度开关）、纯竖排双模式排版、state 重嵌。

## 快速上手

```powershell
# 环境（root venv 已含 PySide6/PIL/numpy/winrt/manga_ocr/jsonschema）
venv\Scripts\python.exe -m unittest discover -s tests        # 173 项测试

# 图形界面（推荐：采集 / 处理 / 审核 / 预览 / 术语 / 设置 一站完成）
venv\Scripts\python.exe one_click_gui.py

# 命令行：stage1 采集（打开 Photos → 用户框选蓝框 → 自动抓帧并生成核对图）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录"

# stage2 后台处理（OCR → 翻译 → 擦字 → 嵌字 → PDF）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --process

# 一步到位（采集完自动处理）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --auto
```

详细用法见 `docs/ONE_CLICK_USAGE.md`，架构见 `docs/ONE_CLICK_ARCHITECTURE.md`。

## 目录

```
manga_localizer/      应用核心包（翻译/排版/擦字/进度/契约等）
one_click_gui.py      一键 GUI（PySide6，调用 one_click_translate.py）
one_click_translate.py 一键流水线 CLI（采集 / 处理 / 审核确认）
photos_capture.py     Photos 蓝框采集底层（batch --fast --frames-only）
island_extract.py     差分岛提取 / 保底数据集
paddle_ocr_worker.py  PaddleOCR 隔离子进程（--crops 四变体）
windows_ocr.py        Windows.Media.Ocr(ja) 无窗口识别
config/               glossary、api_keys（git-ignored）、ui 等
tests/                应用测试（tests/ml）+ 上游模块测试
outputs/exam/         上游样例数据集
outputs/oneclick/     一键流水线产物（capture/check/dataset/typeset/PDF）
var/                  运行时缓存
models/paddle/        PaddleOCR 模型缓存
sakura/               本地翻译模型（Sakura LLM）
legacy/               旧版 GUI/列版流水线归档（不参与构建）
docs/                 架构、使用、设计、经验教训
venv/ + paddle_venv/  两套 Python 环境
```

## 文档
- `docs/ONE_CLICK_ARCHITECTURE.md` 一键流水线架构
- `docs/ONE_CLICK_USAGE.md` 使用说明与参数手册
- `docs/GUI设计规范.md` GUI 设计规范
- `docs/BLUEBOX_CAPTURE.md` 蓝框检测原理
- `docs/SENTENCE_GROUPING_DESIGN.md` 分句/分组设计
- `docs/LESSONS.md` 经验教训

## 环境
- `venv/`：应用 + Windows OCR + manga-ocr（root venv，PySide6/PIL/numpy/jsonschema/winrt/manga_ocr/opencv-contrib）
- `paddle_venv/`：PaddleOCR（paddleocr 3.7 + paddle 3.3，模型在 `models/paddle`）
- `sakura/`：翻译模型（本地接口 http://127.0.0.1:8080/v1）
