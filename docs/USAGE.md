# 使用说明（USAGE）

所有命令在仓库根目录执行，Python 用 `venv\Scripts\python.exe`（Paddle 相关用 `paddle_venv`）。

## 0. 环境

```powershell
venv\Scripts\python.exe -m unittest discover -s tests   # 92 项测试
```

依赖已装：PySide6 / PIL / numpy / jsonschema / winrt(windows_ocr) / manga_ocr / opencv-contrib。
`paddle_venv` 独立（paddleocr 3.7 + paddle 3.3）；Paddle 模型缓存于 `models\paddle`（离线）。
Sakura 翻译模型在 `sakura\`，启动后监听 http://127.0.0.1:8080/v1。

## 1. 上游采集与数据集（只在需要新数据时执行）

```powershell
# 1) Photos 高亮采集（前台，batch --fast：每页新窗、OCR 期间最小化、结束关闭）
venv\Scripts\python.exe photos_capture.py batch --fast --images "<原图目录>" --pages 26-35 --output-dir outputs\exam\masked10 --paddle-device cpu

# 2) 差分岛提取（--fill-color 任意；--contours 默认关=bbox 模式）
venv\Scripts\python.exe island_extract.py --input outputs\exam\masked10 --output outputs\exam\islands10 --pages 26-35 --fill-color 30,144,255

# 3) 交接数据集（islands.json + filled_black/blue + original.jpg）
venv\Scripts\python.exe make_island_dataset.py --images "<原图目录>" --capture outputs\exam\masked10 --output outputs\exam\dataset --pages 26-35
```

## 2. 岛流水线（M1-M4）

### M1 岛契约 + 擦字 + dry-run
```powershell
# dry-run 核对图（岛编号 + mask + 硬边界 + 6px 黄描边）
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset33\33 --dry-run --output var\run\33.dry-run.png

# 擦字（--min-island-area 800 舍弃微岛；岛外像素变化=0 审计）
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --image outputs\exam\dataset\33\original.jpg --erase-only --min-island-area 800 --output var\run\33.erased.png --audit-json var\run\33.erase.audit.json
```

### M2 交叉验证（manga 文本 × Paddle 几何）
```powershell
# 全 10 页：manga 批量识别 + Paddle 岛级 crops + 互证表
venv\Scripts\python.exe scripts\cross_validate_pages.py --dataset-root outputs\exam\dataset --pages 26-35 --output-dir var\run\cv
# 查看 REVIEW 明细
venv\Scripts\python.exe scripts\list_cv_reviews.py --cv-dir var\run\cv
# REVIEW 岛原图重读（修复背景误读，如 間→闇）
venv\Scripts\python.exe scripts\m2_review.py --cv-dir var\run\cv --dataset-root outputs\exam\dataset --pages 26-35 --output-dir var\run\m2
# 人工核对 GUI（懒人开关默认开）
venv\Scripts\python.exe island_gui.py --m2-dir var\run\m2 --dataset-root outputs\exam\dataset
# 应用处置到最终原文
venv\Scripts\python.exe scripts\apply_dispositions.py --m2-dir var\run\m2 --dataset-root outputs\exam\dataset --pages 26-35 --write
```

### M3 整岛翻译
```powershell
# 本地 Sakura（默认）；二次运行全缓存命中
venv\Scripts\python.exe scripts\m3_translate_pages.py --m2-dir var\run\m2 --dataset-root outputs\exam\dataset --pages 26-35 --output-dir var\run\m3 --precision local
# 在线 AI（提高精确度开关：整页打包编号一一对应，失败回退 local）
venv\Scripts\python.exe scripts\m3_translate_pages.py --m2-dir var\run\m2 --dataset-root outputs\exam\dataset --pages 26-35 --output-dir var\run\m3 --precision online --online-url <base_url> --online-model <model>
# 对照表
venv\Scripts\python.exe scripts\build_m3_table.py var\run\m3
```

### M4 排版 + state 重嵌
```powershell
# 批量排版（纯竖排双模式，min_ratio 0.6 可配）
venv\Scripts\python.exe scripts\m4_typeset_pages.py --dataset-root outputs\exam\dataset --erase-dir var\run\erase2 --translation-dir var\run\m3 --paddle-dir var\run\m2 --output-dir var\run\m4 --pages 26-35 --min-ratio 0.6
# 单页排版
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --typeset-only --erased-image var\run\33.erased.png --translation-json var\run\m3\33.translation.json --ocr-lines var\run\m2\33.paddle.lines.json --output var\run\33.typeset.png --audit-json var\run\33.typeset.audit.json
# 只改译文重嵌（跳过 OCR/擦字/翻译；哈希不符拒绝）
venv\Scripts\python.exe island_render.py --dataset outputs\exam\dataset\33 --reembed-state var\run\m4\33.state.json --erased-image var\run\erase2\33.erased.png --edit-translations edits.json --output var\run\33.reembedded.png
```

## 3. Paddle 子进程（独立调用）

```powershell
paddle_venv\Scripts\python.exe paddle_ocr_worker.py <image> <out.json> --device cpu
paddle_venv\Scripts\python.exe paddle_ocr_worker.py <manifest.json> <out.json> --crops --device cpu
# crops manifest: {"source": "...", "crops": [{"id": "...", "bbox": [x1,y1,x2,y2], "variants": ["raw","raw2x","clahe","otsu"]}]}
# 变体合并择优
venv\Scripts\python.exe scripts\merge_paddle_crops.py --crops-json out.json --output lines.json
```

## 4. 验收证据

`var\island_m1\`（M1-M4 全量验收产物）：
- `erase\` 10 页擦字 + 报告（岛外变化=0）
- `cv\` 双引擎互证表 + 标注图（18→8 REVIEW）
- `m2\` 原图重读、最终原文、dispositions、GUI 数据
- `m3\` 翻译对照表（58 translated + 4 kept-source，二次运行 58/58 缓存命中）
- `m4\` 10 页 typeset.png + state.json + REPORT_M4.md（59 typeset + 3 pending_review）

## 5. 常见参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-island-area` | 800 | 微岛舍弃（px²） |
| `--min-ratio` | 0.6 | 排版面积比下限（上限 1.1，硬 1.2） |
| `--precision` | local | 翻译引擎：local=Sakura / online=在线 AI |
| `--erase-strategy` | auto | solid-fill / inpaint / auto |
| `--max-layout-overflow` | 0 | dry-run 额外绘制放宽边界（仅显示） |
