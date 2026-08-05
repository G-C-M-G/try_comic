# 高亮岛采集与交接架构（ISLAND 交接文档）

更新日期：2026-08-05

## 1. 目标与边界

给"下一个项目"（漫画翻译嵌字）提供**高亮岛数据集**：翻译文字必须严格限制在高亮岛内。

- 本仓库（`D:\workplace\opencode\try_comic`）负责：照片"扫描文本"高亮岛采集、差分填黑/填蓝、
  岛几何（bbox/轮廓）提取、岛内文字识别（Windows OCR 优先、Paddle 兜底）与整句翻译验证。
- 下一个项目消费本仓库产出的数据集，不重新采集照片。
- **引擎优先级**：Windows.Media.Ocr(ja) > PaddleOCR > Windows Photos 扫描文本（默认关）。
- **轮廓标注功能默认关闭**（`--contours`），仅验证/调试时开启。

## 2. 数据流

```text
Windows Photos 扫描文本（一次前台采集，窗口最小化/自动关闭）
  -> photos_base.png（原图帧，无蒙版）+ photos_ocr.png（蒙版+高亮岛帧）
  -> 差分：dimmed = (delta>10 且变暗)；压暗占比 ≥15% 才认定蒙版生效
  -> 填黑/填蓝：压暗区域填背景色，未压暗（高亮岛）保留原像素
  -> 岛清单：非黑连通域 -> bbox/center/area（display + image 双坐标）
  -> [可选 --contours] findContours + approxPolyDP -> 岛多边形轮廓 + 标注图
  -> 数据集落盘（filled_black/filled_blue/original/islands.json/标注图）
  -> 岛内文字识别：Windows.Media.Ocr(ja) 优先，空结果自动回退 PaddleOCR
  -> 列聚类 -> Sakura 整句翻译 -> 擦字/嵌字（限制在岛内）
```

## 3. 模块

| 文件 | 职责 |
|---|---|
| `photos_capture.py` | 照片窗口自动化采集；`batch --fast` 批量（每页新窗、OCR 期间最小化、结束关闭）；`--keep-sfx` 保留拟声列；屏幕抓取覆盖层（`capture_as_image` 拿不到组合层蒙版） |
| `windows_ocr.py` | Windows.Media.Ocr(ja) 无窗口识别：`windows_ocr_predictions(path)` 返回行框+文本（confidence=None），需已装日文 OCR 语言包 |
| `island_extract.py` | 后台差分岛提取：`--fill-color`（任意颜色填底）、`--contours`（默认关，输出岛多边形） |
| `make_island_dataset.py` | 生成交接数据集：每页 filled_black/filled_blue（原图分辨率）+ islands.json + original + photos_ocr + [可选] 轮廓标注图 |
| `island_translate.py` | 数据集 → 岛内 OCR（`--engine windows-ocr` 默认/`paddle`）→ 列聚类 → Sakura 整句 → 擦字嵌字出片；Paddle 兜底自动切换 |
| `paddle_ocr_worker.py` | PaddleOCR 隔离子进程（整页 + `--crops` 裁剪重试 + `--serve` 常驻） |

## 4. 关键算法：高亮岛判定（稳定信号 = 差分，非亮度）

照片"扫描文本"机制：整页盖**灰色蒙版**，文字区保持**原亮度**（高亮）。

```text
dimmed = (mean|active - base| > 10) 且 (mean(active - base) < -4)   # 相对原图变暗
压暗占比 ≥ 15%  -> 蒙版生效（真实蒙版实测 51%-92%；UI 面板仅 ~3%）
岛 = 未压暗 且 原图亮度 > 145   # 暗噪点过滤
```

**亮度阈值方案已否决**：29 页 T200 亮区 70%、33 页 T190 亮区 53%——蒙版深浅随页变化，
绝对亮度不普适；差分（相对自身变化）10/10 页稳定复现。

## 5. 数据集契约（下一个项目的输入）

目录：`outputs\exam\dataset33\33\`（示例，全量见 `outputs\exam\masked10` 采集 + 数据集生成）

| 文件 | 说明 |
|---|---|
| `original.jpg` | 原图（985×1400 等） |
| `filled_black.png` / `filled_blue.png` | **原图分辨率**岛掩膜。默认：bbox 矩形填充；`--contours`（**默认关**，即"精准标注高亮岛轮廓"功能）开启后为形状保真：**非背景像素就是高亮岛的真实轮廓**（差分未压暗区域，最近邻缩放保形） |
| `photos_ocr.png` | 照片扫描文本覆盖层帧（参考） |
| `islands.json` | 每岛：`id`、`image_bbox`、`image_center`、`image_area`、`display_bbox` 等；`--contours` 时含 `image_polygon`/`display_polygon` 与 `fill_mode: "shape"` |
| `islands_annotated.png` | （`--contours` 时）原图上绘制岛轮廓+编号 |

`islands.json` 结构：

```json
{
  "schema_version": 1,
  "page": "33",
  "image_size": [985, 1400],
  "fill_black": [0, 0, 0],
  "fill_blue": [30, 144, 255],
  "contours": true,
  "islands": [
    {
      "id": "island_001",
      "image_bbox": [132, 31, 193, 156],
      "image_center": [162, 93],
      "image_area": 10209.0,
      "display_bbox": [89, 21, 130, 105],
      "image_polygon": [[140, 31], [132, 36], "..."]
    }
  ]
}
```

## 6. 使用说明

```powershell
# 1) 一次前台采集（窗口 OCR 期间最小化，结束自动关闭；不残留前台）
venv\Scripts\python.exe photos_capture.py batch --fast `
  --images "<原图目录>" --pages 26-35 --output-dir outputs\exam\masked10 --paddle-device cpu

# 2) 后台差分填黑（颜色可任意：R,G,B 或 #RRGGBB；--contours 默认关）
venv\Scripts\python.exe island_extract.py --input outputs\exam\masked10 `
  --output outputs\exam\islands10 --pages 26-35 --fill-color 30,144,255 [--contours]

# 3) 生成交接数据集（默认：bbox 矩形填图；`--contours` 开启"精准高亮岛轮廓"功能：
#    形状保真填图（非背景=岛精确轮廓）+ 岛多边形 + 标注图）
venv\Scripts\python.exe make_island_dataset.py `
  --images "<原图目录>" --capture outputs\exam\masked10 `
  --output outputs\exam\dataset --pages 26-35 [--contours]

# 4) 岛内文字翻译（Windows OCR 优先，空结果自动回退 Paddle）
venv\Scripts\python.exe island_translate.py `
  --images "<原图目录>" --capture outputs\exam\masked10 `
  --output outputs\exam\island_translated --pages 26-35 `
  --fill-color 30,144,255 --engine windows-ocr
```

## 7. 实测基线（26-35 十页，图源样张）

| 阶段 | 耗时 |
|---|---|
| 前台采集（10 页批量） | 230s（中位 21s/页，窗口即用即关） |
| 后台差分岛提取 | 0.56s / 10 页（0.04-0.06s/页，峰值内存 80MB） |
| 数据集生成（含轮廓） | 0.31s / 页 |
| 岛内翻译 | Windows OCR 页 ~10s、Paddle 兜底页 ~20s（10 页 128s） |

## 8. 已知限制

- Windows OCR 对艺术字/拟声页（31、34）读不出 → 自动回退 Paddle。
- 个别页 OCR 夹带 `:`/`・` 噪声字符与误读（如 `純然たる圓`）→ 照片 Ctrl+A 清单可仲裁
  （下一个项目的文字-框对应计划）。
- 岛轮廓（`--contours`）为近似多边形（approxPolyDP ε=2），用于验证展示，非几何权威。
