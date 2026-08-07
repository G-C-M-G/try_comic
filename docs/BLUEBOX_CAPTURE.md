# 蓝色框采集要点（BLUEBOX_CAPTURE）

> 本页沉淀 Windows Photos "扫描文本" 蓝色选区框的采集经验与检测规则。
> 适用流程：`photos_capture.py batch --fast` -> `one_click_translate.py`。

## 1. 帧采集基础

每页采集三个屏幕帧（`capture/<page>/`）：

| 帧 | 含义 | 用途 |
|---|---|---|
| `photos_base.png` | 普通浏览帧（无扫描） | 严格公式的"原始像素"基线 |
| `photos_ocr.png` | 扫描文本模式帧（灰蒙版 + 高亮岛） | delta 兜底的"原始"帧 |
| `photos_selected.png` | Ctrl+A 全选帧（蓝色选区叠加） | 蓝框检测的目标帧 |

**关键**：蓝框检测比较的是"扫描文本后的图 vs 全选文本的图"
（`photos_ocr` vs `photos_selected`），**不是原图**（`photos_base`）。

## 2. 窗口布局

- 进程以物理像素运行（DPI-aware）：布局按真实分辨率（如 2560x1600）。
  勿用 DPI-unaware 工具（PowerShell 等）读到的缩放值（1707x1067 之类）。
- `--display-scale <n>`：窗口 = 页面 x n%，保持宽高比，整页可见。
- `--parallel 6`：3 列 x 2 行并行；行内摊铺、第二行贴底。
- `--layout-row-gap <px>`：固定列间距（默认摊铺铺满）。
- `--layout-bottom-shift-up <px>`：第二行上移（贴底基础上）。
- 经验值：竖版 1280x1790 页、6 窗并行用 `--display-scale 42`。

## 3. 亮化补偿（重要教训）

Photos 窗口显示图片时有**自适应提亮**：

```
frame ≈ a × 原图 + c      # 逐通道拟合，a≈0.72~1.09，c≈25~87（逐页不同）
```

- 屏幕截图帧明显比原图亮（实测 mean 217 vs 182）。
- **教训**：对帧做"补偿"（反变换回原亮度）后，严格混合公式在补偿帧上
  **不成立**（蒙版偏移 38/67 是显示域参数），导致严格算法 0 框。
  曾因此被迫重采 1-30 页。
- **结论**：**不要补偿帧文件**。严格公式直接在原始（未补偿）帧上成立；
  人眼审查用 check 图（直接叠加在原图上）即可。

## 4. 蓝框检测：严格算法优先，delta 兜底

`detect_selection_boxes(active=photos_ocr, selected=photos_selected, base=photos_base)`：

### 4.1 主路径（严格蓝色蒙版公式，base 给定时）

Photos 蓝色选区 = 蒙版 RGB(0, 120, 211) 以 α=81/255 正常混合：

```
keep = 174/255
c' = keep × c + (0, 38, 67)     # 蒙版后颜色
```

判定：

```
|selected - (keep × base + (0,38,67))| ≤ 12        # 公式匹配（容差）
max|selected - base| > 2                            # paint-delta 守卫
```

- paint-delta 排除"页面固有纯蓝"（其选区前后 ~恒等，不重绘）。
- 容差 12 为实测安全值（真实框残差 ≤3；过紧如 2 会碎框、掉命中率）。

### 4.2 兜底（无 base 或公式无结果）

```
|selected - photos_ocr| 任一 RGB 通道 > 8    # 两帧差异
```

- 适用：预选中态页（ocr 帧已含蓝色，公式无解）、非标准渲染页。
- 部分页（如 128/132/144）只能靠兜底出框——masked 生成时用
  capture.json 已存的兜底框。

### 4.3 整页大框过滤

连通域面积 > 帧面积 20% 的整页选中框**直接丢弃**（不是原子文字框）。

## 5. 硬规则：三次无蓝像素 -> 人工审核

- Ctrl+A 三次重试仍无蓝色像素（`_capture_selection_boxes` 三次尝试）：
  `capture.json` 写 `selection_review_required: true`。
- 该页**不进 stage 2**；stage 1/2 输出"待人工审核"提示与确认命令：
  ```
  python one_click_translate.py --images <漫画> --confirm-no-boxes <页>
  ```
- 用户确认无框后写 `selection_review_confirmed: true`，页才可关闭
  （后续跳过不再警告）。
- 常见诱因：Photos 未识别文字/选区渲染失败（重采通常可恢复）。

## 6. masked（扩框填黑图）生成

- 蓝框 `image_bbox` 必须映射回**原图坐标系**
  （`_display_bbox_to_image`：fit 缩放 + 居中偏移）。
- 框扩 6px（与 OCR 读取一致），框内保留原像素，其余填黑：
  `{page}.bluebox_masked.png`（原图分辨率）。

## 7. 常见问题速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 采集帧比原图亮 | Photos 显示提亮 | 正常；审查用 check 图 |
| 某页 0 框 | Ctrl+A 未渲染蓝选区 | 重采；仍无则走审核确认 |
| 严格公式 0 框但 delta 有框 | 预选中态/非标渲染 | 用 capture.json 兜底框 |
| 框位置对但尺寸错 | image_bbox 未映射原图坐标 | 用 _display_bbox_to_image |
| 补偿帧后严格算法全 0 | 补偿破坏显示域参数 | 不要补偿帧文件，重采 |
