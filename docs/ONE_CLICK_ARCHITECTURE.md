# 一键翻译流水线 — 架构说明

> 覆盖 `one_click_translate.py`（stage1 采集 + stage2 处理）与 `photos_capture.py`（窗口采集）。
> 本流水线以 **Photos 蓝框为 OCR 最小单位**：蓝框 = 用户框选的一次选区，不可分割；
> 蓝框之外的一切（轮廓岛）保持原样，不 OCR、不翻译、不擦字、不嵌字。

## 一、总览

```
原图目录 (--images)
   │
   ▼  stage1 (默认 / --capture-only)        ─── photos_capture.py
Photos 并行窗口 (上3下3, 行距=窗口高150%)
   └ 窗口固定为页面尺寸×25% (--display-scale, 原图文件不变)
   └ 扫描文本 → Ctrl+A 抓取蓝框 (重试 0.4/0.8/1.2s) → selection_blocks / selection_groups
   └ 生成 {page}.bluebox_check.png (蓝框蓝色 / 整页选中红色)
   │
   ▼  stage2 (--process)                    ─── one_click_translate.py
   1. build_bluebox_dataset → islands.json (纯蓝框聚合组, 无高亮岛识别)
   2. build_bluebox_masked → bluebox_masked.png (整页填黑图)
   3. manga-ocr 只读蓝框 (从填黑图裁剪, 蓝框扩6px), 读空2次跳过
   4. 蓝框按组拼接整句 → 方向判定 (宽≥高=横排, 否则竖排右→左)
   5. 本地 Sakura 翻译 (泄漏/重复/超长/短符号/3次上限 防护)
   6. 擦字 (只擦有蓝框的岛) → 嵌字 (两级边界 + 字号约束) → typeset.png
   7. 合并 {书名}.pdf
```

## 二、核心概念

| 概念 | 定义 |
|---|---|
| 蓝框 (blue box) | Photos 选区，`selection_blocks[]`，`page_scale=false` 的才是正文单位 |
| 整页选中 (page_scale) | Photos 误选了整页（封面/跨页），标记红色，不驱动 OCR 与嵌字 |
| 组 (group) | `selection_groups[]`，同一句子的蓝框集合（按句聚合） |
| 岛 (island) | `islands.json` 中的**蓝框聚合组**（纯蓝框，无高亮岛识别；蓝框外内容不产生岛） |
| 填黑图 | 整页一张，蓝框扩 6px 框内原像素、框外填充色（`--bluebox-fill` 默认黑） |
| 有框岛 | 岛 bbox 内含任一蓝框中心 → 擦字 + 嵌字 |
| 无蓝框页 | 无任何可用蓝框 → 整页保持原样（不 OCR、不翻译、不擦字、不嵌字） |

## 三、stage1 — 采集 (photos_capture.py)

- 并行 6 窗口：上 3 下 3，行距 = 窗口高 × 150%（防止遮挡）
- **显示固定 25%**：`--display-scale 25` 时每窗尺寸 = 页面 × 25%（宽高比一致），整页完整可见；
  原图文件不缩放、分辨率不变，坐标经 fit 变换还原回原图原分辨率；`--display-scale 100` 恢复固定平铺
- 窗口重锁 `_ensure_window_rect`：每次操作前核对窗口矩形，被挤跑自动回位
- 蓝框抓取重试：0.4s / 0.8s / 1.2s 三次，仍失败再试整页
- 黑页静默：`_is_black_page`（dark 占比 >0.90）→ 标记 `black_page`；无文字页（Photos 不出现
  OCR 控件/蒙版）→ 标记 `no_text`；两者 stage2 均跳过
- 大框细分：面积 > 页面 20% 的蓝框 → `page_scale` 标记；`_text_aware_split` 对大框做文字感知细分
- 结果：`capture/{page}/capture.json`（选区 + 元数据）、`check/{page}.bluebox_check.png`
  （只画蓝框蓝色 + 整页选中红色，**无高亮岛金框**；蓝框按 OCR 实际使用的扩 6px 版本绘制）

## 四、stage2 — 后台处理 (one_click_translate.py)

### 4.1 数据集构建
- `build_bluebox_dataset` → `dataset/{page}/islands.json`：**纯蓝框聚合组**（每个组 = 一个岛，
  蓝框扩 2px），不做差分高亮岛识别；`selection_blocks.json` 同步写入（嵌字列位来源）
- 无蓝框页 → 不生成 dataset，整页保持原样（PDF 中为该页原图）
- 失败页 fallback：`island_extract.build_fallback_dataset`（fallback_manga 标记页）

### 4.2 填黑图 (build_bluebox_masked, pad=6)
```
masked = 全页填充色 (BLUEBOX_FILL)
每个非 page_scale 蓝框扩 6px → 框内粘贴原像素
```
- 作用：manga-ocr 裁剪输入。蓝框内只含本框文字 → 邻框文本永不泄漏进裁剪

### 4.3 OCR (run_manga)
- 子进程加载 manga-ocr 一次，遍历全部 crops（JSON 行协议输出）
- **只读蓝框**：`{page}:box:{block_id}`，source = 填黑图，bbox = 蓝框扩 6px
- 无框岛**不进 OCR**（填黑图填充区是纯色，manga-ocr 会幻觉输出，如 "それでも、"）
- 读空 2 次跳过：同一蓝框连续两轮读空 → 判定为误选区，丢弃（`empty_counts`）

### 4.4 组拼接与方向 (page_texts / page_directions)
- 组内蓝框按方向排序拼接为整句：
  - 横排（宽≥高的框占多数）：y 升序 → x 升序
  - 竖排：x 降序（右→左）
- 岛文本 = 组拼接结果（一个岛可能含多组，按组序连接）
- 方向判定结果存入 `directions`，供嵌字用

### 4.5 翻译 (island_translate.py)
- 引擎：本地 Sakura（OpenAI 兼容，`http://127.0.0.1:8080/v1`，模型 `sukinishiro`），precision=local
- 防护（`_validate_island_translation`）：
  - 空输出 / 原文照抄 → 拒绝
  - 泄漏提示词字段（本页前文/术语表/人名表/翻译成…）→ 拒绝
  - 单行重复 >4 次 → 拒绝
  - 长度 > max(24, 源×2.5) → 拒绝
  - 假名残留 > max(2, 译文×15%) → 拒绝
  - 泄漏上下文文本 / 无关术语表目标词 → 拒绝
- 短符号岛（≤2 字符标点/符号）→ 零引擎调用，直接保留
- 每岛硬上限 3 次引擎调用 → 仍失败则 kept-source（保留源文本，绝不留空）
- 翻译缓存：`translation_cache/`（signature = 源文本 + 上下文 + 术语表 + 提示词版本）

### 4.6 擦字 (erase_and_typeset_page_with)
- `dataclasses.replace` 构造**只含有框岛的 dataset 子集** → `erase_island_image` 只擦有框岛
- 无框岛像素**保留原样**（原始内容不动）

### 4.7 嵌字 (island_typeset.py)
见下一节。

### 4.8 输出
- `typeset/{page}.typeset.png`、合并 PDF `outputs/oneclick/{书名}/{书名}.pdf`

## 五、嵌字标准（定稿）

### 5.1 布局边界（唯一，`_fit_size`）
- **直接使用组 bbox（外接矩形）**：译文块以组 bbox 为布局边界，块中心 = 组 bbox 中心
- 无蓝框几何（无线/无几何覆盖）时组 bbox 退化为岛 bbox

### 5.2 面积比（唯一算法）
- `ratio = A_text / A_group_bbox`（**唯一基准 = 组 bbox 面积**），目标 **60% – 110%**
- 达标 → `typeset`；不达标 → `typeset_pending_review`（仅质量信号，不再有 120% 面积硬上限）
- 字号从大到小搜索，取首个"包络内 + 比例达标"的字号；无达标字号取包络内比例最大者

### 5.3 硬上限（150% 包络）
- 块宽 ≤ 组 bbox 宽 × **150%** 且 块高 ≤ 组 bbox 高 × **150%**（每边各允许超出 50%）
- 超出包络的字号一律不选；下限字号仍放不下 → 以下限渲染并**裁剪到岛内**（绝不把擦掉的岛留白）
- **列间距/行距永不为 0**（`max(1, 字号//4)`），不靠压扁间距硬塞

### 5.4 字号约束（glyph 限制，扩框版）
- **字号下限 30px**：字号搜索不降到 30 以下
- **竖排**：字号（=字宽）≤ 蓝框**扩 6px 后**的最窄宽（`glyph_width_limit` = min 扩框宽，夹紧到岛边界）
- **横排**：字号（=字高）≤ 蓝框**扩 6px 后**的最矮高（`glyph_height_limit` = min 扩框高，夹紧到岛边界）
- 扩框版 = 原始蓝框尺寸 + 12px，字号上限更宽松

### 5.5 对齐与列位
- **竖排**：列内字自上而下，**同一句（同一岛内）各列顶端对齐**（共享顶边）
- **横排**：行内字自左而右，**各行左对齐**（短行不居中偏移），块整体以组中心水平居中
- 列位 = 蓝框扩 6px（`blue_boxes_to_lines pad=6`），列 x 中心 = 蓝框中心
- 中心容差：横向偏移 ≤ 组 bbox 对角线 5%（竖排只查横向，因列已顶端对齐）

### 5.6 样式（TypesetConfig）
- 黑字白描边：stroke = max(1, 字号//10)
- 字距 = 字号//16；列距/行距 = 字号//4
- 字体 msyh.ttc

## 六、数据契约

```
outputs/oneclick/{书名}/
├── capture/{page}/capture.json          # 选区 + 元数据 (bluebox_masked 前身)
├── check/{page}.bluebox_check.png       # 采集核对图
├── dataset/{page}/
│   ├── islands.json                     # 轮廓岛 (含组聚合)
│   ├── selection_blocks.json            # 蓝框 + 组
│   ├── original.jpg                     # 原图
│   └── bluebox_masked.png               # 填黑图 (OCR 输入)
├── translation_cache/                   # 翻译缓存 (signature 校验)
├── typeset/{page}.typeset.png           # 嵌字图
└── {书名}.pdf                           # 最终产物
```

## 七、已知坑（LESSONS 补充）

1. **manga-ocr 无检测能力**：无自带文本检测，必须靠蓝框定位裁剪。整页/纯色图会幻觉（如纯黑区读出 "それでも、"）——所以无框岛绝不进 OCR
2. **union 面积基准**：掩膜值是 255，`mask.sum()` 会放大 255 倍 → 必须 `np.count_nonzero`
3. **无框岛擦除泄漏**：`erase_island_image` 默认擦全部岛 → 嵌字前必须构造"有框岛子集" dataset
4. **填黑图裁剪**：蓝框裁剪从填黑图取，但**列位/字号约束用原始蓝框**（不扩 pad）——两者不要混用
5. **窗口遮挡**：并行窗口行距必须 > 窗口高，否则 Photos 弹出菜单会遮挡相邻窗口
