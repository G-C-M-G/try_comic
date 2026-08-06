# 一键翻译流水线 — 使用说明

## 一、前置条件

| 依赖 | 说明 |
|---|---|
| Windows Photos | 采集阶段必须，用户在其中框选蓝框 |
| 本地 Sakura 服务 | `http://127.0.0.1:8080/v1`（OpenAI 兼容），模型 `sukinishiro`，启动：`start_sakura.bat` |
| manga-ocr | `venv` 内，模型离线缓存（`HF_HUB_OFFLINE=1`） |
| 原图目录 | 漫画页 jpg/png/webp，文件名数字排序 |

## 二、快速开始

```powershell
# 0) 先启动本地翻译服务
.\start_sakura.bat

# 1) stage1：采集（打开 Photos → 用户框选蓝框 → 自动抓取）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录"

# 2) 人工核对采集结果
#    打开 outputs\oneclick\{书名}\check\{page}.bluebox_check.png
#    蓝色 = 蓝框（正文单位，按 OCR 实际使用的扩 6px 版本绘制）；
#    红色 = 整页选中（封面等）；无蓝框页保持原样不处理

# 3) stage2：后台处理（OCR → 翻译 → 嵌字 → PDF）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --process

# 一步到位（采集完自动处理，跳过人工核对）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --auto
```

## 三、参数

```
--images <dir>      原图目录（默认：桌面《甲书》）
--pages <list>      页列表，如 "1-24" 或 "1,3,5"
--parallel <n>      Photos 并行窗口数（默认 3；6 窗口布局为上3下3）
--display-scale <p> 窗口显示缩放百分比（默认 25：窗口=页面×25%，整页完整可见，
                    原图文件不缩放、坐标还原原分辨率；100 = 固定平铺旧布局）
--capture-only      只跑 stage1（采集 + 核对图）
--process           只跑 stage2（后台处理），跳过采集
--auto              采集完成立即处理
--bluebox-fill R,G,B  填黑图填充色（默认 0,0,0 黑；若文字非黑色可用白 255,255,255）
```

## 四、采集注意事项（重要）

1. **蓝框 = 一个不可分割的 OCR 单位**：一个蓝框内应只有一句/一行文字。
   一句完整的话会被自动聚合为一个组（selection group）。
2. **只处理蓝框**：蓝框外的任何内容（SFX、装饰、页眉页脚、版权行）都会原样保留，
   不识别高亮岛、不翻译、不擦除、不嵌字。
3. **整页封面**：Photos 可能把整个页面框成一个大选区（红色标记）。
   这类选区标记为 `page_scale`，不参与 OCR 与嵌字；封面默认原样输出。
4. **误框**：框了没文字的区域（manga 读空 2 次）会被自动丢弃，不影响结果。
5. **黑页/无文字页**：全黑页或无文字插图页自动标记跳过，不处理。
6. **无蓝框页**：某页没有任何蓝框时整页保持原样（PDF 中为该页原图）；
   若该页实际有文字，说明采集时 Ctrl+A 蓝框提取失败，删除该页
   `capture/{page}/` 后重跑 stage1。
7. **窗口布局**：窗口尺寸 = 页面×25%（`--display-scale 25`），并行 6 窗口时上 3 下 3
   （下行贴屏幕底部、三列摊满屏宽）；请勿手动移动/遮挡窗口；
   程序会实时锁定窗口位置，被挤跑自动回位。

## 五、stage2 输出核对

```
outputs/oneclick/{书名}/
├── check/{page}.bluebox_check.png   # 采集核对（人工检查蓝框是否齐全）
├── typeset/{page}.typeset.png       # 嵌字结果图（逐页人工抽查）
└── {书名}.pdf                       # 最终 PDF
```

核对点：
- 蓝框内文字是否全部翻译并嵌字
- 蓝框外是否保持原样（无翻译、无擦除、无新增文字）
- 嵌字字号是否 ≤ 蓝框宽度（竖排）/ 蓝框高度（横排）
- 面积比 60%-110% 内 → 正常；不达标页面在 `typeset.png` 中字号偏小属预期

## 六、嵌字标准速查

| 规则 | 值 |
|---|---|
| 布局边界 | 组 bbox（外接矩形），块中心 = 组 bbox 中心 |
| 面积比（唯一） | A_text / A_group_bbox ∈ [60%, 110%]（质量信号；达标 typeset，否则 pending_review） |
| 硬上限 | 块宽/高 ≤ 组 bbox 宽/高 × 150%（每边各允超 50%）；仍放不下 → 30px 渲染并裁剪 |
| 字号下限 | 30px |
| 字号上限 | 竖排 ≤ 蓝框扩 6px 后的最窄宽；横排 ≤ 蓝框扩 6px 后的最矮高 |
| 对齐 | 竖排同句各列顶端对齐；横排各行左对齐 |
| 列位 | 蓝框扩 6px，x 中心 = 蓝框中心；中心横向容差 = 对角线 5% |
| 样式 | 黑字白描边（描边=字号//10），字距=字号//16，列距=字号//4（≥1，永不为 0） |

## 七、翻译缓存

- 位置：`outputs/oneclick/{书名}/translation_cache/`
- 缓存键：源文本 + 上下文 + 术语表哈希 + 提示词版本
- 删除 `translation_cache/` 可强制全部重新翻译
- 注意：缓存是**跨页共享**的，同文本不同上下文不会误命中；
  若曾用旧逻辑跑过且产生幻觉缓存（如源文本 "それでも、"），请清理后再跑

## 八、故障排查

| 现象 | 处理 |
|---|---|
| 采集阶段窗口乱跑 | 检查是否有其他程序抢占焦点；`--parallel 1` 单窗口重试 |
| 某页没有 bluebox_check.png | 该页黑页被跳过，属正常 |
| 翻译全部 kept-source（保留日文） | 检查 Sakura 服务是否启动（`Invoke-RestMethod http://127.0.0.1:8080/v1/models`） |
| 蓝框外出现中文 | 旧版本产物或幻觉缓存残留；用当前代码重跑 `--process` |
| 嵌字字太小 | 蓝框窄导致字号上限小，属约束预期；确认蓝框框得足够宽 |
| manga-ocr 失败 | 检查 venv 与模型离线缓存；`HF_HUB_OFFLINE=1` 下需先在线下载过 |
