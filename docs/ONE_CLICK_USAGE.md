# 一键翻译流水线 — 使用说明

## 一、前置条件

| 依赖 | 说明 |
|---|---|
| Windows Photos | 采集阶段必须，用户在其中框选蓝框 |
| AI 翻译 Key（推荐） | DeepSeek 等 OpenAI 兼容接口；填入 GUI 设置或 `config/api_keys.json`；**配置后翻译走 AI 整页批量，本地 Sakura 兜底** |
| 本地 Sakura 服务 | 兜底翻译引擎：`http://127.0.0.1:8080/v1`（OpenAI 兼容），模型 `sukinishiro`，启动：`start_sakura.bat` |
| manga-ocr | 本地 OCR，`venv` 内，模型离线缓存（`HF_HUB_OFFLINE=1`） |
| AI OCR（可选） | OpenAI 兼容视觉模型（如 Qwen-VL），替代/补强 manga-ocr 对漏读框的识别 |
| 原图目录 | 漫画页 jpg/png/webp，文件名数字排序 |

## 二、快速开始

```powershell
# 0) 先启动本地翻译服务
.\start_sakura.bat

# 1) stage1：采集（打开 Photos → 用户框选蓝框 → 自动抓取）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录"

# 2) 人工核对采集结果
#    打开 outputs\oneclick\{书名}\check\{page}.bluebox_check.png
#    蓝色 = 蓝框（正文单位，按 OCR 实际使用的扩 6px 版本绘制，叠加在原图上）；
#    若提示"待人工审核"页（三次采集无蓝色像素），查看后确认无框：
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --confirm-no-boxes 100,120

# 3) stage2：后台处理（OCR → 翻译 → 嵌字 → PDF）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --process

# 一步到位（采集完自动处理，跳过人工核对）
venv\Scripts\python.exe one_click_translate.py --images "C:\原图目录" --auto
```

## 三、参数

> CLI 默认值见下表；**GUI 默认值与 CLI 不同**：并行窗口 6、显示缩放 42%。

```
--images <dir>      原图目录（默认：D:\workplace\opencode\try_comic\漫画翻译\原文\《甲书》）
--pages <list>      页列表，如 "1-24" 或 "1,3,5"（采集与处理共用；单页重采见下文）
--parallel <n>      Photos 并行窗口数（CLI 默认 3；GUI 默认 6；6 窗口布局为上3下3）
--display-scale <p> 窗口显示缩放百分比（CLI 默认 25；GUI 默认 42：窗口=页面×42%，整页完整可见，
                    原图文件不缩放、坐标还原原分辨率；100 = 固定平铺旧布局）
--layout-row-gap <px>        固定列间距（默认摊铺铺满；右列左移用更小值）
--layout-bottom-shift-up <px> 第二行上移量（贴底基础上，默认 0）
--layout-row-spacing <px>    上下两排窗口的垂直间距（默认 200）
--layout-right-gap <px>      最右列窗口距右边缘的间隙（默认 100，防弹出菜单遮挡）
--capture-only      只跑 stage1（采集 + 核对图）
--process           只跑 stage2（后台处理），跳过采集
--stage {ocr,translate,typeset}  只跑 stage2 的某一个阶段（--pages 选出该阶段页；前置产物必须已存在，缺前置的页会跳过并提示）
--auto              采集完成立即处理
--confirm-no-boxes <pages>   人工审核确认某页确无蓝色框（标记后 stage2 静默跳过）
--ai-url <url>       AI 翻译接口（默认 https://api.deepseek.com/v1）
--ai-model <name>    AI 翻译模型（默认 deepseek-chat）
--ai-api-key <key>   AI 翻译 Key（或 DEEPSEEK_API_KEY 环境变量 / config/api_keys.json）
                    设置后翻译优先 AI 整页批量，失败回退本地 Sakura；不设则全部走本地
--ai-ocr-url <url>   AI OCR 视觉接口（OpenAI 兼容，如 GLM-4V/Qwen-VL；留空=本地 manga-ocr）
--ai-ocr-model <name> AI OCR 视觉模型
--ai-ocr-api-key <key> AI OCR Key（或 AI_OCR_API_KEY 环境变量 / config/api_keys.json）
--ocr-engine {auto,windows-ocr,paddle}  采集期遮罩页 OCR 引擎（默认 auto：Windows OCR 优先、
                    Paddle 兜底；强制指定某引擎时，不可用/读不出即报错停止，不回退）
--stage2-engine {manga-ocr,ai-ocr}  stage2 读蓝框文字引擎（默认 manga-ocr；ai-ocr 需配置
                    --ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key，未配置即报错）
--resume             断点续跑：跳过已有 typeset 输出的页
--overwrite          覆盖旧结果（与 --resume 互斥）：重采/重做选中页，自动清除旧产物，无需手动删除文件夹
--bluebox-fill R,G,B  填黑图填充色（默认 0,0,0 黑；若文字非黑色可用白 255,255,255）
```

## 三.5 运行进度与断点

- 每次运行在 `outputs/oneclick/{书名}/run.json` 实时写入进度
  （阶段/当前页/计数/耗时/ETA），GUI 的进度条与终端提示据此显示；
  `run.log` 滚动保留运行日志。
- `--resume`（GUI"断点续跑"勾选）：跳过已有 `typeset/{page}.typeset.png` 的页；
  翻译命中缓存天然跳过已翻内容；**OCR 读数缓存**（`ocr_readings.json`）
  使重跑跳过已识别的框——断点续跑近乎零成本。
- 单页失败不中断整批：失败页记录并在 PDF 中保持原图。
- `--max-workers 1-3`：翻译+嵌字按页并行（AI 在线翻译收益最明显）。

## 四、采集注意事项（重要）

1. **蓝框 = 一个不可分割的 OCR 单位**：一个蓝框内应只有一句/一行文字。
   一句完整的话会被自动聚合为一个组（selection group）。
2. **只处理蓝框**：蓝框外的任何内容（SFX、装饰、页眉页脚、版权行）都会原样保留，
   不识别高亮岛、不翻译、不擦除、不嵌字。
3. **蓝框检测单路径**（详见 `BLUEBOX_CAPTURE.md`）：严格蓝色蒙版公式
   `|selected - (keep×原图+(0,38,67))| ≤ 5`，keep=174/255；
   缺 `photos_base.png` 或三次 Ctrl+A 仍无蓝色像素 → 该页进审核门。
4. **整页大框**：覆盖帧面积 >20% 的整页选中框直接丢弃（非正文单位，封面原样输出）。
5. **误框**：框了没文字的区域（manga 读空 2 次）会被自动丢弃，不影响结果。
6. **黑页/无文字页**：全黑页或无文字插图页自动标记跳过，不处理。
7. **待人工审核页**：三次 Ctrl+A 仍无蓝色像素 → 提示待审核、不进 stage2；
   查看该页截图确认确实无框后，`--confirm-no-boxes <页>` 确认关闭；
   若其实有文字，删除 `capture/{page}/` 后重跑 stage1。
8. **窗口布局（auto-fit）**：`--display-scale` 是窗口缩放**上限**——布局会在
   `--layout-region`（默认 2400x1400）区域内自动缩小到不重叠（窗间距 32px）；
   每次抓帧前自动重锁窗口矩形，被挤跑自动回位；请勿手动移动/遮挡窗口。

## 五、stage2 输出核对

```
outputs/oneclick/{书名}/
├── capture/{page}/              # 采集帧 + capture.json（每页 base/ocr/selected 三帧）
├── check/{page}.bluebox_check.png   # 采集核对（原图 + 蓝框叠加）
├── dataset/{page}/              # islands.json（分组）、selection_blocks.json、original.jpg、填黑图
├── translation_cache/           # 本地翻译缓存
├── translation_batch_cache/     # AI 整页批量翻译缓存
├── ocr_readings.json            # OCR 读数缓存（非空读数；重跑免 OCR）
├── translations.json         # 书级译文落盘（{page: {island_id: 译文}}；翻译阶段/全链写入，嵌字阶段读取）
├── typeset/{page}.typeset.png       # 嵌字结果图（逐页人工抽查）
└── {书名}.pdf                       # 最终 PDF
```

核对点：
- 蓝框内文字是否全部翻译并嵌字
- 蓝框外是否保持原样（无翻译、无擦除、无新增文字）
- 嵌字字号是否 ≤ 蓝框宽度（竖排）/ 蓝框高度（横排）
- 面积比 60%-110% 内 → 正常；不达标页面在 `typeset.png` 中字号偏小属预期

## 五.5 常见重跑操作（缓存友好）

| 目的 | 操作 | 代价 |
|---|---|---|
| 只重嵌字（改样式/字号/规则后） | `--process --pages ... --no-pdf`（OCR/翻译全命中缓存） | 分钟级 |
| 改分组规则后全书重分组 | 删 `dataset/*/islands.json` → `--process` | 仅合并组重译（API）+ 重嵌字，OCR 0 |
| 强制重翻译 | 删 `translation_cache/` + `translation_batch_cache/` | 全书 API 重翻 |
| 强制重识别 | 删 `ocr_readings.json` | 全书 ~40 分钟 OCR |
| 断点续跑 | `--resume`（跳过已有 typeset 页） | — |
| 单页调试 | `--process --pages 12 --no-pdf` | 分钟级 |
| **单页/多页重采**（蓝框不准时） | 删 `capture/<页>/`（必要时同时删 `dataset/<页>/`）→ `--pages 41` 重跑 stage1 | 单页 ~20s 采集 |
| 只重 OCR 若干页 | `--stage ocr --pages 5-10` | 秒级~分钟级 |
| 只重翻译若干页 | `--stage translate --pages 5-10 --overwrite` | 缓存命中零引擎费 |
| 只重嵌字若干页 | `--stage typeset --pages 5-10 --no-pdf` | 分钟级 |
| 重采若干页（覆盖旧） | `--pages 41 --overwrite`（原：删 `capture/<页>/` 后重跑） | 单页 ~20s |

## 五.6 蓝框重采流程（本次实测）

当某页 `check/<页>.bluebox_check.png` 蓝框不准（框错/漏框/框到无关区域）时：

```powershell
# 1) 覆盖重采（推荐）：--pages 41 --overwrite 直接重采选中页——自动清除 capture\41、check 图、
#    dataset\41，并剔除 ocr_readings.json / translations.json 中该页键，无需手动删除文件夹
#    备选（原删除方式）：手动清除旧数据后重跑
Remove-Item capture\41 -Recurse; Remove-Item dataset\41 -Recurse

# 2) 单页重跑 stage1（只打开该页的 Photos 窗口，约 20 秒）
venv\Scripts\python.exe one_click_translate.py --images "原图目录" --pages 41 --parallel 6 --display-scale 42

# 3) 核对新蓝框
#    打开 check\41.bluebox_check.png（蓝色 = 蓝框，按 OCR 实际扩 6px 版绘制）

# 4) 确认无误后单页重跑 stage2（OCR/翻译缓存命中，只重嵌该页）
venv\Scripts\python.exe one_click_translate.py --images "原图目录" --pages 41 --process --no-pdf
```

- 采集是 GUI 自动化（打开真实 Photos 窗口），期间**不要移动/遮挡窗口**；
- 重采后 `dataset/`、`ocr_readings.json` 中旧页键自动失效，重跑即重建，无需手动清；
- 若整页确实无文字/全黑（如插图页），删除 `capture/<页>/` 后无需重采，stage2 自动按原图跳过。

## 五.7 备份与清理

**备份**（产物备份 = 整个 `outputs/oneclick/{书名}/` 目录，含 capture/dataset/缓存/typeset/PDF，复制即完整备份）：

```powershell
# 整库备份（含全部书 + config）到项目内 backups/（已被 .gitignore 忽略）
robocopy outputs\oneclick backups\2026-08-09\oneclick /E /NFL /NDL /NJH /NP /R:1 /W:1
robocopy config backups\2026-08-09\config /E /NFL /NDL /NJH /NP /R:1 /W:1
```

- 代码/规则版本走 git（`config/api_keys.json`、`outputs/` 已在 .gitignore，不提交）；
- 恢复 = 把备份目录铺回原路径即可继续（缓存命中零重算）。

**可安全清理的冗余**：

| 文件 | 说明 |
|---|---|
| `crops.*.json`（书目录下，多个 PID 后缀） | 每次运行产生的 OCR 临时任务文件，跑完即可删 |
| `_logs/`（书目录下） | 旧分段处理日志，可删 |
| `check/*.erased.png`、`*.erasure_mask.png`、`*.ranges_overlay.png`、`*.islands_grouped.png` | 旧管线调试图，新管线不再生成，可删 |
| `build/`（项目根） | PyInstaller 打包中间产物（dist/ 下的 exe 才是成品），可删 |
| `check/` 中同页的旧格式 jpg | 新采集生成 `*.bluebox_check.png` 后，同页旧 `*.bluebox_check.jpg` 属旧管线遗留，可删 |

**不可删**：`dataset/`、`typeset/`、`ocr_readings.json`、`translation_cache/`、`translation_batch_cache/`（删了就要重算）。

## 六、嵌字标准速查

| 规则 | 值 |
|---|---|
| 蓝框分组 | 扩框(±6px)间距 < 20%×较小扩框宽/高 且 重叠 > 40%×较大扩框高/宽；混合方向与沿阅读方向堆叠不合并 |
| 统一框 B | 每岛蓝框组并集 ±6px；擦字范围 = 布局边界 = 合成裁剪 = 面积比基准 = B |
| 面积比（唯一） | A_text / A_B ∈ [60%, 110%]（质量信号；达标 typeset，否则 pending_review） |
| 硬上限 | 块必须完整在 B 内（宽/高 ≤ B 宽/高）；放不下 → 缩字号至 12px 绝对下限，仍放不下才允许 150% 包络（pending_review） |
| 字号下限 | 首选 28px，可降到 12px |
| 字号上限 | 竖排 ≤ 蓝框扩 6px 后的最窄宽；横排 ≤ 蓝框扩 6px 后的最矮高 |
| 对齐 | 竖排同句各列顶端对齐；横排各行左对齐；块以 B 中心居中 |
| 列位 | 整组居中均分（不跟随各蓝框位置）；中心横向容差 = B 对角线 5% |
| 样式 | 黑字白描边（描边=字号//10），字距=字号//16，列距=字号//4（≥1，永不为 0） |

## 七、翻译缓存

- 位置：`outputs/oneclick/{书名}/translation_cache/`（本地兜底）+ `translation_batch_cache/`（AI 整页批量）
- 缓存键：源文本 + 上下文 + 术语表哈希 + 提示词版本
- 删除 `translation_cache/` 可强制全部重新翻译
- 注意：缓存是**跨页共享**的，同文本不同上下文不会误命中；
  若曾用旧逻辑跑过且产生幻觉缓存（如源文本 "それでも、"），请清理后再跑

## 七.5 OCR 读数缓存

- 位置：`outputs/oneclick/{书名}/ocr_readings.json`
- 内容：`{"{page}:box:{box_id}": "识别文本"}`，**只存非空读数**（空框每次照常复读判误选）
- 用途：任何重跑（重分组/重嵌字/重合并 PDF）**跳过 OCR**，识别零成本
- 删除该文件可强制全书重新识别（~40 分钟）；重新采集过的页旧键自动失效

## 八、故障排查

| 现象 | 处理 |
|---|---|
| 采集阶段窗口乱跑 | 检查是否有其他程序抢占焦点；`--parallel 1` 单窗口重试 |
| 某页没有 bluebox_check.png | 该页黑页被跳过，属正常 |
| 翻译全部 kept-source（保留日文） | 检查 Sakura 服务是否启动（`Invoke-RestMethod http://127.0.0.1:8080/v1/models`） |
| 蓝框外出现中文 | 旧版本产物或幻觉缓存残留；用当前代码重跑 `--process` |
| 嵌字字太小 | 蓝框窄导致字号上限小，属约束预期；确认蓝框框得足够宽 |
| manga-ocr 失败 | 检查 venv 与模型离线缓存；`HF_HUB_OFFLINE=1` 下需先在线下载过 |
