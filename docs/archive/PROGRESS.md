# 任务进度快照 (2026-08-01)

## 目标
日文漫画 → 自动翻译 + 嵌字为中文（引擎: MITS + sakura 本地部署）

## 已完成
- [x] Python 3.11.9 安装（%LOCALAPPDATA%\Programs\Python\Python311）
- [x] Git 2.55 安装（C:\Program Files\Git）并配置代理 127.0.0.1:7890
- [x] MSVC C++ Build Tools 2022 17.14（C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools）
- [x] venv: `D:\workplace\opencode\try_comic\venv`（Python 3.11 + torch 2.13.0+cu126 + MITS 全部依赖含 pydensecrf）
- [x] MITS 源码: `D:\workplace\opencode\try_comic\manga-image-translator`（main 分支）
  - 运行方式: 在仓库目录下 `..\venv\Scripts\python.exe -m manga_translator ...`
  - 运行时自动下载模型到 `./models`
- [x] sakura 7B v1.0 模型下载完成（3.96GB, 已验证 GGUF 魔数）
  - `sakura\models\sakura-7b-qwen2.5-v1.0-iq4xs.gguf`
- [x] llama.cpp b10216 CUDA 版解压: `sakura\llama\llama-server.exe`
- [x] llama-server 曾启动成功，监听 127.0.0.1:8080，但模型加载在 CPU（未识别 GPU）

## 当前问题 / 下一步
1. ~~GPU 加速~~ ✅ 已解决（cudart64_12.dll 放入 sakura\llama）
2. ~~"？？"乱码排查~~ ✅ 已解决：根源是 **PowerShell 5.1 发送中文 JSON 的编码问题**！
   - 排查中曾误判 llama-server 有 bug（实际 llama-server 可能没问题）
   - 最终方案：**llama-cpp-python (0.3.34, cu124) + 自建 FastAPI server**（sakura\sakura_server.py）
   - 服务运行: `venv\Scripts\python.exe sakura\sakura_server.py` → 127.0.0.1:8080
   - 已验证: "こんにちは、佐藤さん！" → "你好，佐藤同学！今天天气真好呢。" (1s, GPU)
   - ⚠️ 测试 API 必须用 Python 客户端（httpx/openai），勿用 PowerShell Invoke-RestMethod 发中文
3. **MITS demo 运行**（下一步）:
   - 环境变量: SAKURA_API_BASE=http://127.0.0.1:8080/v1, SAKURA_VERSION=0.10
   - 在 manga-image-translator 目录: `..\venv\Scripts\python.exe -m manga_translator local -i "样图" --use-gpu --translator sakura --target-lang CHS`
   - 首次运行自动下载模型到 ./models
4. 已安装: llama-cpp-python（依赖 cudart64_12.dll，需 sakura\llama 在 PATH）

## 手动文本框调整（PPT式）
- 状态导出: translate.py `--manual-dir`（MITS 新增 mid_pipeline_hook: 擦字后保存 擦字图+原图+文本框JSON）
- 调整器: adjust_gui.py（拖动移动/角点边点缩放/改字号/前后翻页/保存）
- 重嵌字: re_render.py（读取JSON→MITS render() 直接渲染→输出）
- 流程已端到端验证（181.jpg: 导出→改框改字号→重嵌成功）
- 使用说明/架构说明已更新；备份已同步

## 重新嵌字改造（2026-08-01 v2）
- re_render.py 改用 MITS 原生渲染（resize_regions_to_font_size + render）：
  保留旋转角度(斜框文字跟随气泡)、110%扩展、字号超框自适应缩小(下限6px)不裁字
- ✅ 验收：181.jpg 未调整状态重嵌输出与自动翻译输出同分辨率 99.6% 像素一致，差异全在文本框内
- 状态 JSON 新增 direction/alignment 字段（旧状态无字段自动兜底）
- workbench.py 新增：显示原文参照层(半透明原图)、选中框直接改译文、角度 spinbox(-45~45)
- adjust_gui.py 修复: _save 未叠加 box.pos() 导致拖动位移丢失的 bug

## 重新嵌字 v3 修复（2026-08-01）
- ✅ 修复"重嵌后旧译文残留"：MITS 渲染会原地修改 img_inpainted，状态导出的擦字图实际含中文译文
  → manga_translator.py 渲染前保存 img_inpainted_clean 副本，translate.py 状态导出改用纯擦字图
- ✅ workbench.py 新增进度条（翻译中/嵌字中 %v/%m），任务期间禁用编辑控件防覆盖
- ✅ re_render.py 新增 _allow_multi_line：按保存字号放宽行列上限（框放得下→保留字号多行列，放不下→按宽度上限自动缩字号），
  修复长译文被缩到 6~7px 蚂蚁字的问题
- 验收：181.jpg 未调整重嵌 vs 自动输出 差异仅 0.217%（仅一处 2 列排版改良），无双重文字、无变形

## 自动翻译与重新嵌字分离（2026-08-01 v4）
- ✅ translate.py 新增 `--analyze-only` 识别模式：输入已翻译图片 → 检测+OCR(48px 识别中文)+擦字，跳过翻译/渲染（免 sakura）
  - translator=original（译文=OCR 文本，绕过空译文过滤）、renderer=none、ocr=48px；`--analyze-ocr` 可换模型
- ✅ workbench.py 按钮改为「识别文本并导出状态」，不再重复翻译；进度条显示"识别中 X/Y"
- ✅ re_render.py 字号兜底：OCR 无字号时按框估算（竖排按列宽、横排按行高）
- 验收：已翻译 181.jpg → 识别 15 个区域，中文 OCR 准确，擦字图文字擦净，重嵌输出无双重文字（与原文差异 3.18%，为字号估算基线差异，调整阶段修正）

## 恢复全流程 + 双输出分离（2026-08-01 v5）
- 中文 OCR(48px) 识别效果不理想（漏检区域被过滤掉，工作台里连框都没有）→ **彻底删除 --analyze-only 识别模式**
- workbench.py 恢复「翻译并导出调整状态」全流程（mocr+sakura+擦字），需 sakura 服务
- gui.py 新增「同时导出调整状态」勾选：翻译时一并产出状态到 `桌面\翻译\调整状态\<输入名>\`（可编辑），
  **自动翻译与重新嵌字真正分离**：新图 gui.py 一次运行双输出，工作台直接载入；旧图工作台跑一次全流程补状态（同一管线结果一致）

## 编辑器交互修复（2026-08-01 v6）
- ✅ 修复"改译文/字号不实时显示"：根因是 box.isSelected() 不可靠——点击落在文字预览图/角点手柄上时 Qt 场景选中不到
  TextBox。改为 workbench/adjust_gui 显式记录选中框索引（_selected_idx，空白点击清除），
  改译文/字号/角度实时刷新框内预览并正确保存
- ✅ workbench 启动不再自动载入默认状态目录（移除 _auto_load）

## 输出格式选择（2026-08-01 v7）
- gui.py 新增「输出格式」下拉框：与原图相同（默认，沿用输入扩展名）/ png / jpg / webp，透传 `--format`

## 重嵌字号真实化 + 输出格式（2026-08-01 v8）
- ✅ 修复"改字号输出没变"：根因是 MITS render() 把文字图透视拉伸填满文本框（warp-to-box），字号只影响断行
  → re_render.py 改为**自然尺寸渲染**：按 JSON 字号真实大小绘制、居中于框中心、按角度旋转，改字号真正生效
  （附带效果：消除自动输出的拉伸变形像素，竖排/横排均验证字号 22→44 墨量提升 3.6x）
- ✅ 重嵌输出格式可选：re_render.py 新增 `--format png|jpg|webp|auto`；状态 JSON 记录原图扩展名(src_ext)，
  默认 auto=与原图相同（无记录则 png）；workbench 输出行加格式下拉框，命令行目录模式逐页解析
- 注意：未调整状态重嵌与自动输出存在布局差异（真实字号 vs 拉伸填框），属预期行为

## 重嵌所见即所得（2026-08-01 v9）
- ✅ 修复"输出文字明显小于22"：根因是 freetype 竖排 vertAdvance≈0.7×字号（紧凑），而预览按 1.2×字号排字
  → 新增 text_paint.py（PIL 渲染，工作台预览与 re_render **共用同一模块**，横/竖排间距均为 1.2×字号），
  输出与 GUI 预览完全一致（含描边、换行分列）
- ✅ 拖动/缩放文本框 → 字号自动适配（横排按 0.75×框高、竖排按 0.9×框宽），预览实时刷新并保存，
  输出与 GUI 对应；纯移动不改字号
- ✅ 旋转方向与检测框一致（PIL rotate(-angle)），单列/多列竖排均验证间距≈1.2×字号

## 前端 GUI (gui.py, PySide6)
- 输入选择(文件/文件夹/zip) + 选项(字号/拟声词/覆盖/输出目录) + 开始/停止 + 日志区 + 服务状态
- 后台 QThread 调 translate.py 子进程，实时日志，完成弹窗
- start_gui.bat 一键启动；GUI 运行已验证
- 架构说明.md：钩子点/未来规划(漏检补全/打包/质量提升)

## 版本备份
- ✅ `backup_字体未调整\`（2026-08-01, 1GB）：字体调整前版本（min字号未定/字号可放大1.5x/竖排从右到左）
- ✅ `backup_字体调整\`（2026-08-01, 1GB）：字号/竖排算法版
- ✅ `backup_翻译初版\`（2026-08-01, 11.9GB）：完整可运行版（含 sakura 模型+GUI+音译+拟声词）
- ✅ `backup_翻译2.0-重新嵌字\`（2026-08-02, 11.8GB）：重新嵌字完整版
  （状态导出/工作台调整/re_render 所见即所得/字号真实化/格式选择/图片转pdf，不含 venv）
- 清理: 已删除 logs/demo_output/_t.py/_err.txt/运行日志/__pycache__ 等冗余文件（sakura 运行中的 server 日志除外）

## 空间优化 + 服务重启验证（2026-08-02）
- ✅ 工作区 46.5GB→31.1GB（-15GB）：删未用 iq4xs 模型×3(11.9GB)、venv site-packages/bin 重复 dll(920MB)、
  卸载 rusty_manga_image_translator/bitsandbytes/accelerate(445MB)、pip cache(4.5GB)、llama CLI 工具(15MB)
- ✅ 重启验证：venv llama_cpp 加载+GPU 推理正常（"こんにちは…"→"佐藤同学你好！…" 0.8s），health OK
- ⚠️ 环境发现：本机存在「系统 Python311 子进程」现象——venv 进程运行 sakura_server.py 时会出现系统 python
  进程占用 8080（python 代码层 audit hook 铁证无 spawn，属环境层异常；服务功能不受影响，此前用户环境一直如此）
- ⚠️ 备注：bash 工具/计划任务会在命令结束后清理进程树，独立启动服务需用 WMI（Invoke-CimMethod Win32_Process Create）；
  临时目录勿放 bisect.py 等同名文件（会劫持 sys.path 导致 import 失败）

## 字体
- 默认回退字体链: Arial-Unicode → msyh.ttc(微软雅黑,中文) → msgothic.ttc，中文无缺字风险
- 最小字号: translate.py `--font-min` 默认 22px（config font_size_minimum 同步）

## 人名音译 (name_translation.py)
- 规则: 翻译失败(译文为空/与原文相同)且含片假名 → 按音节映射中文音译（阿/伊/卡/塔…）
- 位置: MITS 过滤前 transliteration_hook（新加钩子），汉字/英文原样保留
- 已验证: アリス→阿莉苏, マリア→玛莉阿, 佐藤アリス→佐藤阿莉苏, シェリー→谢莉
- 报告输出到 sfx_report.txt（[音译] 前缀）

## 批量验证
- ✅ zip 输入→解压→批量翻译→重打包→序号输出（test_chapter.zip → 桌面\翻译\20.zip, 2页完整）

## 后续里程碑
- [x] ~~验证 sakura API 翻译质量~~ ✅ llama-cpp-python 方案稳定（弃用 llama-server，其在本机生成异常）
- [x] ~~MITS 单图跑通~~ ✅ 4 张样图全部成功（含困难页）
- [x] ~~拟声词智能过滤算法~~ ✅ sfx_filter.py（v2: 阈值190/占比0.5）
  - 规则: 纯假名判定拟声词；>10 个全部不翻译；气泡内（框扩50%白色多）翻译，其余保留原文
  - 诊断数据: 气泡多为米白(亮度185~205)，阈值 190 比 200 准
- [x] translate.py 封装（文件/文件夹/zip/cbz + 弹窗通知）
  - 用法: venv\Scripts\python.exe translate.py <输入> [输出] [--overwrite] [--no-sfx]
  - 完成后 Windows 弹窗通知（不弹图），--no-popup 关闭
  - 已验证: 单文件 ✅ zip 输入+重打包 ✅
- [ ] start_sakura.bat 已建（需用户实测）
- [ ] 中文字体配置（当前默认，暂未见缺字）
- [ ] 用户试用 + 质量调参

## 环境备忘
- 系统代理: 127.0.0.1:7890（git 已配 http.proxy/https.proxy）
- pip 安装需临时设 HTTP_PROXY/HTTPS_PROXY
- aria2c: C:\Users\YLP\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\aria2-1.37.0-win-64bit-build1\aria2c.exe
- 显卡: RTX 4060 Laptop 8GB, 驱动 610.74
