# OCR 引擎样图对比（8.jpg / 19.jpg）

更新日期：2026-08-05

## 1. 样图与数据位置

输入：`outputs\paddle_batch_input\8.jpg`、`19.jpg`（日文漫画页，非翻译效果，只展示 OCR 识别区域）。

输出目录 `outputs\ocr_compare_samples\`：

| 引擎 | 样图 | 明细 JSON |
|---|---|---|
| Windows WinRT OCR（已装日文语言包） | `8_winrt.png` / `19_winrt.png` | `*_winrt.json` |
| 微信 OCR（wxocr.dll 4.x） | `8_wechat.png` / `19_wechat.png` | `*_wechat.json` |
| PP-OCRv6 medium（PaddleOCR） | `8_paddle.png` / `19_paddle.png` | `*_paddle.json` |
| CTD + MangaOCR（现有管线） | `8_mocr.png` / `19_mocr.png` | `*_mocr.json` |
| Windows"照片"基准（参考） | `8_photos.png` / `19_photos.png` | `*_photos.json` |

样图规则：每个引擎用不同颜色画出检测框，框上标注编号、识别文字和置信度。图只表示**识别区域**，不代表擦字或嵌字效果。

## 2. 区域数量对比

| 页面 | WinRT | 微信 | PP-OCRv6 | MangaOCR | 照片基准 |
|---|---:|---:|---:|---:|---:|
| 8.jpg | 6 | 6 | 6 | 6 | 6 |
| 19.jpg | 30 | 28 | 30 | 28 | 29 |

说明：

- 8.jpg 共 6 列正文，五个引擎都检出 6 个区域，数量一致。
- 19.jpg 照片基准 29 个区域中含 1 个单字符（`し`）；PP-OCRv6 与 WinRT 检出 30 个（WinRT 多出的 `のイ`/`ーし` 与 PP-OCRv6 的空框、低置信 `ッ…` 同属小碎片区域）；微信与 MangaOCR 各 28 个。
- 2026-08-05 已成功安装日文 OCR 语言包（`Language.OCR~~~ja-JP~0.0.1.0`，安装过程出现的 0x80070002 是日文字体包的次要错误，系统已忽略），WinRT 现返回真实日文结果；旧版"仅中文语言包"的错字基线已废弃。

## 3. 文字质量对比（19.jpg 关键文本）

| 照片基准 | WinRT | 微信 OCR | PP-OCRv6 | MangaOCR |
|---|---|---|---|---|
| そのことを… | そのことを : | そのことを.. | そのことを… | そのことを．． |
| のですか? | のですか ? | のです加? | のですか？ | のですか？ |
| 黙っていた | 黙っていた | 黙っていた | 黙っていた | 黙っていた |
| それは… | それは : | そ机は... | それは… | それは．．． |
| 存在か…… | 存在か : | 存在か...... | 存在か…… | 存在か．．． |
| 有難く | 有難く | 有難 | 有難く | 有難く |
| おっと | おっと | おっと | おっと | おっと |
| 油断していた | 油断していた | 油断していた | 油断していた | 油断していた |

观察：

- **PP-OCRv6** 与照片基准几乎逐字一致，置信度普遍 0.99+，且正确识别 `ッ…`、空框说明其框检测更积极。
- **WinRT（日文包）** 正文全部识别正确，与 PP-OCRv6 同样检出 30 个区域；文字间带空格、句读 `。！？…` 显示为 `:`/`..`（引擎输出风格），框更细碎（如 `のイ`、`ーし` 拆成小块）。
- **微信 OCR** 整体准确，竖排列框与照片一致；少数错误如 `のです加?`（应是 `のですか?`）、`そ机は...`（应是 `それは…`）、`有難`（缺 `く`）。
- **MangaOCR** 与照片在正文上一致；标点省略号形式不同（`．．．` vs `…`），属于等效归一范围。

## 4. 引擎结论与部署状态

| 引擎 | 对本项目日文漫画的可用性 | 部署状态 |
|---|---|---|
| WinRT OCR | 装日文语言包后正文识别正确，可作复核/对照；框偏细碎需合并 | venv 已装 winrt 接口；`Language.OCR~~~ja-JP~0.0.1.0` 已安装（2026-08-05） |
| 微信 OCR | 可识别竖排日文，准确率接近照片基准，可作为高精度复核/补漏后端；已决定保留 | 封装 `wechat_ocr\`（wcocr.pyd + 自动发现封装，共 0.15 MB），调用微信自带 `wxocr.dll`，未安装任何额外模型 |
| PP-OCRv6 | 准确率高、置信度稳定，与照片一致性最好 | 已在 `paddle_venv`（既有部署） |
| MangaOCR+CTD | 正文准确，与照片一致，保留现有角色 | 既有部署 |

## 5. 微信 OCR 资源占用（实测）

| 项目 | 数值 |
|---|---|
| Python 调用进程内存 | 约 20 MB（RSS 16.5 → 19.8 MB） |
| 临时宿主进程 Weixin.exe | 峰值约 92 MB，调用后自动退出 |
| 合计峰值 | 约 110 MB |
| 单页识别耗时 | 首次 0.3 秒，后续约 1.5~2 秒 |
| 磁盘占用 | 引擎文件 49.5 MB（微信自带）+ 封装 0.15 MB（本项目） |
| CPU | 调用期间约 0% |

## 6. 复现命令

```powershell
venv\Scripts\python.exe ocr_compare_samples.py `
  "outputs\paddle_batch_input\8.jpg" "outputs\paddle_batch_input\19.jpg" `
  --engines winrt wechat paddle mocr photos `
  --out "outputs\ocr_compare_samples"
```

说明：

- 微信 OCR 封装 `wcocr.pyd` 来自 `swigger/wechat-ocr` 官方 release demo7，加载本机 `wxocr.dll`（微信 4.1.x 自带，未安装任何额外模型）。
- PP-OCRv6 通过 `paddle_venv` 独立子进程调用，CPU 推理（原管线默认）；GPU 路径在 `enable_mkldnn` 下有 onednn 兼容问题，与生产一致用 CPU。
- MangaOCR 通过 `background_ocr.py --no-paddle --no-manual-review` 子进程调用（纯 CTD+MangaOCR，不掺 Paddle）。

## 7. 未完成项

- Windows 日文 OCR 语言包已安装（`Language.OCR~~~ja-JP~0.0.1.0`，2026-08-05）。安装期间出现的 0x80070002 错误来自日文字体包（Fonts-Jpan）下载缺失，被系统忽略，不影响 OCR；若字体包也需补齐可执行：

```powershell
Add-WindowsCapability -Online -Name "Language.Fonts.Jpan~~~und-JPAN~0.0.1.0"
```

- 微信 OCR 已决定保留，封装位于 `wechat_ocr\`（`wcocr.pyd` 来自 `swigger/wechat-ocr` demo7 官方预编译，MIT），未安装任何额外模型；若未来不再需要，删除 `wechat_ocr\` 目录即可。
