# 四页端到端验收报告

日期：2026-08-04

命令：

```powershell
.venv\Scripts\python.exe batch.py `
  --images samples\images `
  --ocr-dir samples\ocr `
  --output-dir artifacts\batch `
  --translator mock `
  --mock-translations-dir config\sample_translations `
  --glossary config\glossary.json `
  --max-layout-overflow 10
```

验收译文使用确定性整句备用基准，产物如实记录 backend=`mock`。默认 Sakura 接口仍是生产默认值；
本次基准不采用 Sakura 的原始候选，因为候选曾出现列标记丢失、缺标点和人物指向错误。

## 总结

| 页 | 完整句 | OCR 列 | 每列实际变更 | 排版框 >= 80% | 同句统一基础字号 | 整句中心一致 | 使用 20% | 20% 外像素 | 像素检查 |
|---|---:|---:|---|---|---|---|---:|---:|---:|
| 1 | 6 | 18 | 是 | 是 | 是 | 是 | 0 | 0 | 0.080s |
| 8 | 1 | 6 | 是 | 是 | 是 | 是 | 0 | 0 | 0.035s |
| 19 | 6 | 28 | 是 | 是 | 是 | 是 | 0 | 0 | 0.119s |
| 22 | 7 | 18 | 是 | 是 | 是 | 是 | 0 | 0 | 0.084s |

四页共 20 个完整翻译单元、70 个保留 OCR 列。擦字图在组合 mask 外的像素变化均为 0。
最终嵌字均使用默认 10% 软边界，没有项目进入 20% 人工核对。句末连续省略号或终止符最多占一个
较小字号字位；该例外不改变同一完整句的统一基础字号。

完整测试：71/71 通过（含 PySide6 GUI 工作区、后台 dry-run 与核对文件原子保存测试）。相同输入整批复跑后，
四张最终 PNG 的 SHA-256 均保持不变。

## 第 1 页

- 核对图：[1.acceptance.png](../artifacts/batch/1.acceptance.png)
- 原图：[1.jpg](../samples/images/1.jpg)
- mask：[mask.png](../artifacts/batch/1.stages/mask.png)
- 擦字图：[erased.png](../artifacts/batch/1.stages/erased.png)
- 最终图：[1.translated.png](../artifacts/batch/1.translated.png)
- state：[1.state.json](../artifacts/batch/1.state.json)
- 越界审计：[typeset.audit.json](../artifacts/batch/1.stages/typeset.audit.json)

## 第 8 页

- 核对图：[8.acceptance.png](../artifacts/batch/8.acceptance.png)
- 原图：[8.jpg](../samples/images/8.jpg)
- mask：[mask.png](../artifacts/batch/8.stages/mask.png)
- 擦字图：[erased.png](../artifacts/batch/8.stages/erased.png)
- 最终图：[8.translated.png](../artifacts/batch/8.translated.png)
- state：[8.state.json](../artifacts/batch/8.state.json)
- 越界审计：[typeset.audit.json](../artifacts/batch/8.stages/typeset.audit.json)

## 第 19 页

- 核对图：[19.acceptance.png](../artifacts/batch/19.acceptance.png)
- 原图：[19.jpg](../samples/images/19.jpg)
- mask：[mask.png](../artifacts/batch/19.stages/mask.png)
- 擦字图：[erased.png](../artifacts/batch/19.stages/erased.png)
- 最终图：[19.translated.png](../artifacts/batch/19.translated.png)
- state：[19.state.json](../artifacts/batch/19.state.json)
- 越界审计：[typeset.audit.json](../artifacts/batch/19.stages/typeset.audit.json)

## 第 22 页

- 核对图：[22.acceptance.png](../artifacts/batch/22.acceptance.png)
- 原图：[22.jpg](../samples/images/22.jpg)
- mask：[mask.png](../artifacts/batch/22.stages/mask.png)
- 擦字图：[erased.png](../artifacts/batch/22.stages/erased.png)
- 最终图：[22.translated.png](../artifacts/batch/22.translated.png)
- state：[22.state.json](../artifacts/batch/22.state.json)
- 越界审计：[typeset.audit.json](../artifacts/batch/22.stages/typeset.audit.json)

## 人工观察

第 1、22 页部分画面文字所在区域仍能看到局部 inpainting 的模糊底纹。变化严格限制在对应 mask 内，
不会跨气泡、分镜或相邻台词；若要提升出版质量，下一轮应优化 C 阶段画面文字修复或提供人工背景修补，
而不是扩大 mask 或排版边界。
