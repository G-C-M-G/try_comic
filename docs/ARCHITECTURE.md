# 架构（ARCHITECTURE）

版本：2026-08-06（两仓合并后）

## 1. 总数据流

```
[上游] Windows Photos 扫描文本（photos_capture.py, batch --fast）
  -> photos_base.png + photos_ocr.png（高亮蒙版帧）
  -> 差分判定（变暗≥15%）-> 岛=未压暗且亮度>145
  -> island_extract.py / make_island_dataset.py
  -> outputs/exam/dataset/{page}/   （islands.json + filled_black/blue + original.jpg）  [只读输入]
        |
        v
[应用] 岛流水线（manga_localizer/ + scripts/）
  A. 岛契约解析与校验（island_contract.py）
  B. 岛内擦字 + 岛外零改动审计（island_erasure.py, min_island_area=800 微岛舍弃）
  C. 双引擎交叉验证（cross_validate.py）：
       文本权威 = manga-ocr（整岛一次识别，scripts/manga_ocr_islands.py）
       几何权威 = Paddle 岛级裁剪（paddle_ocr_worker.py --crops 四变体择优）
       片段包含判定 + 编辑距离 >0.3 分歧 + 行置信 <0.6 低置信 + manga 空岛兜底
  C2. 人工核对（island_gui.py 懒人开关默认承认 manga）+ 原图重读修复
  D. 整岛一段翻译（island_translate.py）：
       --precision local（Sakura）| online（整页打包给在线 AI）
       校验：提示词泄漏（含上下文片段）/kana/长度；contextual→concise 重试；
       kept-source 决策缓存（符号/拟声岛保留原文）
  E. 纯竖排双模式排版（island_typeset.py）：
       模式1 列位跟随 Paddle 列（右→左）+ 模式2 外接矩形加大字号
       min_ratio=0.6 可配；白描边 max(1,字号//10)；中心锁定 5% 圆形容差；
       岛外像素变化=0（岛并集掩膜裁剪兜底）
  F. state 与重嵌（island_state.py）：四路 SHA 绑定，只改译文重嵌（<0.1s）
```

## 2. 引擎职责（经验固化）

| 引擎 | 职责 | 原因 |
|---|---|---|
| manga-ocr | 文本权威（整岛读取） | 漫画专用训练，整句质量最高 |
| PaddleOCR（岛级 crops） | 几何权威 + 佐证/兜底 | 行框+置信度；整页识别碎片多，岛级裁剪后置信与文本明显改善 |
| Windows.Media.Ocr(ja) | 已弃用（代码保留） | 艺术字误读率太高（`ョドーシ〕が` 等） |
| 在线 AI（OpenAI 兼容） | --precision online 翻译 | 口语化/上下文，整页打包编号一一对应 |

## 3. 关键机制

- **擦字 mask 一律裁剪到岛内**（字形检测的膨胀不越岛）——岛外零改动是结构性的，不只靠审计。
- **原图重读优于填黑图**：filled_black 黑背景导致 manga-ocr 误读（`間属性`），
  原图裁剪读出正确 `闇属性`；bbox 扩区改变缩放比例反而降质（已取消扩区，黄色描边仅作可视化）。
- **交叉验证一致性** = 归一化后 Paddle 片段包含于 manga 文本（顺序以 manga 为准）；
  片段全为短字符时回退编辑距离。
- **翻译校验三层**：提示词字段泄漏、上下文文本片段泄漏、kana/长度异常；全部通过才写缓存。
- **kept-source 约定**：两轮翻译（contextual/concise）都失败 → 判定符号/拟声岛，译文=原文，
  决策缓存 `*.kept.json` 免重复请求；M4 仍嵌字回原符号（黑字白描边）。
- **缓存签名**：原文 SHA + 模型 + 提示版本 + precision + 上下文；改原文即失效，不串页。
- **排版**：竖排右→左（模式1 列位跟随原 Paddle 列 x 中心，模式2 居中块加大字号到
  A_rect 的 60%-110%）；字距=字号//16、列距=字号//4；中心偏移 ≤ 岛对角线×5% 圆形。
- **state 重嵌**：绑定 original/islands/erased/translations 四路 SHA，任一不符拒绝，
  防旧译文套新图。

## 4. 状态语义

- 岛核对：`pass / review / dropped_tiny`（面积 <800px²）/ `skipped`
- 翻译：`translated / kept-source`（决策已缓存）/ `failed`
- 排版：`typeset / typeset_pending_review`（面积比越界或中心偏移超差）/ `translation_failed`
- 人工核对：懒人开关默认开 = REVIEW 直接承认 manga；关闭后逐项 批准/改字/拒绝/标SFX
