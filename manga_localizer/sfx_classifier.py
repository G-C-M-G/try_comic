"""岛级"台词 vs 拟声"小分类器（**用我们自己的真值训出来的**，不是外来的预训练模型）。

来历：`var/train_three_books.py`（**现行训练脚本**：三本书 1536 岛、D:S=5:1；
第一版 `var/sfx_classifier.py` 只有 748 岛/两本书，留档）；这里是**推理侧**，供管线调用。
模型是**逻辑回归**（numpy 手写梯度下降，无 sklearn / 无 torch），11 个廉价特征
（第一版还有第 12 个"模型判 S 与否"，接线时发现部署时恒为 0、会掉准确率，删掉重训）。
类别权重 D:S = 5:1（保台词优先），训自三本书 1536 个岛 / 199 页的人工真值。
权重由训练脚本导出成 JSON（`sfx_classifier_weights.json`，特征清单与训练集以文件内 meta 为准）。

为什么需要它（实测，见 开发记录 6c-2 / 6c-3）：
* 几何规则（岛底不平坦 + 短字）在 748 个标注岛上：误杀台词 24 / 混入拟声 258（81%）
* 同一批上这个分类器（阈值 0.4）：误杀台词 **10** / 混入 278（87%）
  → 台词误杀减半多，混入率只涨 6pp；阈值即"保台词↔少混入"的旋钮
* 试过的七个判据都不行（几何/裁剪图视觉/整页+红框/词性/中文侧/deepseek 看日文/日文母语模型），
  原因：判别信号在图上、文本里没有；而这个分类器把图上的信号（占比、方向、框尺寸）
  与文本信号**一起**用，再加人工标签校准阈值，所以能超过任何单一路线。

**特征里不含"模型判 S 与否"**——那条只有调接口才有，放进去会让部署时的特征与训练时
不一致（实测过：差一个恒为 0 的特征就会让准确率掉下来）。所以这个分类器**完全不依赖接口**。

**权重缺失时调用方必须退回几何规则**（不能因为缺个文件就改变行为）。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

FEATURE_NAMES = (
    "bg_flat_ratio",      # 岛底平坦占比（图上信号）
    "chars",              # 非空白字数 / 10
    "vertical",           # 竖排
    "area_ratio",         # 岛面积 / 页面积 × 100
    "aspect",             # 短边/长边
    "has_kanji",          # 含汉字
    "kana_only",          # 纯假名（无汉字）
    "is_sfx_like",        # text_rules.is_sfx_like
    "content_words",      # fugashi 实词数 / 3
    "particles",          # fugashi 助词数 / 2
    "no_ratio",           # 没有 bg_flat_ratio
)
_KANA = re.compile(r"[\u3040-\u30ffー]")
_KANJI = re.compile(r"[\u4e00-\u9fff々]")
WEIGHTS_PATH = Path(__file__).with_name("sfx_classifier_weights.json")


def _tokens(text: str) -> tuple[int, int]:
    """(实词数, 助词数)。fugashi 不可用时给 0（退化但仍可用）。"""
    try:
        from fugashi import Tagger

        global _TAGGER
        try:
            _TAGGER
        except NameError:
            _TAGGER = Tagger()
        content = particles = 0
        for token in _TAGGER(text):
            pos = token.feature.pos1 if token.feature else ""
            if pos in ("名詞", "動詞", "形容詞", "副詞", "接続詞", "連体詞"):
                content += 1
            elif pos == "助詞":
                particles += 1
        return content, particles
    except Exception:  # noqa: BLE001 - 词典不可用不影响主判据
        return 0, 0


def extract_features(*, text: str, ratio, direction: str, bbox,
                     page_size) -> list[float]:
    """与训练脚本**逐字段一致**的特征（改动会破坏权重，改前请重训）。"""
    from .text_rules import is_sfx_like

    chars = len([c for c in (text or "").strip() if not c.isspace()])
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    page_area = max(1.0, float(page_size[0] * page_size[1]))
    content, particles = _tokens(text or "")
    return [
        (float(ratio) if ratio is not None else 0.5),
        min(chars, 40) / 10.0,
        1.0 if direction == "vertical" else 0.0,
        (width * height) / page_area * 100.0,
        min(width, height) / max(width, height),
        1.0 if _KANJI.search(text or "") else 0.0,
        1.0 if (_KANA.search(text or "") and not _KANJI.search(text or "")) else 0.0,
        1.0 if is_sfx_like(text or "") else 0.0,
        min(content, 6) / 3.0,
        min(particles, 4) / 2.0,
        int(ratio is None),
    ]


def load_weights(path: Path | None = None) -> dict | None:
    """读权重 JSON；没有就返回 None（调用方退回几何规则）。"""
    target = Path(path) if path else WEIGHTS_PATH
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not all(key in payload for key in ("mean", "std", "w", "b")):
        return None
    if len(payload["w"]) != len(FEATURE_NAMES):
        return None
    return payload


def probability(features: list[float], weights: dict) -> float:
    """P(这是台词/该翻译)。阈值越小越保守（丢得越少、台词越安全）。"""
    total = float(weights["b"])
    for value, mean, std, weight in zip(features, weights["mean"],
                                        weights["std"], weights["w"]):
        total += ((value - mean) / (std or 1e-6)) * weight
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))
