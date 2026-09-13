"""跨路径共用的阈值与补偿量。

Photos 蓝框路径与 Paddle 检测路径**必须共用同一套规则**，这些数字此前各写
一份（`one_click_translate.py` 与 `photos_capture.py` 各有一个 0.20），存在
"规则应当相同、实际各自漂移"的隐患。集中到这里，两边 import 同一份。

集中后两条路径的差异只剩"框从哪来、框多松"，规则本身不再分叉。
"""

from __future__ import annotations

# --- 共用规则 ---------------------------------------------------------------

# 整页大框过滤：框面积 > 该比例 × 页面积 → 丢弃（不是原子文字单位）
WHOLE_PAGE_AREA_RATIO = 0.20

# 大拟音/效果字过滤：框短边 > 该比值 × 同方向基准（横比横、竖比竖；
# 同方向样本 <3 时回落全页基准）且 > 页短边 × 该比例 → 丢弃
SFX_MIN_SIDE_RATIO = 2.0
SFX_MIN_SIDE_PAGE = 0.10

# --- 按框来源的补偿量 -------------------------------------------------------
# Photos 蓝框是"文本选区"，紧贴笔画；Paddle 检测框是"文本行 + 余量"，约松
# 2px/边。两者用同一套嵌字规则，靠这两个补偿量把观感拉平。

# 布局边界 = 擦字范围 = 成员框并集 ±该值（canonical box）
BBOX_PAD = {"paddle": 4, "photos": 6}
# 字号上限 = 组内最窄框 + 该值
GLYPH_MARGIN = {"paddle": 5, "photos": 12}

_DEFAULT_SOURCE = "photos"


def bbox_pad(box_source: str | None) -> int:
    """布局边界/擦字范围的扩边量。"""
    return BBOX_PAD.get(str(box_source or _DEFAULT_SOURCE), BBOX_PAD[_DEFAULT_SOURCE])


def glyph_margin(box_source: str | None) -> int:
    """字号上限相对框尺寸的余量。"""
    return GLYPH_MARGIN.get(
        str(box_source or _DEFAULT_SOURCE), GLYPH_MARGIN[_DEFAULT_SOURCE]
    )
