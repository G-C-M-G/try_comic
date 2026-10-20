"""Island-restricted Chinese typesetting (M4, E stage).

Vertical (default): columns right to left, characters top to bottom inside
each column; all columns of one sentence share the same TOP edge.
Horizontal (e.g. afterword pages whose atomic blue boxes are wider than
tall): rows top to bottom, characters left to right inside each row, all
rows LEFT-ALIGNED (a short last row does not shift).  The direction of
every island is decided by the caller through ``direction_overrides``
(derived from the Photos blue selection boxes: box width >= height =>
horizontal).

Single layout path: the translated block is a centred block laid out
inside the ISLAND bbox -- the canonical "one box" (the blue-box group
expanded 6 px).  The island bbox is the layout bounds, the containment
boundary, the centre, and the single area-ratio base:
r = A_text / A_island with target [0.6, 1.1]; an off-target ratio is a
quality signal only (pending_review).  The block must fit ENTIRELY inside
the island bbox; the font search descends below ``min_font_size`` (28px)
down to the absolute floor ``MIN_FONT_SIZE_MIN`` (12px) so the text is
always fully visible inside the box.  Only when nothing fits even at the
absolute floor may the block exceed the island bbox up to the 150%
envelope (soft cap, pending_review); beyond that the text is rendered at
the floor and clipped -- an erased island is never left blank.
Glyph size is additionally capped by the PADDED box dimensions: vertical
size <= narrowest 6px-padded box width, horizontal size <= shortest padded
box height.  The column gap is never compressed to zero.
``union_overrides`` carries per-island glyph caps
(``glyph_width_limit`` / ``glyph_height_limit``).

Hard boundary: island image_bbox (no silent cropping beyond it), white
stroke on every glyph (stroke_width = max(1, size // 10), colour
configurable), character tracking and column/row gap are explicit layout
parameters, and the pixel audit guarantees zero changes outside the islands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Collection, Mapping, NamedTuple, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .thresholds import SFX_MAX_FONT_SIZE, STROKE_OVERFLOW_MARGIN
from .text_rules import normalize_ellipsis
from .errors import ErasureError, TypesettingError
from .island_contract import IslandDataset, IslandInfo
from .island_dry_run import OcrLine
from .ocr_contract import BBox
from .sentence_splitter import ColumnGroup, _column_key

MIN_RATIO = 0.6
MAX_RATIO = 1.1
HARD_RATIO_CAP = 1.2
CENTER_TOLERANCE_FRACTION = 0.05
STROKE_DIVISOR = 10
TRACKING_DIVISOR = 16
COLUMN_GAP_DIVISOR = 4
# Preferred readability floor: the font search prefers sizes at or above
# this, but keeps descending below it (down to ``MIN_FONT_SIZE_MIN``) so
# the text always fits inside the island box instead of being truncated.
MIN_FONT_SIZE = 28
# Absolute floor: below this the text is never legible anyway; if even
# this does not fit, the block may exceed the island up to the 150%
# envelope (soft cap) or is rendered and clipped -- never left blank.
MIN_FONT_SIZE_MIN = 12
# Last-resort layout cap: only when nothing fits even at the absolute
# floor may the block exceed the island bbox by up to this fraction of
# the island's size per side (a 150% envelope).  The column gap is never
# compressed to zero to force a fit.
MAX_GROUP_BBOX_OVERFLOW_RATIO = 0.5


@dataclass(frozen=True)
class TypesetColumnSlot:
    index: int
    x_center: float
    y_center: float
    char_count: int


@dataclass(frozen=True)
class IslandTypesetAudit:
    island_id: str
    text: str
    font_size: int
    stroke_width: int
    tracking: int
    column_gap: int
    area_base: Optional[float]
    area_text: Optional[float]
    area_ratio: Optional[float]
    center_offset: float
    center_tolerance: float
    block_bbox: Optional[BBox]
    slots: Tuple[TypesetColumnSlot, ...]
    status: str  # typeset | typeset_pending_review | translation_failed | skipped
    reasons: Tuple[str, ...]
    changed_inside: int
    changed_outside: int


@dataclass(frozen=True)
class IslandTypesetResult:
    image: Image.Image
    difference: Image.Image
    changed_mask: Image.Image
    audits: Tuple[IslandTypesetAudit, ...]
    typeset_seconds: float


@dataclass(frozen=True)
class TypesetConfig:
    font_name: str = "msyh.ttc"
    stroke_color: Tuple[int, int, int] = (255, 255, 255)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    background_color: Optional[Tuple[int, int, int]] = None
    min_ratio: float = MIN_RATIO
    max_ratio: float = MAX_RATIO
    hard_ratio_cap: float = HARD_RATIO_CAP
    center_tolerance_fraction: float = CENTER_TOLERANCE_FRACTION
    stroke_divisor: int = STROKE_DIVISOR
    tracking_divisor: int = TRACKING_DIVISOR
    column_gap_divisor: int = COLUMN_GAP_DIVISOR
    row_gap_divisor: int = COLUMN_GAP_DIVISOR
    min_font_size: int = MIN_FONT_SIZE
    # 字号上限（见 thresholds.SFX_MAX_FONT_SIZE 的说明）。作用于调用方判定为
    # "短原文/拟声词"的岛（``typeset_island_page(capped_islands=...)``），
    # 只封顶不放大。
    sfx_max_font_size: int = SFX_MAX_FONT_SIZE
    padding: int = 0

    def tracking(self, size: int) -> int:
        return max(0, size // self.tracking_divisor)

    def column_gap(self, size: int) -> int:
        return max(1, size // self.column_gap_divisor)

    def row_gap(self, size: int) -> int:
        return max(1, size // self.row_gap_divisor)

    def stroke(self, size: int) -> int:
        return max(1, size // self.stroke_divisor)


def _char_step(config: TypesetConfig, size: int) -> int:
    return size + config.tracking(size)


def _pitch(config: TypesetConfig, size: int, *, horizontal: bool = False) -> int:
    # Vertical layout: pitch = glyph + column gap.  Horizontal layout:
    # pitch = glyph + row gap (independent control via ``row_gap_divisor``).
    if horizontal:
        return size + config.row_gap(size)
    return size + config.column_gap(size)


# --- 禁则（避头尾）---------------------------------------------------------
#
# 中文/日文排版的硬规矩：某些标点**不能出现在行首**（避头），某些**不能出现在
# 行末**（避尾）。之前这里完全没有这层——标点占一个格子和汉字一样排，于是成品
# 里会出现"一列以「，」开头"这种肉眼可见的事故。
#
# 实现上有一条必须守住的约束：**只移动断点，不调换字符顺序**（禁则是"在哪里断
# 行"的问题，不是字序问题），而且调整后每行字数仍落在 [1, capacity] 内。这样块
# 的宽高只会变小不会变大，"字始终完整落在岛内"这个前提不会被破坏。

PROHIBITED_LINE_START = frozenset(
    "、。，．！？：；）］｝〉》」』】〕〗〙〛”’…ー,.!?:;)]}～"
)
PROHIBITED_LINE_END = frozenset("（［｛〈《「『【〔〖〘〚“‘([{")


def _line_breaks(counts: Sequence[int]) -> list:
    """每行起始位置（累计字数），最后一行之后不返回（它不是断点）。"""
    positions, total = [], 0
    for count in counts[:-1]:
        total += count
        positions.append(total)
    return positions


def kinsoku_violations(text: str, counts: Sequence[int]) -> int:
    """这个折行方案有几处避头尾违规（0 = 干净）。

    第一行开头**不算违规**——那是原文自己的开头（例如整句以「…」起），不是断行
    造成的。这与参考实现（`ShapeAwareTextLayout.respectsKinsoku` 跳过 index 0）
    一致。
    """
    if not text or not counts:
        return 0
    bad = 1 if text[-1] in PROHIBITED_LINE_END else 0
    for position in _line_breaks(counts):
        if not 0 < position < len(text):
            continue
        if text[position] in PROHIBITED_LINE_START:
            bad += 1
        if text[position - 1] in PROHIBITED_LINE_END:
            bad += 1
    return bad


def _shift_break_earlier(counts: list, index: int, capacity: int) -> bool:
    """断点往前挪一格（**追い出し**）：第 index 行首的字挪到第 index-1 行末。

    第 index-1 行因此满出来，就把它的末字继续挤到下一行——沿后续行级联，直到
    撞上某一行还有空位（通常是最后一行，它是余数）。级联让中间各行的字数不变，
    所以只需改两处。
    """
    if counts[index - 1] <= 1:
        return False
    room = next((k for k in range(index, len(counts)) if counts[k] < capacity), None)
    if room is None:
        return False
    counts[index - 1] -= 1
    counts[room] += 1
    return True


def _shift_break_later(counts: list, index: int, capacity: int) -> bool:
    """断点往后挪一格（**追い込み**）：第 index-1 行末的字提上来当第 index 行首。

    与上一个方向相反，空位得从**前面**的行里找。
    """
    if counts[index] <= 1:
        return False
    room = next((k for k in range(index - 1, -1, -1) if counts[k] < capacity), None)
    if room is None:
        return False
    counts[index] -= 1
    counts[room] += 1
    return True


def apply_kinsoku(text: str, counts: Sequence[int],
                  capacity: int, passes: int = 4) -> Tuple[int, ...]:
    """按避头尾调整断点，返回调整后的每行字数。

    不变量：**行数不变、逐行字数仍在 [1, capacity]、总和不变**。所以调用方拿到
    的块宽高只会不变或变小，放得下这个结论不会被推翻。

    两个方向都试，**只在真的减少了违规数时才采纳**（否则撤回），所以它可能修
    不干净（例如相邻两个标点都该避开、无解），但绝不会把原来正确的改坏。这与
    参考实现把 `respectsKinsoku` 当**硬否决**（宁可缩字号）的选择不同——这里
    宁可留一处违规也不缩字号，因为我们的字号是"填满框"驱动的，缩字号会让整块
    观感变化，代价比一个标点位置大。
    """
    counts = list(counts)
    if len(counts) < 2 or not text:
        return tuple(counts)
    for _ in range(passes):
        if kinsoku_violations(text, counts) == 0:
            break
        improved = False
        for index in range(1, len(counts)):
            trial = list(counts)
            if (_shift_break_earlier(trial, index, capacity)
                    or _shift_break_later(trial, index, capacity)):
                if (kinsoku_violations(text, trial)
                        < kinsoku_violations(text, counts)):
                    counts = trial
                    improved = True
        if not improved:
            break
    return tuple(counts)


class LinePlan(NamedTuple):
    """一次折行计划：每列/每行的字数（已按禁则调过断点）+ 块宽高。

    ``counts`` 取代了原来"每行字数相同（``chars_per_column``）+ 末行余数"的
    表示——禁则调整会让各行字数不再相同，所以必须存整个序列。
    """

    counts: Tuple[int, ...]
    block_width: int
    block_height: int

    @property
    def line_count(self) -> int:
        return len(self.counts)


def _column_layout(
    text: str,
    size: int,
    island: IslandInfo,
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
) -> LinePlan:
    """Vertical layout plan.  ``layout_bbox`` overrides the island bbox as the
    layout bounds -- used by union mode so the block can never leave the
    blue-box union.
    """
    x1, y1, x2, y2 = layout_bbox if layout_bbox is not None else island.image_bbox
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    step = _char_step(config, size)
    pitch = _pitch(config, size)
    chars_per_column = max(1, usable_h // step)
    column_count = max(1, math.ceil(len(text) / chars_per_column))
    # columns never exceed the island width; the font search clamps size
    while column_count * pitch - config.column_gap(size) > usable_w and column_count > 1:
        column_count -= 1
    # After shrinking the column count, the remaining columns must still
    # hold every character: recompute chars_per_column so the last column
    # never exceeds the island height (an over-capacity last column would
    # render outside the island).
    chars_per_column = max(1, math.ceil(len(text) / column_count))
    last_count = len(text) - (column_count - 1) * chars_per_column
    if last_count <= 0:
        column_count = math.ceil(len(text) / chars_per_column)
        last_count = len(text) - (column_count - 1) * chars_per_column
    counts = apply_kinsoku(
        text, [chars_per_column] * (column_count - 1) + [last_count],
        chars_per_column,
    )
    # 禁则只重分配断点，所以 max(counts) <= chars_per_column → 块只会变矮
    block_height = (max(counts) - 1) * step + size
    block_width = len(counts) * pitch - config.column_gap(size)
    return LinePlan(counts, block_width, block_height)


def _counts_follow_columns(text: str, original_columns: Sequence[ColumnGroup]) -> list:
    """Spread mode 的折行：一个块列对一个原列，字数均分（禁则只调断点）。"""
    column_count = min(len(text), len(original_columns))
    chars_per_column = max(1, math.ceil(len(text) / column_count))
    last_count = len(text) - (column_count - 1) * chars_per_column
    if last_count <= 0:
        # The columns would overflow the text (e.g. 5 chars across 4
        # boxes): shrink the column count so the last column is never
        # empty or negative.
        column_count = math.ceil(len(text) / chars_per_column)
        last_count = len(text) - (column_count - 1) * chars_per_column
    return apply_kinsoku(
        text, [chars_per_column] * (column_count - 1) + [last_count],
        chars_per_column,
    )


def _slots_follow_columns(
    text: str,
    size: int,
    original_columns: Sequence[ColumnGroup],
    island: IslandInfo,
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
    block_centre_y: Optional[float] = None,
    spread: bool = False,
    top_align: Optional[float] = None,
) -> Sequence[TypesetColumnSlot]:
    if spread and original_columns:
        # Spread the text across the original columns: one block column per
        # box, so every column sits on its own box (union mode for spread
        # text; the box count drives the column count, not the union
        # height).
        counts = _counts_follow_columns(text, original_columns)
    else:
        counts = _column_layout(
            text, size, island, config, layout_bbox=layout_bbox
        ).counts
    column_count = len(counts)
    # reading order: rightmost original column first
    ordered = sorted(original_columns, key=_column_key)
    pitch = _pitch(config, size)
    step = _char_step(config, size)
    bounds = layout_bbox if layout_bbox is not None else island.image_bbox
    slots = []
    for i in range(column_count):
        if i < len(ordered):
            cx = (ordered[i].bbox[0] + ordered[i].bbox[2]) / 2.0
            cy = (ordered[i].bbox[1] + ordered[i].bbox[3]) / 2.0
        else:
            base_cx = (ordered[-1].bbox[0] + ordered[-1].bbox[2]) / 2.0
            cx = base_cx - (i - len(ordered) + 1) * pitch
            cy = (bounds[1] + bounds[3]) / 2.0
        count = counts[i]
        if top_align is not None:
            # All columns of one sentence share the same top edge.
            cy = top_align + ((count - 1) * step + size) / 2.0
        elif block_centre_y is not None:
            # Union mode: the whole block is vertically centred on the group
            # centre instead of following each box's own y-range, so the
            # block never rides above or below the boxes.
            cy = block_centre_y
        slots.append(TypesetColumnSlot(index=i, x_center=cx, y_center=cy, char_count=count))
    return slots


def _slots_centered(
    text: str,
    size: int,
    island: IslandInfo,
    centre: Tuple[float, float],
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
    top_align: Optional[float] = None,
) -> Sequence[TypesetColumnSlot]:
    plan = _column_layout(text, size, island, config, layout_bbox=layout_bbox)
    pitch = _pitch(config, size)
    step = _char_step(config, size)
    # reading order: column 0 sits rightmost, later columns go leftwards
    right = centre[0] + plan.block_width / 2.0
    slots = []
    for i, count in enumerate(plan.counts):
        cx = right - size / 2.0 - i * pitch
        if top_align is not None:
            # All columns of one sentence share the same top edge.
            cy = top_align + ((count - 1) * step + size) / 2.0
        else:
            cy = centre[1]
        slots.append(TypesetColumnSlot(index=i, x_center=cx, y_center=cy, char_count=count))
    return slots


def _row_layout(
    text: str,
    size: int,
    island: IslandInfo,
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
) -> LinePlan:
    """Horizontal layout plan (每行字数已按禁则调过断点)."""
    x1, y1, x2, y2 = layout_bbox if layout_bbox is not None else island.image_bbox
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    step = _char_step(config, size)
    row_pitch = _pitch(config, size, horizontal=True)
    chars_per_row = max(1, usable_w // step)
    row_count = max(1, math.ceil(len(text) / chars_per_row))
    while row_count * row_pitch - config.row_gap(size) > usable_h and row_count > 1:
        row_count -= 1
    # After shrinking the row count, the remaining rows must still hold
    # every character: recompute chars_per_row so the last row never
    # exceeds the island width (an over-capacity last row would render
    # outside the island).
    chars_per_row = max(1, math.ceil(len(text) / row_count))
    last_count = len(text) - (row_count - 1) * chars_per_row
    counts = apply_kinsoku(
        text, [chars_per_row] * (row_count - 1) + [last_count], chars_per_row)
    # 禁则只会让某行变短 → 块宽 <= 原宽，不会溢出
    block_width = (max(counts) - 1) * step + size
    block_height = len(counts) * row_pitch - config.column_gap(size)
    return LinePlan(counts, block_width, block_height)


def _slots_horizontal(
    text: str,
    size: int,
    island: IslandInfo,
    centre: Tuple[float, float],
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
) -> Sequence[TypesetColumnSlot]:
    """Horizontal rows: characters left to right, rows top to bottom.

    Rows are LEFT-ALIGNED: every row starts at the block's left edge (the
    block itself stays horizontally centred on ``centre``), so a short last
    row does not shift the text.
    """
    plan = _row_layout(text, size, island, config, layout_bbox=layout_bbox)
    row_pitch = _pitch(config, size, horizontal=True)
    step = _char_step(config, size)
    top = centre[1] - plan.block_height / 2.0
    left = centre[0] - plan.block_width / 2.0
    slots = []
    for i, count in enumerate(plan.counts):
        y_center = top + size / 2.0 + i * row_pitch
        x_center = left + (count - 1) * step / 2.0
        slots.append(TypesetColumnSlot(index=i, x_center=x_center, y_center=y_center, char_count=count))
    return slots


def _block_metrics_horizontal(
    slots: Sequence[TypesetColumnSlot],
    size: int,
    config: TypesetConfig,
) -> Tuple[BBox, Tuple[float, float]]:
    """Metrics for horizontal row slots (row width = count * step).

    The drawn extent of a row is ``[x_center - (count-1)*step/2,
    x_center - (count-1)*step/2 + (count-1)*step + size]`` -- the same
    formula ``_draw_slots`` uses -- so the block bbox always equals what
    is actually rendered (the naive ``x_center +/- width/2`` would
    understate the right edge by ``size/2``).
    """
    step = _char_step(config, size)
    row_pitch = _pitch(config, size)
    xs = [
        slot.x_center - (slot.char_count - 1) * step / 2.0
        for slot in slots
    ] + [
        slot.x_center - (slot.char_count - 1) * step / 2.0
        + (slot.char_count - 1) * step + size
        for slot in slots
    ]
    ys = [
        s.y_center - size / 2.0 for s in slots
    ] + [
        s.y_center + (row_pitch - config.column_gap(size)) / 2.0 for s in slots
    ]
    bbox = (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )
    centre = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    return bbox, centre


def _block_metrics(
    slots: Sequence[TypesetColumnSlot],
    size: int,
    config: TypesetConfig,
) -> Tuple[BBox, Tuple[float, float]]:
    step = _char_step(config, size)
    xs = []
    ys = []
    for slot in slots:
        xs.extend((slot.x_center - size / 2.0, slot.x_center + size / 2.0))
        top = slot.y_center - ((slot.char_count - 1) * step + size) / 2.0
        bottom = slot.y_center + ((slot.char_count - 1) * step + size) / 2.0
        ys.extend((top, bottom))
    bbox = (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )
    centre = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    return bbox, centre



# 符号回退字体：主字体（雅黑等）缺 ♡ ♥ ♪ ♫ ❤ ∀ 这类字形，会画成豆腐块 □。
# 逐字挑选第一个"有该字形"的字体——中文仍用主字体，符号交给符号字体。
# 判定用"与私用区字符的渲染结果比对"（豆腐块的位图是固定的），结果按
# (字体, 字符) 缓存，只在首次遇到某符号时算一次。
FONT_FALLBACKS = (
    r"C:\Windows\Fonts\seguisym.ttf",   # Segoe UI Symbol：♡♥♪♫❤∀ 全覆盖
    r"C:\Windows\Fonts\seguiemj.ttf",   # Segoe UI Emoji：兜底
)
_TOFU_REF = ""                        # 私用区：字体必然没有的字形


class _FontChain:
    """按字符挑字体：主字体 → 回退字体（缺字形则跳过）。"""

    def __init__(self, primary: str, size: int,
                 fallbacks: Sequence[str] = FONT_FALLBACKS) -> None:
        self._primary = ImageFont.truetype(primary, size=size)
        self._size = size
        self._fallbacks: list = []
        for path in fallbacks:
            try:
                self._fallbacks.append(ImageFont.truetype(path, size=size))
            except OSError:
                continue        # 该回退字体不存在就跳过
        self._cache: dict[str, object] = {}
        self._tofu: dict[int, bytes] = {}

    def _reference(self, font) -> bytes:
        key = id(font)
        if key not in self._tofu:
            image = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(image).text((4, 4), _TOFU_REF, font=font, fill=255)
            self._tofu[key] = image.tobytes()
        return self._tofu[key]

    def _has_glyph(self, font, char: str) -> bool:
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).text((4, 4), char, font=font, fill=255)
        return image.tobytes() != self._reference(font)

    def font_for(self, char: str):
        """该字符该用哪个字体：主字体有就用它，否则第一个有字形的回退字体。"""
        cached = self._cache.get(char)
        if cached is not None:
            return cached
        chosen = self._primary
        if not self._has_glyph(self._primary, char):
            for fallback in self._fallbacks:
                if self._has_glyph(fallback, char):
                    chosen = fallback
                    break
        self._cache[char] = chosen
        return chosen


def _draw_slots(
    canvas: Image.Image,
    text: str,
    slots: Sequence[TypesetColumnSlot],
    size: int,
    config: TypesetConfig,
    direction: str = "vertical",
) -> ImageFont.ImageFont:
    # 主字体 + 符号回退（否则 ♡♥♪♫❤ 会被画成豆腐块 □）
    fonts = _FontChain(config.font_name, size)
    draw = ImageDraw.Draw(canvas)
    stroke = config.stroke(size)
    step = _char_step(config, size)
    index = 0

    def _background_box(x: float, y: float) -> None:
        if config.background_color is None:
            return
        pad = stroke + 2
        draw.rectangle(
            [x - pad, y - pad, x + size + pad, y + size + pad],
            fill=config.background_color,
        )

    if direction == "horizontal":
        # row slots: characters left to right inside each row
        for slot in slots:
            left = slot.x_center - ((slot.char_count - 1) * step) / 2.0
            for j in range(slot.char_count):
                if index >= len(text):
                    break
                char = text[index]
                index += 1
                x = left + j * step
                y = slot.y_center - size / 2.0
                _background_box(x, y)
                draw.text(
                    (x, y),
                    char,
                    font=fonts.font_for(char),
                    fill=config.text_color,
                    stroke_width=stroke,
                    stroke_fill=config.stroke_color,
                )
        return fonts._primary
    for slot in slots:
        top = slot.y_center - ((slot.char_count - 1) * step + size) / 2.0
        for j in range(slot.char_count):
            if index >= len(text):
                break
            char = text[index]
            index += 1
            x = slot.x_center - size / 2.0
            y = top + j * step
            _background_box(x, y)
            draw.text(
                (x, y),
                char,
                font=fonts.font_for(char),
                fill=config.text_color,
                stroke_width=stroke,
                stroke_fill=config.stroke_color,
            )
    return fonts._primary


def _fit_size(
    text: str,
    island: IslandInfo,
    area_rect: float,
    centre: Tuple[float, float],
    config: TypesetConfig,
    direction: str = "vertical",
    glyph_width_limit: Optional[float] = None,
    glyph_height_limit: Optional[float] = None,
    sfx: bool = False,
    ceiling_px: Optional[int] = None,
) -> Tuple[int, Sequence[TypesetColumnSlot], Optional[float], Tuple[str, ...]]:
    """Single-path fit: a centred block laid out inside the ISLAND bbox.

    The island bbox is the canonical "one box" (the blue-box group
    expanded 6 px): it is the layout bounds, the containment boundary and
    the area-ratio base A_text / A_island with target
    [min_ratio, max_ratio]:
    * the block must fit ENTIRELY inside the island bbox -- the font size
      descends below ``config.min_font_size`` (down to the absolute floor
      ``MIN_FONT_SIZE_MIN``) if needed, so the text is always fully
      visible inside the box (never truncated);
    * only when nothing fits even at the absolute floor may the block
      exceed the island bbox, up to the 150% envelope (soft cap, flagged
      pending_review); beyond that the text is rendered at the floor and
      clipped -- an erased island is never left blank;
    * off-target area ratios are quality signals only (pending_review);
    * the column gap is never compressed to zero.
    """
    x1, y1, x2, y2 = island.image_bbox
    bounds = island.image_bbox
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    max_size = max(MIN_FONT_SIZE_MIN, int(max(1, min(usable_w, usable_h))))
    if direction == "vertical" and glyph_width_limit is not None and glyph_width_limit > 0:
        max_size = max(MIN_FONT_SIZE_MIN, min(max_size, int(glyph_width_limit)))
    if direction == "horizontal" and glyph_height_limit is not None and glyph_height_limit > 0:
        max_size = max(MIN_FONT_SIZE_MIN, min(max_size, int(glyph_height_limit)))
    if sfx:
        # 短原文/拟声词只封顶不放大：框小的时候仍按框自适应
        max_size = max(
            MIN_FONT_SIZE_MIN,
            min(max_size, int(config.sfx_max_font_size)),
        )
    if ceiling_px and ceiling_px > 0:
        # 整页字号上限（相对该页常规字号）：同样只封顶不放大
        max_size = max(MIN_FONT_SIZE_MIN, min(max_size, int(ceiling_px)))

    gw = max(1, x2 - x1)
    gh = max(1, y2 - y1)
    preferred_floor = max(MIN_FONT_SIZE_MIN, config.min_font_size)

    def _slots_for(size_, centre_):
        if direction == "horizontal":
            return _slots_horizontal(text, size_, island, centre_, config,
                                     layout_bbox=bounds)
        return _slots_centered(text, size_, island, centre_, config,
                               layout_bbox=bounds, top_align=bounds[1] + 2)

    def _metrics_for(slots_, size_):
        if direction == "horizontal":
            return _block_metrics_horizontal(slots_, size_, config)
        return _block_metrics(slots_, size_, config)

    def _block_size(slots_, size_):
        bbox, _ = _metrics_for(slots_, size_)
        return bbox, bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _fits_island(bbox) -> bool:
        return (
            bbox[0] >= x1 and bbox[1] >= y1
            and bbox[2] <= x2 and bbox[3] <= y2
        )

    def _ratio(bbox) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / area_rect

    best: Optional[Tuple[int, Sequence[TypesetColumnSlot], float]] = None

    def _consider(size_: int, ratio_: float) -> None:
        nonlocal best
        if best is None or ratio_ > best[2]:
            best = (size_, _slots_for(size_, centre), ratio_)

    # Pass 1: preferred range down to the preferred floor (28 px default).
    for size in range(max_size, preferred_floor - 1, -1):
        slots = _slots_for(size, centre)
        bbox, block_w, block_h = _block_size(slots, size)
        if block_w > gw or block_h > gh or not _fits_island(bbox):
            continue
        ratio = _ratio(bbox)
        if config.min_ratio <= ratio <= config.max_ratio:
            return size, slots, ratio, ()
        _consider(size, ratio)
    # Pass 2: below the preferred floor the font keeps shrinking (down to
    # the absolute floor) so the text always fits the island box.
    for size in range(min(preferred_floor - 1, max_size), MIN_FONT_SIZE_MIN - 1, -1):
        slots = _slots_for(size, centre)
        bbox, block_w, block_h = _block_size(slots, size)
        if block_w > gw or block_h > gh or not _fits_island(bbox):
            continue
        ratio = _ratio(bbox)
        if config.min_ratio <= ratio <= config.max_ratio:
            return size, slots, ratio, ()
        _consider(size, ratio)
    if best is not None:
        size, slots, ratio = best
        return size, slots, ratio, (
            f"area ratio {ratio:.2f} outside [{config.min_ratio}, {config.max_ratio}]",
        )
    # Nothing fits even at the absolute floor: the block may exceed the
    # island bbox up to the 150% envelope (soft cap); beyond that it is
    # rendered at the floor and clipped.  An erased island is never blank.
    size = MIN_FONT_SIZE_MIN
    slots = _slots_for(size, centre)
    bbox, block_w, block_h = _block_size(slots, size)
    allow_x1 = x1 - MAX_GROUP_BBOX_OVERFLOW_RATIO * gw
    allow_x2 = x2 + MAX_GROUP_BBOX_OVERFLOW_RATIO * gw
    allow_y1 = y1 - MAX_GROUP_BBOX_OVERFLOW_RATIO * gh
    allow_y2 = y2 + MAX_GROUP_BBOX_OVERFLOW_RATIO * gh
    if (
        block_w <= 1.5 * gw and block_h <= 1.5 * gh
        and bbox[0] >= allow_x1 and bbox[1] >= allow_y1
        and bbox[2] <= allow_x2 and bbox[3] <= allow_y2
    ):
        reasons = (
            f"does not fit the island bbox at the absolute font floor "
            f"({MIN_FONT_SIZE_MIN}px)",
            "block exceeds the island bbox (soft cap within the 150% envelope)",
        )
    else:
        reasons = (
            f"does not fit the island bbox at the absolute font floor "
            f"({MIN_FONT_SIZE_MIN}px); rendered and clipped",
        )
    return size, slots, 0.0, reasons


def _compute_island_plan(
    island: IslandInfo,
    text: str,
    direction: str,
    config: TypesetConfig,
    union_overrides: Optional[Mapping[str, dict]] = None,
    sfx: bool = False,
    ceiling_px: Optional[int] = None,
) -> dict:
    """Layout-only geometry for one island (no drawing, no pixel access).

    The island bbox is the canonical "one box" (the blue-box group
    expanded 6 px): the layout bounds, the containment boundary, the
    centre, and the area-ratio base A_text / A_island.  The reported
    ``block_bbox`` is the actual laid-out text block (inside the island
    box except for the absolute-floor fallback).
    """
    x1, y1, x2, y2 = island.image_bbox
    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    area_rect = float((x2 - x1) * (y2 - y1))
    try:
        union = (union_overrides or {}).get(island.id)
        size, slots, ratio, reasons = _fit_size(
            text, island, area_rect, centre, config,
            direction=direction,
            glyph_width_limit=union.get("glyph_width_limit") if union else None,
            glyph_height_limit=union.get("glyph_height_limit") if union else None,
            sfx=sfx,
            ceiling_px=ceiling_px,
        )
    except TypesettingError as error:
        return {
            "island_id": island.id, "text": text, "direction": direction,
            "font_size": 0, "stroke_width": 0, "tracking": 0, "column_gap": 0,
            "area_base": area_rect, "area_text": None, "area_ratio": None,
            "center_offset": 0.0, "center_tolerance": 0.0, "block_bbox": None,
            "slots": (), "status": "translation_failed", "reasons": (str(error),),
            "margin": 2,
        }
    if direction == "horizontal":
        bbox, block_centre = _block_metrics_horizontal(slots, size, config)
    else:
        bbox, block_centre = _block_metrics(slots, size, config)
    margin = max(2, config.stroke(size) + 2)
    left_room = bbox[0] - (x1 + margin)
    right_room = (x2 - margin) - bbox[2]
    top_room = bbox[1] - (y1 + margin)
    bottom_room = (y2 - margin) - bbox[3]

    def _clamped_shift(room_low, room_high, low, high, delta):
        # Prefer the margin zone [room_low, room_high]; when the block is
        # too big for it (inverted interval), fall back to keeping the
        # block inside the island bbox -- a fitted block is never shifted
        # out of the canonical box.
        if room_high < room_low:
            room_low, room_high = low, high
        return max(room_low, min(room_high, delta))

    shift_x = _clamped_shift(
        -left_room, right_room, x1 - bbox[0], x2 - bbox[2],
        centre[0] - block_centre[0],
    )
    shift_y = _clamped_shift(
        -top_room, bottom_room, y1 - bbox[1], y2 - bbox[3],
        centre[1] - block_centre[1],
    )
    adjusted = [
        TypesetColumnSlot(
            index=slot.index,
            x_center=slot.x_center + shift_x,
            y_center=slot.y_center + shift_y,
            char_count=slot.char_count,
        )
        for slot in slots
    ]
    # Audit the geometry that is actually rendered (after the centre
    # shift), not the pre-shift layout.
    if direction == "horizontal":
        bbox, block_centre = _block_metrics_horizontal(adjusted, size, config)
    else:
        bbox, block_centre = _block_metrics(adjusted, size, config)
    # Vertical blocks are top-aligned (all columns share the top edge),
    # so only the horizontal centring against the island centre is
    # checked; horizontal blocks are centred both ways.
    if direction == "horizontal":
        offset = math.hypot(
            block_centre[0] - centre[0], block_centre[1] - centre[1]
        )
    else:
        offset = abs(block_centre[0] - centre[0])
    diagonal = math.hypot(x2 - x1, y2 - y1)
    tolerance = diagonal * config.center_tolerance_fraction
    status = "typeset"
    audit_reasons = list(reasons)
    if any("soft cap" in reason for reason in reasons):
        # The block exceeds the island bbox (within the 150% envelope):
        # rendered but flagged for review.
        status = "typeset_pending_review"
    if ratio is not None and (ratio < config.min_ratio or ratio > config.max_ratio):
        # The area ratio is a quality signal only; the fit itself keeps
        # the block inside the island bbox (the canonical box).
        status = "typeset_pending_review"
    if offset > tolerance:
        status = "typeset_pending_review"
        audit_reasons.append(f"center offset {offset:.1f}px > {tolerance:.1f}px tolerance")
    area_text = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return {
        "island_id": island.id, "text": text, "direction": direction,
        "font_size": size, "stroke_width": config.stroke(size),
        "tracking": config.tracking(size), "column_gap": config.column_gap(size),
        "area_base": area_rect, "area_text": area_text, "area_ratio": ratio,
        "center_offset": round(offset, 2), "center_tolerance": round(tolerance, 2),
        "block_bbox": bbox, "slots": tuple(adjusted), "status": status,
        "reasons": tuple(audit_reasons), "margin": margin,
    }


def typeset_island_page(
    erased: Image.Image,
    dataset: IslandDataset,
    translations: Mapping[str, str],
    paddle_lines: Sequence[OcrLine] = (),
    *,
    config: TypesetConfig = TypesetConfig(),
    geometry_overrides: Optional[
        Mapping[str, Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]], bool, Sequence[ColumnGroup]]]
    ] = None,
    direction_overrides: Optional[Mapping[str, str]] = None,
    union_overrides: Optional[Mapping[str, dict]] = None,
    capped_islands: Optional[Collection[str]] = None,
    font_ceiling_px: Optional[int] = None,
) -> IslandTypesetResult:
    if erased.size != dataset.image_size:
        raise ErasureError(
            f"erased image size {erased.size} differs from islands.json {dataset.image_size}"
        )
    started = time.perf_counter()
    canvas = erased.convert("RGB").copy()
    capped_ids = frozenset(capped_islands or ())
    audits = []
    for island in dataset.islands:
        if island.id not in translations:
            audits.append(
                IslandTypesetAudit(
                    island_id=island.id, text="", font_size=0, stroke_width=0,
                    tracking=0, column_gap=0, area_base=None, area_text=None,
                    area_ratio=None, center_offset=0.0, center_tolerance=0.0,
                    block_bbox=None, slots=(), status="skipped",
                    reasons=("no translation (dropped tiny / excluded)",),
                    changed_inside=0, changed_outside=0,
                )
            )
            continue
        # 省略号统一 3 点（两边路径共用同一规则；覆盖保留源文本的符号岛
        # 与历史译文，重嵌字即生效）
        text = normalize_ellipsis(str(translations.get(island.id) or "")).strip()
        if not text:
            audits.append(
                IslandTypesetAudit(
                    island_id=island.id, text="", font_size=0, stroke_width=0,
                    tracking=0, column_gap=0, area_base=None, area_text=None,
                    area_ratio=None, center_offset=0.0, center_tolerance=0.0,
                    block_bbox=None, slots=(), status="translation_failed",
                    reasons=("empty translation",),
                    changed_inside=0, changed_outside=0,
                )
            )
            continue
        direction = str((direction_overrides or {}).get(island.id) or "vertical")
        if direction not in ("horizontal", "vertical"):
            direction = "vertical"
        plan = _compute_island_plan(
            island, text, direction, config,
            union_overrides=union_overrides,
            sfx=island.id in capped_ids,
            ceiling_px=font_ceiling_px,
        )
        if plan.get("block_bbox") is None:
            audits.append(
                IslandTypesetAudit(
                    island_id=island.id, text=plan.get("text") or text,
                    font_size=0, stroke_width=0, tracking=0, column_gap=0,
                    area_base=plan.get("area_base"), area_text=None,
                    area_ratio=None, center_offset=0.0, center_tolerance=0.0,
                    block_bbox=None, slots=(), status="translation_failed",
                    reasons=tuple(plan.get("reasons") or ()),
                    changed_inside=0, changed_outside=0,
                )
            )
            continue
        _draw_slots(
            canvas, plan["text"], plan["slots"], plan["font_size"],
            config, direction=direction,
        )
        audits.append(
            IslandTypesetAudit(
                island_id=island.id, text=plan["text"],
                font_size=plan["font_size"], stroke_width=plan["stroke_width"],
                tracking=plan["tracking"], column_gap=plan["column_gap"],
                area_base=plan["area_base"], area_text=plan["area_text"],
                area_ratio=plan["area_ratio"],
                center_offset=plan["center_offset"],
                center_tolerance=plan["center_tolerance"],
                block_bbox=plan["block_bbox"], slots=plan["slots"],
                status=plan["status"], reasons=plan["reasons"],
                changed_inside=0, changed_outside=0,
            )
        )
    difference = ImageChops.difference(erased, canvas)
    changed = difference.convert("L").point(lambda value: 255 if value else 0)
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    for island in dataset.islands:
        # 掩膜按 STROKE_OVERFLOW_MARGIN 放宽：描边向外画，贴着框边时会伸出去，
        # 不放宽就会被"岛外零改动"的硬裁剪切掉（边缘字缺笔）。这一带在 stage2
        # 已被一并擦除（extra_regions 同样放宽），所以描边落在已擦除区域上，
        # 不会覆盖原图像素。
        x1, y1, x2, y2 = island.image_bbox
        m = STROKE_OVERFLOW_MARGIN
        draw.rectangle(
            (max(0, x1 - m), max(0, y1 - m),
             min(canvas.size[0], x2 + m), min(canvas.size[1], y2 + m)),
            fill=255,
        )
    # Hard island-exterior guarantee: nothing outside the island bboxes may
    # reach the output, so the final composite clips to the island union.
    canvas = Image.composite(canvas, erased, mask)
    difference = ImageChops.difference(erased, canvas)
    changed = difference.convert("L").point(lambda value: 255 if value else 0)
    outside = ImageChops.darker(changed, ImageChops.invert(mask)).histogram()[255]
    if outside:
        raise TypesettingError(
            f"typesetter changed {outside} pixels outside the island bboxes"
        )
    return IslandTypesetResult(
        image=canvas,
        difference=difference,
        changed_mask=changed,
        audits=tuple(audits),
        typeset_seconds=time.perf_counter() - started,
    )


def infer_box_direction(bbox) -> str:
    """Photos blue-box direction: wider than tall => horizontal."""
    x1, y1, x2, y2 = bbox
    return "horizontal" if (x2 - x1) >= (y2 - y1) else "vertical"


def infer_island_directions(
    selection_blocks_path,
    islands: Sequence[IslandInfo],
) -> Mapping[str, str]:
    """Derive each island's typesetting direction from its blue boxes.

    An island whose atomic blue boxes are mostly horizontal (width >=
    height) is horizontal; otherwise vertical.  Islands without blue boxes
    default to vertical.  Returns ``{island_id: "horizontal"|"vertical"}``.
    """
    import json

    result: dict[str, str] = {}
    try:
        payload = json.loads(
            Path(selection_blocks_path).read_text(encoding="utf-8-sig")
        )
        boxes = payload.get("selection_blocks") or []
    except (OSError, ValueError, json.JSONDecodeError):
        boxes = []
    box_boxes = [
        list(item.get("image_bbox") or ()) for item in boxes
        if len(item.get("image_bbox") or ()) == 4
    ]
    for island in islands:
        ix1, iy1, ix2, iy2 = island.image_bbox
        inside = [
            box for box in box_boxes
            if ix1 <= (box[0] + box[2]) // 2 <= ix2
            and iy1 <= (box[1] + box[3]) // 2 <= iy2
        ]
        if not inside:
            result[island.id] = "vertical"
            continue
        horizontal = sum(
            1 for box in inside if infer_box_direction(box) == "horizontal"
        )
        result[island.id] = (
            "horizontal" if horizontal * 2 >= len(inside) else "vertical"
        )
    return result
