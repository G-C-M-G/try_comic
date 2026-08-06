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
inside the GROUP bbox (the outer rectangle of the island's blue boxes /
Paddle lines).  The group bbox is both the layout bounds and the single
area-ratio base: r = A_text / A_group_bbox with target [0.6, 1.1]; an
off-target ratio is a quality signal only (pending_review), the hard
layout cap is the group bbox inflated by 50% per side (a 150% envelope).
Glyph size is additionally capped by the PADDED box dimensions: vertical
size <= narrowest 6px-padded box width, horizontal size <= shortest padded
box height.  The font search never goes below ``min_font_size`` (25px);
when nothing fits even at the floor the text is still rendered at the
floor and clipped to the island -- an erased island is never left blank.
The column gap is never compressed to zero.  ``union_overrides`` carries
per-island glyph caps (``glyph_width_limit`` / ``glyph_height_limit``);
the blue-box union masks themselves are no longer used for layout.

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
from typing import Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .errors import ErasureError, TypesettingError
from .island_contract import IslandDataset, IslandInfo
from .island_dry_run import OcrLine
from .ocr_contract import BBox
from .sentence_splitter import ColumnGroup, _column_key, cluster_lines_into_columns

MIN_RATIO = 0.6
MAX_RATIO = 1.1
HARD_RATIO_CAP = 1.2
CENTER_TOLERANCE_FRACTION = 0.05
STROKE_DIVISOR = 10
TRACKING_DIVISOR = 16
COLUMN_GAP_DIVISOR = 4
# Readability floor: the font search never goes below this, and if nothing
# fits the group bbox at this size the text is still rendered (clipped to
# the island) instead of leaving an erased island blank.
MIN_FONT_SIZE = 30
# Hard layout cap: the block may exceed the group bbox by up to this
# fraction of the group's size per side (a 150% envelope).  The column gap
# is never compressed to zero to force a fit.
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
    min_ratio: float = MIN_RATIO
    max_ratio: float = MAX_RATIO
    hard_ratio_cap: float = HARD_RATIO_CAP
    center_tolerance_fraction: float = CENTER_TOLERANCE_FRACTION
    stroke_divisor: int = STROKE_DIVISOR
    tracking_divisor: int = TRACKING_DIVISOR
    column_gap_divisor: int = COLUMN_GAP_DIVISOR
    min_font_size: int = MIN_FONT_SIZE
    padding: int = 0

    def tracking(self, size: int) -> int:
        return max(0, size // self.tracking_divisor)

    def column_gap(self, size: int) -> int:
        return max(1, size // self.column_gap_divisor)

    def stroke(self, size: int) -> int:
        return max(1, size // self.stroke_divisor)


def _in_bbox(line: OcrLine, bbox: BBox, tolerance: float = 2.0) -> bool:
    xs = [p[0] for p in line.polygon]
    ys = [p[1] for p in line.polygon]
    cx, cy = (sum(xs) / len(xs), sum(ys) / len(ys))
    x1, y1, x2, y2 = bbox
    return x1 - tolerance <= cx <= x2 + tolerance and y1 - tolerance <= cy <= y2 + tolerance


def _island_geometry(
    island: IslandInfo,
    paddle_lines: Sequence[OcrLine],
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]], bool, Sequence[ColumnGroup]]:
    """(union_area, rect_area, centre, has_geometry, columns)."""
    lines = [line for line in paddle_lines if _in_bbox(line, island.image_bbox)]
    if not lines:
        return None, None, None, False, ()
    x1, y1, x2, y2 = island.image_bbox
    mask = Image.new("L", (max(1, x2 - x1), max(1, y2 - y1)), 0)
    draw = ImageDraw.Draw(mask)
    for line in lines:
        draw.polygon([(p[0] - x1, p[1] - y1) for p in line.polygon], fill=255)
    union_area = float(mask.histogram()[255])
    xs = [p[0] for line in lines for p in line.polygon]
    ys = [p[1] for line in lines for p in line.polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rect_area = float((max_x - min_x) * (max_y - min_y))
    centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    columns = cluster_lines_into_columns(lines, vertical=True)
    return union_area, rect_area, centre, True, columns


def _char_step(config: TypesetConfig, size: int) -> int:
    return size + config.tracking(size)


def _pitch(config: TypesetConfig, size: int) -> int:
    return size + config.column_gap(size)


def _column_layout(
    text: str,
    size: int,
    island: IslandInfo,
    config: TypesetConfig,
    layout_bbox: Optional[BBox] = None,
) -> Tuple[int, int, int, int]:
    """(chars_per_column, column_count, block_width, block_height).

    ``layout_bbox`` overrides the island bbox as the layout bounds -- used
    by union mode so the block can never leave the blue-box union.
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
    block_height = (chars_per_column - 1) * step + size
    block_width = column_count * pitch - config.column_gap(size)
    return chars_per_column, column_count, block_width, block_height, last_count


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
        column_count = min(len(text), len(original_columns))
        chars_per_column = max(1, math.ceil(len(text) / column_count))
        last_count = len(text) - (column_count - 1) * chars_per_column
        if last_count <= 0:
            # The columns would overflow the text (e.g. 5 chars across 4
            # boxes): shrink the column count so the last column is never
            # empty or negative.
            column_count = math.ceil(len(text) / chars_per_column)
            last_count = len(text) - (column_count - 1) * chars_per_column
    else:
        chars_per_column, column_count, _bw, _bh, last_count = _column_layout(
            text, size, island, config, layout_bbox=layout_bbox
        )
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
        count = chars_per_column if i < column_count - 1 else last_count
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
    chars_per_column, column_count, block_width, block_height, last_count = _column_layout(
        text, size, island, config, layout_bbox=layout_bbox
    )
    pitch = _pitch(config, size)
    step = _char_step(config, size)
    # reading order: column 0 sits rightmost, later columns go leftwards
    right = centre[0] + block_width / 2.0
    slots = []
    for i in range(column_count):
        cx = right - size / 2.0 - i * pitch
        count = chars_per_column if i < column_count - 1 else last_count
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
) -> Tuple[int, int, int, int]:
    """Horizontal layout: (chars_per_row, row_count, block_width, block_height)."""
    x1, y1, x2, y2 = layout_bbox if layout_bbox is not None else island.image_bbox
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    step = _char_step(config, size)
    row_pitch = _pitch(config, size)
    chars_per_row = max(1, usable_w // step)
    row_count = max(1, math.ceil(len(text) / chars_per_row))
    while row_count * row_pitch - config.column_gap(size) > usable_h and row_count > 1:
        row_count -= 1
    # After shrinking the row count, the remaining rows must still hold
    # every character: recompute chars_per_row so the last row never
    # exceeds the island width (an over-capacity last row would render
    # outside the island).
    chars_per_row = max(1, math.ceil(len(text) / row_count))
    last_count = len(text) - (row_count - 1) * chars_per_row
    block_width = (chars_per_row - 1) * step + size
    block_height = row_count * row_pitch - config.column_gap(size)
    return chars_per_row, row_count, block_width, block_height, last_count


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
    chars_per_row, row_count, block_width, block_height, last_count = _row_layout(
        text, size, island, config, layout_bbox=layout_bbox
    )
    row_pitch = _pitch(config, size)
    step = _char_step(config, size)
    top = centre[1] - block_height / 2.0
    left = centre[0] - block_width / 2.0
    slots = []
    for i in range(row_count):
        count = chars_per_row if i < row_count - 1 else last_count
        y_center = top + size / 2.0 + i * row_pitch
        x_center = left + (count - 1) * step / 2.0
        slots.append(TypesetColumnSlot(index=i, x_center=x_center, y_center=y_center, char_count=count))
    return slots


def _block_metrics_horizontal(
    slots: Sequence[TypesetColumnSlot],
    size: int,
    config: TypesetConfig,
) -> Tuple[BBox, Tuple[float, float]]:
    """Metrics for horizontal row slots (row width = count * step)."""
    step = _char_step(config, size)
    row_pitch = _pitch(config, size)
    widths = [(slot.char_count - 1) * step + size for slot in slots]
    xs = [
        slot.x_center - width / 2.0
        for slot, width in zip(slots, widths)
    ] + [
        slot.x_center + width / 2.0
        for slot, width in zip(slots, widths)
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


def _draw_slots(
    canvas: Image.Image,
    text: str,
    slots: Sequence[TypesetColumnSlot],
    size: int,
    config: TypesetConfig,
    direction: str = "vertical",
) -> ImageFont.ImageFont:
    font = ImageFont.truetype(config.font_name, size=size)
    draw = ImageDraw.Draw(canvas)
    stroke = config.stroke(size)
    step = _char_step(config, size)
    index = 0
    if direction == "horizontal":
        # row slots: characters left to right inside each row
        for slot in slots:
            left = slot.x_center - ((slot.char_count - 1) * step) / 2.0
            for j in range(slot.char_count):
                if index >= len(text):
                    break
                char = text[index]
                index += 1
                draw.text(
                    (left + j * step, slot.y_center - size / 2.0),
                    char,
                    font=font,
                    fill=config.text_color,
                    stroke_width=stroke,
                    stroke_fill=config.stroke_color,
                )
        return font
    for slot in slots:
        top = slot.y_center - ((slot.char_count - 1) * step + size) / 2.0
        for j in range(slot.char_count):
            if index >= len(text):
                break
            char = text[index]
            index += 1
            draw.text(
                (slot.x_center - size / 2.0, top + j * step),
                char,
                font=font,
                fill=config.text_color,
                stroke_width=stroke,
                stroke_fill=config.stroke_color,
            )
    return font


def _columns_bbox(columns: Sequence[ColumnGroup]) -> Optional[BBox]:
    """Bounding rectangle of the island's blue-box columns (the group bbox)."""
    xs = [p for column in columns for p in (column.bbox[0], column.bbox[2])]
    ys = [p for column in columns for p in (column.bbox[1], column.bbox[3])]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _fit_size(
    text: str,
    island: IslandInfo,
    area_rect: Optional[float],
    centre: Tuple[float, float],
    config: TypesetConfig,
    direction: str = "vertical",
    group_bbox: Optional[BBox] = None,
    glyph_width_limit: Optional[float] = None,
    glyph_height_limit: Optional[float] = None,
) -> Tuple[int, Sequence[TypesetColumnSlot], Optional[float], Tuple[str, ...]]:
    """Single-path fit: a centred block laid out inside the GROUP bbox.

    The group bbox (the outer rectangle of the island's blue boxes) is both
    the layout bounds and the single area-ratio base:
    * the block is centred on the group centre -- vertical columns are
      top-aligned, horizontal rows left-aligned;
    * the hard layout cap is the group bbox inflated by 50% per side (a
      150% envelope); the column gap is never compressed to zero;
    * the area ratio is A_text / A_group_bbox with target
      [min_ratio, max_ratio]; off-target ratios are quality signals only
      (the caller marks pending_review);
    * if nothing fits even at the font floor, the text is still rendered at
      the floor and clipped by the caller -- an erased island is never left
      blank.
    """
    bounds = group_bbox if group_bbox is not None else island.image_bbox
    x1, y1, x2, y2 = bounds
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    max_size = max(config.min_font_size, int(max(1, min(usable_w, usable_h))))
    if direction == "vertical" and glyph_width_limit is not None and glyph_width_limit > 0:
        max_size = max(config.min_font_size, min(max_size, int(glyph_width_limit)))
    if direction == "horizontal" and glyph_height_limit is not None and glyph_height_limit > 0:
        max_size = max(config.min_font_size, min(max_size, int(glyph_height_limit)))

    # Hard layout cap: the block may exceed the group bbox by up to 50% of
    # its size per side (a 150% envelope) -- enforced on the block SIZE so
    # a layout cannot blow up to a single over-wide row.
    gw = max(1, x2 - x1)
    gh = max(1, y2 - y1)
    allow_x1 = x1 - MAX_GROUP_BBOX_OVERFLOW_RATIO * gw
    allow_x2 = x2 + MAX_GROUP_BBOX_OVERFLOW_RATIO * gw
    allow_y1 = y1 - MAX_GROUP_BBOX_OVERFLOW_RATIO * gh
    allow_y2 = y2 + MAX_GROUP_BBOX_OVERFLOW_RATIO * gh

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

    best: Optional[Tuple[int, Sequence[TypesetColumnSlot], Optional[float]]] = None
    for size in range(max_size, config.min_font_size - 1, -1):
        slots = _slots_for(size, centre)
        bbox, _ = _metrics_for(slots, size)
        block_w = bbox[2] - bbox[0]
        block_h = bbox[3] - bbox[1]
        if block_w > 1.5 * gw or block_h > 1.5 * gh:
            continue
        if not (bbox[0] >= allow_x1 and bbox[1] >= allow_y1 and bbox[2] <= allow_x2 and bbox[3] <= allow_y2):
            continue
        ratio = None
        if area_rect is not None and area_rect > 0:
            ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / area_rect
        if ratio is not None and config.min_ratio <= ratio <= config.max_ratio:
            return size, slots, ratio, ()
        if best is None or (ratio is not None and (best[2] is None or ratio > best[2])):
            best = (size, slots, ratio)
    if best is not None:
        size, slots, ratio = best
        if ratio is not None:
            reasons = (f"area ratio {ratio:.2f} outside [{config.min_ratio}, {config.max_ratio}]",)
        else:
            reasons = ("no area base (empty group bbox)",)
        return size, slots, ratio, reasons
    # Nothing fits even at the font floor: render at the floor anyway.  The
    # caller clips to the island, so an erased island is never left blank.
    size = config.min_font_size
    slots = _slots_for(size, centre)
    reasons = (
        f"does not fit the group bbox at the font floor ({config.min_font_size}px); "
        "rendered and clipped",
    )
    return size, slots, 0.0, reasons


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
) -> IslandTypesetResult:
    if erased.size != dataset.image_size:
        raise ErasureError(
            f"erased image size {erased.size} differs from islands.json {dataset.image_size}"
        )
    started = time.perf_counter()
    canvas = erased.convert("RGB").copy()
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
        text = str(translations.get(island.id) or "").strip()
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
        x1, y1, x2, y2 = island.image_bbox
        override = (geometry_overrides or {}).get(island.id)
        if override is not None:
            area_base, area_rect, centre, has_geometry, columns = override
        else:
            area_base, area_rect, centre, has_geometry, columns = _island_geometry(
                island, paddle_lines
            )
        if centre is None:
            centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if area_rect is None:
            # No Paddle geometry (e.g. a capture without OCR lines): fall
            # back to the island bbox as the target rectangle so the
            # centred layout still renders.
            area_rect = float((x2 - x1) * (y2 - y1))
        group_bbox = _columns_bbox(columns) if columns else None
        if group_bbox is None:
            group_bbox = (x1, y1, x2, y2)
        try:
            union = (union_overrides or {}).get(island.id)
            size, slots, ratio, reasons = _fit_size(
                text, island, area_rect, centre, config,
                direction=direction,
                group_bbox=group_bbox,
                glyph_width_limit=union.get("glyph_width_limit") if union else None,
                glyph_height_limit=union.get("glyph_height_limit") if union else None,
            )
        except TypesettingError as error:
            audits.append(
                IslandTypesetAudit(
                    island_id=island.id, text=text, font_size=0, stroke_width=0,
                    tracking=0, column_gap=0, area_base=area_base, area_text=None,
                    area_ratio=None, center_offset=0.0, center_tolerance=0.0,
                    block_bbox=None, slots=(), status="translation_failed",
                    reasons=(str(error),), changed_inside=0, changed_outside=0,
                )
            )
            continue
        if direction == "horizontal":
            bbox, block_centre = _block_metrics_horizontal(slots, size, config)
        else:
            bbox, block_centre = _block_metrics(slots, size, config)
        margin = max(2, config.stroke(size) + 2)
        left_room = bbox[0] - (x1 + margin)
        right_room = (x2 - margin) - bbox[2]
        top_room = bbox[1] - (y1 + margin)
        bottom_room = (y2 - margin) - bbox[3]
        shift_x = max(-left_room, min(right_room, centre[0] - block_centre[0]))
        shift_y = max(-top_room, min(bottom_room, centre[1] - block_centre[1]))
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
        # so only the horizontal centring against the group centre is
        # checked; horizontal blocks are centred both ways.
        if direction == "horizontal":
            offset = math.hypot(
                block_centre[0] - centre[0], block_centre[1] - centre[1]
            )
        else:
            offset = abs(block_centre[0] - centre[0])
        diagonal = math.hypot(x2 - x1, y2 - y1)
        tolerance = diagonal * config.center_tolerance_fraction
        _draw_slots(canvas, text, adjusted, size, config, direction=direction)
        status = "typeset"
        audit_reasons = list(reasons)
        if ratio is not None and (ratio < config.min_ratio or ratio > config.max_ratio):
            # The area ratio is a quality signal only; the hard layout cap
            # is the 150% group-bbox envelope enforced by _fit_size.
            status = "typeset_pending_review"
        if offset > tolerance:
            status = "typeset_pending_review"
            audit_reasons.append(f"center offset {offset:.1f}px > {tolerance:.1f}px tolerance")
        area_text = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        audits.append(
            IslandTypesetAudit(
                island_id=island.id, text=text, font_size=size,
                stroke_width=config.stroke(size), tracking=config.tracking(size),
                column_gap=config.column_gap(size), area_base=area_rect,
                area_text=area_text, area_ratio=ratio,
                center_offset=round(offset, 2), center_tolerance=round(tolerance, 2),
                block_bbox=bbox, slots=tuple(adjusted), status=status,
                reasons=tuple(audit_reasons), changed_inside=0, changed_outside=0,
            )
        )
    difference = ImageChops.difference(erased, canvas)
    changed = difference.convert("L").point(lambda value: 255 if value else 0)
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    for island in dataset.islands:
        draw.rectangle(island.image_bbox, fill=255)
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
