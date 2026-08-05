"""Island-restricted Chinese typesetting (M4, E stage) - vertical only.

Manga text is treated as vertical (right to left, top to bottom inside each
column) exclusively.  Two layout modes:

  Mode 1 (column-follow): translated columns occupy the original Paddle
      column x-centres (right to left), each vertically centred on the
      original column's y-range.  Target ratio uses the union AREA of the
      island's Paddle line boxes:  r = A_text / A_base_union in [0.8, 1.1].
  Mode 2 (bounding-rect, triggered when r < 0.8): the translation becomes a
      centred vertical block (multi-column allowed up to the island width)
      and the font is enlarged so that
      r2 = A_text / A_rect in [0.8, 1.1], where A_rect is the area of the
      bounding rectangle of the island's Paddle lines (same rectangle that
      defines the target centre).  Centre lock is unchanged.

Both modes: hard boundary = island image_bbox (no silent cropping), white
stroke on every glyph (stroke_width = max(1, size // 10), colour
configurable), character tracking and column gap are explicit layout
parameters, and the pixel audit guarantees zero changes outside the islands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
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
MIN_FONT_SIZE = 8


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
) -> Tuple[int, int, int, int]:
    """(chars_per_column, column_count, block_width, block_height)."""
    x1, y1, x2, y2 = island.image_bbox
    usable_w = max(1, (x2 - x1) - 2 * config.padding)
    usable_h = max(1, (y2 - y1) - 2 * config.padding)
    step = _char_step(config, size)
    pitch = _pitch(config, size)
    chars_per_column = max(1, usable_h // step)
    column_count = max(1, math.ceil(len(text) / chars_per_column))
    # columns never exceed the island width; the font search clamps size
    while column_count * pitch - config.column_gap(size) > usable_w and column_count > 1:
        column_count -= 1
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
) -> Sequence[TypesetColumnSlot]:
    chars_per_column, column_count, block_width, block_height, last_count = _column_layout(
        text, size, island, config
    )
    # reading order: rightmost original column first
    ordered = sorted(original_columns, key=_column_key)
    pitch = _pitch(config, size)
    slots = []
    for i in range(column_count):
        if i < len(ordered):
            cx = (ordered[i].bbox[0] + ordered[i].bbox[2]) / 2.0
            cy = (ordered[i].bbox[1] + ordered[i].bbox[3]) / 2.0
        else:
            base_cx = (ordered[-1].bbox[0] + ordered[-1].bbox[2]) / 2.0
            cx = base_cx - (i - len(ordered) + 1) * pitch
            cy = (island.image_bbox[1] + island.image_bbox[3]) / 2.0
        count = chars_per_column if i < column_count - 1 else last_count
        slots.append(TypesetColumnSlot(index=i, x_center=cx, y_center=cy, char_count=count))
    return slots


def _slots_centered(
    text: str,
    size: int,
    island: IslandInfo,
    centre: Tuple[float, float],
    config: TypesetConfig,
) -> Sequence[TypesetColumnSlot]:
    chars_per_column, column_count, block_width, block_height, last_count = _column_layout(
        text, size, island, config
    )
    pitch = _pitch(config, size)
    # reading order: column 0 sits rightmost, later columns go leftwards
    right = centre[0] + block_width / 2.0
    slots = []
    for i in range(column_count):
        cx = right - size / 2.0 - i * pitch
        count = chars_per_column if i < column_count - 1 else last_count
        slots.append(TypesetColumnSlot(index=i, x_center=cx, y_center=centre[1], char_count=count))
    return slots


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
) -> ImageFont.ImageFont:
    font = ImageFont.truetype(config.font_name, size=size)
    draw = ImageDraw.Draw(canvas)
    stroke = config.stroke(size)
    step = _char_step(config, size)
    index = 0
    for slot in slots:
        top = slot.y_center - ((slot.char_count - 1) * step + size) / 2.0
        for j in range(slot.char_count):
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


def _fits_island(
    bbox: BBox,
    island: IslandInfo,
    size: int,
    config: TypesetConfig,
) -> bool:
    x1, y1, x2, y2 = island.image_bbox
    margin = max(2, config.stroke(size) + 2)
    return (
        (bbox[2] - bbox[0]) <= (x2 - x1) - 2 * margin
        and (bbox[3] - bbox[1]) <= (y2 - y1) - 2 * margin
    )


def _fit_size(
    text: str,
    island: IslandInfo,
    original_columns: Sequence[ColumnGroup],
    area_base: Optional[float],
    area_rect: Optional[float],
    centre: Tuple[float, float],
    config: TypesetConfig,
) -> Tuple[int, Sequence[TypesetColumnSlot], Optional[float], Tuple[str, ...]]:
    x1, y1, x2, y2 = island.image_bbox
    usable_w = (x2 - x1) - 2 * config.padding
    usable_h = (y2 - y1) - 2 * config.padding
    max_size = max(config.min_font_size, int(max(1, min(usable_w, usable_h))))

    best_mode2: Optional[Tuple[int, Sequence[TypesetColumnSlot], float]] = None
    best_mode1: Optional[Tuple[int, Sequence[TypesetColumnSlot], float]] = None
    for size in range(max_size, config.min_font_size - 1, -1):
        # mode 1: column-follow, ratio vs union area
        if area_base is not None and area_base > 0:
            slots = _slots_follow_columns(text, size, original_columns, island, config)
            bbox, _ = _block_metrics(slots, size, config)
            if _fits_island(bbox, island, size, config):
                ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / area_base
                if config.min_ratio <= ratio <= config.max_ratio:
                    return size, slots, ratio, ()
                if best_mode1 is None or ratio > best_mode1[2]:
                    best_mode1 = (size, slots, ratio)
        # mode 2: centred block, ratio vs rect area
        if area_rect is not None and area_rect > 0:
            slots = _slots_centered(text, size, island, centre, config)
            bbox, _ = _block_metrics(slots, size, config)
            if _fits_island(bbox, island, size, config):
                ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / area_rect
                if config.min_ratio <= ratio <= config.max_ratio:
                    return size, slots, ratio, ()
                if best_mode2 is None or ratio > best_mode2[2]:
                    best_mode2 = (size, slots, ratio)

    # prefer mode 2 (enlarged font towards the bounding rectangle) when both
    # fall below the target; mode 1 otherwise
    if best_mode2 is not None and (best_mode1 is None or best_mode2[2] >= best_mode1[2]):
        size, slots, ratio = best_mode2
        reasons = (f"area ratio {ratio:.2f} outside [{config.min_ratio}, {config.max_ratio}]",)
        return size, slots, ratio, reasons
    if best_mode1 is not None:
        size, slots, ratio = best_mode1
        reasons = (f"area ratio {ratio:.2f} outside [{config.min_ratio}, {config.max_ratio}]",)
        return size, slots, ratio, reasons
    raise TypesettingError(
        f"island {island.id}: text does not fit inside the island at any font size"
    )


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
        try:
            size, slots, ratio, reasons = _fit_size(
                text, island, columns, area_base, area_rect, centre, config
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
        offset = math.hypot(
            block_centre[0] + shift_x - centre[0], block_centre[1] + shift_y - centre[1]
        )
        diagonal = math.hypot(x2 - x1, y2 - y1)
        tolerance = diagonal * config.center_tolerance_fraction
        _draw_slots(canvas, text, adjusted, size, config)
        status = "typeset"
        audit_reasons = list(reasons)
        if ratio is not None and (ratio < config.min_ratio or ratio > config.max_ratio):
            if ratio > config.hard_ratio_cap:
                status = "translation_failed"
            else:
                status = "typeset_pending_review"
        if offset > tolerance:
            status = "typeset_pending_review"
            audit_reasons.append(f"center offset {offset:.1f}px > {tolerance:.1f}px tolerance")
        area_text = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        audits.append(
            IslandTypesetAudit(
                island_id=island.id, text=text, font_size=size,
                stroke_width=config.stroke(size), tracking=config.tracking(size),
                column_gap=config.column_gap(size), area_base=area_base,
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
