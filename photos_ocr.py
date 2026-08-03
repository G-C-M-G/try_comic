"""Normalize column-level OCR captured from Windows Photos.

Windows Photos selects one vertical column at a time.  A translation unit can
contain several of those columns, so this module keeps both levels:

* column: text and geometry copied from one Photos selection;
* region: one or more columns ordered for translation and later rendering.

The module deliberately does not automate the Photos UI.  It consumes a JSON
manifest produced by a capture helper or by manual calibration, which keeps the
unstable UI boundary separate from the translation pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, replace
from statistics import median
from typing import Iterable, Sequence


BBox = tuple[float, float, float, float]
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_TERMINAL_RE = re.compile(r"[。！？!?…]+[』」）)]?$|(?:です|ます)[。！？!?]$")
_SKIPPED_ROLES = {"ruby", "sfx", "ignore"}


def _bbox(value: Sequence[float], field: str = "bbox") -> BBox:
    if len(value) != 4:
        raise ValueError(f"{field} must contain four numbers")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"{field} contains a non-finite value")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field} must have positive width and height")
    return x1, y1, x2, y2


def normalize_column_text(text: str, direction: str = "vertical") -> str:
    """Remove selection line breaks without breaking horizontal Latin text."""
    value = str(text or "").strip()
    if direction == "vertical" and _CJK_RE.search(value):
        return re.sub(r"\s+", "", value)
    return re.sub(r"\s+", " ", value)


def map_display_bbox(
    display_bbox: Sequence[float],
    image_viewport: Sequence[float],
    image_size: Sequence[int],
) -> BBox:
    """Map a Photos-window selection rectangle back to source image pixels.

    ``image_viewport`` is the exact displayed image rectangle inside Photos,
    excluding letterboxing and toolbars.  It must be measured for the same
    window state as ``display_bbox``.
    """
    bx1, by1, bx2, by2 = _bbox(display_bbox, "display_bbox")
    vx1, vy1, vx2, vy2 = _bbox(image_viewport, "image_viewport")
    if len(image_size) != 2:
        raise ValueError("image_size must contain width and height")
    image_w, image_h = (int(v) for v in image_size)
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_size must be positive")

    scale_x = image_w / (vx2 - vx1)
    scale_y = image_h / (vy2 - vy1)
    mapped = (
        (bx1 - vx1) * scale_x,
        (by1 - vy1) * scale_y,
        (bx2 - vx1) * scale_x,
        (by2 - vy1) * scale_y,
    )
    x1, y1, x2, y2 = mapped
    x1, x2 = sorted((max(0.0, min(float(image_w), x1)), max(0.0, min(float(image_w), x2))))
    y1, y2 = sorted((max(0.0, min(float(image_h), y1)), max(0.0, min(float(image_h), y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("display_bbox does not overlap the displayed image")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class OcrColumn:
    id: str
    text: str
    image_bbox: BBox
    selection_order: int = 0
    direction: str = "vertical"
    role: str = "text"
    panel_id: str | None = None
    container_id: str | None = None
    display_bbox: BBox | None = None
    break_after: bool = False

    @property
    def width(self) -> float:
        return self.image_bbox[2] - self.image_bbox[0]

    @property
    def height(self) -> float:
        return self.image_bbox[3] - self.image_bbox[1]

    @property
    def center_x(self) -> float:
        return (self.image_bbox[0] + self.image_bbox[2]) / 2


def _column_from_dict(
    value: dict,
    index: int,
    image_viewport: Sequence[float] | None,
    image_size: Sequence[int] | None,
) -> OcrColumn:
    column_id = str(value.get("id") or f"column_{index + 1:03d}")
    direction = str(value.get("direction") or "vertical").lower()
    text = normalize_column_text(value.get("text", ""), direction)
    display_bbox = value.get("display_bbox")
    image_bbox = value.get("image_bbox")
    if image_bbox is None:
        if display_bbox is None or image_viewport is None or image_size is None:
            raise ValueError(f"{column_id} needs image_bbox or display mapping data")
        image_bbox = map_display_bbox(display_bbox, image_viewport, image_size)
    else:
        image_bbox = _bbox(image_bbox, f"{column_id}.image_bbox")

    return OcrColumn(
        id=column_id,
        text=text,
        image_bbox=image_bbox,
        selection_order=int(value.get("selection_order", index)),
        direction=direction,
        role=str(value.get("role") or "text").lower(),
        panel_id=str(value["panel_id"]) if value.get("panel_id") is not None else None,
        container_id=(
            str(value["container_id"]) if value.get("container_id") is not None else None
        ),
        display_bbox=_bbox(display_bbox, f"{column_id}.display_bbox") if display_bbox else None,
        break_after=bool(value.get("break_after", False)),
    )


def _vertical_overlap(a: OcrColumn, b: OcrColumn) -> float:
    top = max(a.image_bbox[1], b.image_bbox[1])
    bottom = min(a.image_bbox[3], b.image_bbox[3])
    return max(0.0, bottom - top) / max(1.0, min(a.height, b.height))


def _horizontal_gap(a: OcrColumn, b: OcrColumn) -> float:
    ax1, _, ax2, _ = a.image_bbox
    bx1, _, bx2, _ = b.image_bbox
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0.0


def _can_group(a: OcrColumn, b: OcrColumn, max_gap: float, min_overlap: float) -> bool:
    if a.direction != b.direction:
        return False
    if a.panel_id and b.panel_id and a.panel_id != b.panel_id:
        return False
    if a.container_id is not None or b.container_id is not None:
        return a.container_id is not None and a.container_id == b.container_id
    if a.direction != "vertical":
        return False
    if _horizontal_gap(a, b) > max_gap:
        return False
    if _vertical_overlap(a, b) < min_overlap:
        return False
    width_ratio = max(a.width, b.width) / max(1.0, min(a.width, b.width))
    return width_ratio <= 2.25


def _ordered_columns(columns: Iterable[OcrColumn]) -> list[OcrColumn]:
    values = list(columns)
    if values and all(column.direction == "vertical" for column in values):
        return sorted(values, key=lambda column: (-column.center_x, column.image_bbox[1], column.selection_order))
    return sorted(values, key=lambda column: (column.image_bbox[1], column.image_bbox[0], column.selection_order))


def _quad(bbox: BBox) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [
        [round(x1), round(y1)],
        [round(x2), round(y1)],
        [round(x2), round(y2)],
        [round(x1), round(y2)],
    ]


def _sentence_groups(columns: Sequence[OcrColumn]) -> list[list[OcrColumn]]:
    """Split an ordered container at column-final punctuation or manual breaks."""
    result: list[list[OcrColumn]] = []
    current: list[OcrColumn] = []
    for column in columns:
        current.append(column)
        if column.break_after or _TERMINAL_RE.search(column.text):
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def group_columns(
    columns: Sequence[OcrColumn],
    *,
    max_gap_ratio: float = 2.4,
    min_vertical_overlap: float = 0.35,
) -> list[dict]:
    """Build conservative sentence/bubble regions from column OCR records.

    Explicit ``container_id`` values take priority.  Without them, only nearby
    similarly sized vertical columns with substantial y overlap are joined.
    Geometry-only multi-column regions are marked ``needs_review`` because two
    adjacent speech bubbles can otherwise look deceptively similar.
    """
    active = [column for column in columns if column.text and column.role not in _SKIPPED_ROLES]
    if not active:
        return []
    typical_width = median(column.width for column in active)
    max_gap = max(8.0, typical_width * max_gap_ratio)

    parent = list(range(len(active)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(active)):
        for right in range(left + 1, len(active)):
            if _can_group(active[left], active[right], max_gap, min_vertical_overlap):
                union(left, right)

    components: dict[int, list[OcrColumn]] = {}
    for index, column in enumerate(active):
        components.setdefault(find(index), []).append(column)

    ordered_components = sorted(
        components.values(),
        key=lambda group: (
            min(column.image_bbox[1] for column in group),
            -max(column.image_bbox[2] for column in group),
        ),
    )
    regions = []
    for component in ordered_components:
        ordered = _ordered_columns(component)
        explicit_container = all(column.container_id is not None for column in ordered)
        same_container = explicit_container and len({column.container_id for column in ordered}) == 1
        container_id = ordered[0].container_id if same_container else None
        for sentence_index, sentence in enumerate(_sentence_groups(ordered), 1):
            segments = [
                {"id": column.id, "text": column.text, "lines": _quad(column.image_bbox)}
                for column in sentence
            ]
            regions.append(
                {
                    "id": f"region_{len(regions) + 1:03d}",
                    "text": "".join(column.text for column in sentence),
                    "column_ids": [column.id for column in sentence],
                    "selection_orders": [column.selection_order for column in sentence],
                    "direction": sentence[0].direction,
                    "source_segments": segments,
                    "lines": [point for segment in segments for point in segment["lines"]],
                    "container_id": container_id,
                    "sentence_index": sentence_index,
                    "grouping_basis": "container" if same_container else "geometry",
                    "needs_review": len(sentence) > 1 and not same_container,
                }
            )
    return regions


def normalize_manifest(manifest: dict, *, include_non_dialogue: bool = False) -> dict:
    """Validate a capture manifest and append source-pixel columns and regions."""
    source = dict(manifest.get("source") or {})
    capture = dict(manifest.get("capture") or {})
    image_size = None
    if source.get("width") is not None and source.get("height") is not None:
        image_size = (int(source["width"]), int(source["height"]))
    image_viewport = capture.get("image_viewport")
    raw_columns = manifest.get("columns") or []
    columns = [
        _column_from_dict(value, index, image_viewport, image_size)
        for index, value in enumerate(raw_columns)
    ]
    if include_non_dialogue:
        columns = [replace(column, role="text") for column in columns]

    normalized_columns = []
    for column in columns:
        value = {
            "id": column.id,
            "text": column.text,
            "image_bbox": [round(v, 3) for v in column.image_bbox],
            "selection_order": column.selection_order,
            "direction": column.direction,
            "role": column.role,
        }
        if column.display_bbox:
            value["display_bbox"] = [round(v, 3) for v in column.display_bbox]
        if column.panel_id is not None:
            value["panel_id"] = column.panel_id
        if column.container_id is not None:
            value["container_id"] = column.container_id
        if column.break_after:
            value["break_after"] = True
        normalized_columns.append(value)

    result = dict(manifest)
    result["schema_version"] = 1
    result["columns"] = normalized_columns
    result["regions"] = group_columns(columns)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize Windows Photos column OCR manifest")
    parser.add_argument("manifest", help="input capture JSON")
    parser.add_argument("-o", "--output", help="output JSON (default: <input>.normalized.json)")
    parser.add_argument("--include-non-dialogue", action="store_true")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result = normalize_manifest(manifest, include_non_dialogue=args.include_non_dialogue)
    output = args.output or os.path.splitext(args.manifest)[0] + ".normalized.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
