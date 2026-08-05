from __future__ import annotations

from typing import Sequence, Tuple

from .errors import LayoutBoundaryError


BBox = Tuple[int, int, int, int]
VALID_OVERFLOW_LEVELS = (10, 20)


def validate_base_bbox(bbox: Sequence[float], image_size: Tuple[int, int]) -> BBox:
    if len(bbox) != 4:
        raise LayoutBoundaryError("image_bbox must contain four coordinates")
    if any(float(value) != int(value) for value in bbox):
        raise LayoutBoundaryError("image_bbox must use integer pixel coordinates")
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise LayoutBoundaryError(
            f"image_bbox {list(bbox)} is outside {width}x{height}"
        )
    return x1, y1, x2, y2


def compute_layout_bbox(
    complete_bbox: Sequence[float],
    overflow_level: int,
    image_size: Tuple[int, int],
) -> BBox:
    """Expand the reviewed complete OCR box directly by total width/height percent.

    The calculation always starts from ``complete_bbox``. Passing an already expanded
    10% box would violate the API contract and must be prevented by the caller's data
    model; there is intentionally no function that expands a layout box again.
    """

    if overflow_level not in VALID_OVERFLOW_LEVELS:
        raise LayoutBoundaryError(
            f"overflow_level must be 10 or 20, got {overflow_level}"
        )
    x1, y1, x2, y2 = validate_base_bbox(complete_bbox, image_size)
    width, height = image_size
    per_side_ratio = overflow_level / 200.0
    pad_x = max(1, round((x2 - x1) * per_side_ratio))
    pad_y = max(1, round((y2 - y1) * per_side_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def assert_layout_bbox_10pct(
    complete_bbox: Sequence[float],
    layout_bbox_10pct: Sequence[float],
    image_size: Tuple[int, int],
) -> None:
    expected = compute_layout_bbox(complete_bbox, 10, image_size)
    actual = validate_base_bbox(layout_bbox_10pct, image_size)
    if actual != expected:
        raise LayoutBoundaryError(
            "layout_bbox_10pct must be computed directly from the complete "
            f"image_bbox: expected {list(expected)}, got {list(actual)}"
        )


def measured_overflow_percent(
    complete_bbox: Sequence[float],
    actual_bbox: Sequence[float],
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    base = validate_base_bbox(complete_bbox, image_size)
    actual = validate_base_bbox(actual_bbox, image_size)
    base_width = base[2] - base[0]
    base_height = base[3] - base[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    return (
        max(0.0, (actual_width / base_width - 1.0) * 100.0),
        max(0.0, (actual_height / base_height - 1.0) * 100.0),
    )
