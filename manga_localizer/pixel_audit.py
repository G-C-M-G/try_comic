from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

from .ocr_contract import BBox


@dataclass(frozen=True)
class RegionChangeAudit:
    bbox: BBox
    center_bbox: BBox
    center_dimension_ratio: float
    sample_stride: int
    center_sampled_pixels: int
    center_changed_samples: int
    full_sampled_pixels: int
    full_changed_samples: int
    changed: bool
    status: str


def audit_region_change(
    before: Image.Image,
    after: Image.Image,
    bbox: BBox,
    *,
    center_dimension_ratio: float = 0.2,
    sample_stride: int = 2,
) -> RegionChangeAudit:
    """Quickly verify a rendered region using deterministic pixel sampling.

    The center check is intentionally small. The full box is sampled at a fixed
    stride as a fallback, avoiding an exhaustive per-pixel scan during retries.
    """

    if before.size != after.size:
        raise ValueError("before and after images must have the same size")
    if not (0 < center_dimension_ratio <= 1):
        raise ValueError("center_dimension_ratio must be in (0, 1]")
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= before.width and 0 <= y1 < y2 <= before.height):
        raise ValueError("audit bbox is outside the image")
    center_bbox = _center_bbox(bbox, center_dimension_ratio)
    center_sampled, center_changed = _sampled_change_count(
        before,
        after,
        center_bbox,
        sample_stride,
    )
    full_sampled, full_changed = _sampled_change_count(
        before,
        after,
        bbox,
        sample_stride,
    )
    changed = bool(center_changed or full_changed)
    return RegionChangeAudit(
        bbox=bbox,
        center_bbox=center_bbox,
        center_dimension_ratio=center_dimension_ratio,
        sample_stride=sample_stride,
        center_sampled_pixels=center_sampled,
        center_changed_samples=center_changed,
        full_sampled_pixels=full_sampled,
        full_changed_samples=full_changed,
        changed=changed,
        status="changed" if changed else "translation_failed",
    )


def _sampled_change_count(
    before: Image.Image,
    after: Image.Image,
    bbox: BBox,
    stride: int,
) -> Tuple[int, int]:
    first = np.asarray(before.convert("RGB").crop(bbox))[::stride, ::stride]
    second = np.asarray(after.convert("RGB").crop(bbox))[::stride, ::stride]
    changed = np.any(first != second, axis=2)
    return int(changed.size), int(np.count_nonzero(changed))


def _center_bbox(bbox: BBox, ratio: float) -> BBox:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    core_width = max(1, round(width * ratio))
    core_height = max(1, round(height * ratio))
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    core_x1 = max(x1, center_x - core_width // 2)
    core_y1 = max(y1, center_y - core_height // 2)
    return (
        core_x1,
        core_y1,
        min(x2, core_x1 + core_width),
        min(y2, core_y1 + core_height),
    )
