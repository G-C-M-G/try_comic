"""Detect Windows Photos Ctrl+A blue selection boxes.

Two modes
---------
* **dual** (preferred): ``photos_ocr`` (active, no select) + ``photos_selected``
  Uses change + blue dominance — same signal as ``photos_capture.detect_selection_boxes``.
* **single**: only the already-selected frame (semi-transparent blue overlays).
  Color mask + x-projection split. Slightly weaker (native blue art can false-hit).

Call for other AIs
------------------
::

    from blue_boxes import detect_blue_boxes_single, annotate_boxes
    boxes = detect_blue_boxes_single(image_rgb_uint8)  # [{id, bbox:[x1,y1,x2,y2]}]
    vis = annotate_boxes(image_rgb_uint8, boxes)

Algorithm (single frame)
------------------------
1. Blue mask: (B-R > t_br) & (B-G > t_bg) & (B > t_b), drop near-gray pixels.
2. MORPH_OPEN 3x3 only (never large CLOSE — adjacent columns sit a few px apart).
3. Connected components; drop tiny area/width/height.
4. Split each component on empty x-projection columns; tighten y to ink rows.
5. Sort (y, -x), id as sel_001...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Measured blend family for the semi-transparent Photos blue overlay
# (mask RGB(0, 120, 211), alpha 81/255): detection compares the scan-text
# frame with the all-selected frame; box pixels changed between the two
# frames (per-channel RGB delta) and sit in the bright blue overlay family
# (B high, R suppressed).
_GRAY_CHROMA = 18  # max(|R-G|+|G-B|+|B-R|)/2-ish gate via pairwise
_MIN_AREA = 40
_MIN_W = 5
_MIN_H = 7
_OPEN_K = 3


def _boxes_from_mask(mask: np.ndarray) -> list[dict]:
    """CC + x-projection split → list of {bbox, area}."""
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
    if mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_OPEN_K, _OPEN_K)),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    values: list[dict] = []
    for x, y, width, height, area in stats[1:]:
        if area < _MIN_AREA or width < _MIN_W or height < _MIN_H:
            continue
        sub = mask_u8[y : y + height, x : x + width]
        proj = sub.sum(axis=0)
        segments: list[tuple[int, int]] = []
        start = None
        for index in range(proj.shape[0] + 1):
            present = index < proj.shape[0] and proj[index] > 0
            if present and start is None:
                start = index
            elif not present and start is not None:
                segments.append((start, index - 1))
                start = None
        for left, right in segments:
            box_w = right - left + 1
            if box_w < _MIN_W:
                continue
            band = sub[:, left : right + 1]
            rows = np.where(band.any(axis=1))[0]
            if rows.size == 0:
                continue
            y1 = y + int(rows.min())
            y2 = y + int(rows.max()) + 1
            if y2 - y1 < _MIN_H:
                continue
            x1 = int(x + left)
            x2 = int(x + right + 1)
            values.append({
                "bbox": [x1, y1, x2, y2],
                "area": int((x2 - x1) * (y2 - y1)),
            })
    values.sort(key=lambda item: (item["bbox"][1], -item["bbox"][0]))
    for index, item in enumerate(values, 1):
        item["id"] = f"sel_{index:03d}"
    return values


def detect_blue_boxes_single(image: np.ndarray) -> list[dict]:
    """Detect Photos blue selection overlays on one RGB frame.

    Parameters
    ----------
    image : HxWx3 uint8 RGB
    Returns
    -------
    list of dicts: id, bbox [x1,y1,x2,y2] (x2/y2 exclusive), area
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    rgb = image.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Single-frame approximation (no original frame available): keep pixels
    # consistent with the overlay family (B high, R suppressed, G not above
    # B) and reject near-gray scan-text dim.
    blue = (
        (b - r > 40)
        & (b - g > -5)
        & (b >= 150)
    )
    chroma = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    blue &= chroma > _GRAY_CHROMA
    return _boxes_from_mask(blue)


def detect_blue_boxes_dual(
    active: np.ndarray, selected: np.ndarray, island_blocks=None
) -> list[dict]:
    """Dual-frame detection (active=scan-text, selected=Ctrl+A).

    Wraps ``photos_capture.detect_selection_boxes`` when available; otherwise
    uses the same blend-match formula inline.
    """
    try:
        from photos_capture import detect_selection_boxes

        blocks = detect_selection_boxes(active, selected, island_blocks=island_blocks)
        values = []
        for index, block in enumerate(blocks, 1):
            x1, y1 = int(block.x), int(block.y)
            x2, y2 = x1 + int(block.width), y1 + int(block.height)
            values.append({
                "id": f"sel_{index:03d}",
                "bbox": [x1, y1, x2, y2],
                "area": int(block.width * block.height),
            })
        return values
    except Exception:
        pass

    if active.shape != selected.shape:
        raise ValueError(f"frame size mismatch {active.shape} vs {selected.shape}")
    active_f = active.astype(np.int16)
    selected_f = selected.astype(np.int16)
    # Compare the two frames: pixels that changed between scan-text and
    # all-selected AND sit in the bright blue overlay family are boxes
    # (same signal as photos_capture.detect_selection_boxes).
    delta = np.max(np.abs(selected_f - active_f), axis=2)
    blue = (
        (selected_f[..., 2] - selected_f[..., 0] > 40)
        & (selected_f[..., 2] - selected_f[..., 1] > -5)
        & (selected_f[..., 2] >= 150)
    )
    mask = (delta > 12) & blue
    return _boxes_from_mask(mask)


def annotate_boxes(image: np.ndarray, boxes: Sequence[dict]) -> np.ndarray:
    """Draw numbered rectangles for human review."""
    canvas = Image.fromarray(np.asarray(image).copy())
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 64, 64),
        (64, 200, 64),
        (255, 160, 40),
        (80, 120, 255),
        (220, 80, 200),
        (40, 200, 200),
    ]
    for index, item in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in item["bbox"])
        color = colors[index % len(colors)]
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=2)
        label = item.get("id", str(index + 1))
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return np.array(canvas)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect Photos blue selection boxes")
    parser.add_argument("--image", required=True, help="selected frame (or sole input)")
    parser.add_argument("--active", default=None, help="optional unselected OCR frame")
    parser.add_argument("--output", default=None, help="output directory")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    out_dir = Path(args.output) if args.output else image_path.parent / "blue_boxes_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_rgb(image_path)
    if args.active:
        boxes = detect_blue_boxes_dual(load_rgb(args.active), selected)
        method = "dual"
    else:
        boxes = detect_blue_boxes_single(selected)
        method = "single"

    annotated = annotate_boxes(selected, boxes)
    Image.fromarray(annotated).save(out_dir / "annotated.png")
    payload = {
        "schema_version": 1,
        "method": method,
        "source": str(image_path.resolve()),
        "image_size": [int(selected.shape[1]), int(selected.shape[0])],
        "box_count": len(boxes),
        "boxes": boxes,
    }
    (out_dir / "boxes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"method={method} boxes={len(boxes)} -> {out_dir / 'annotated.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
