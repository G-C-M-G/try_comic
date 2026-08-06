"""Background diff-based highlight island extraction.

Compares ``photos_base.png`` (original, no overlay) with ``photos_ocr.png``
(scan-text enabled): pixels darkened relative to the original are the gray
mask; the undimmed regions are Photos' highlighted islands.  Every dimmed
pixel is blackened, and each island's bounding box / center / area is recorded
for later text-to-box association work.  No UI, no re-capture.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

_DIM_DELTA = 10.0
_DIM_SIGNED = -4.0
_OVERLAY_FRACTION = 0.15
_MIN_AREA = 40
_MIN_WIDTH = 5
_MIN_HEIGHT = 9


def parse_color(value: str) -> tuple[int, int, int]:
    """Parse ``R,G,B`` or ``#RRGGBB`` into an RGB tuple."""
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise ValueError(f"invalid hex color {value!r}")
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"invalid color {value!r}; use R,G,B or #RRGGBB")
    color = tuple(int(part) for part in parts)
    if not all(0 <= channel <= 255 for channel in color):
        raise ValueError(f"color channels must be 0-255: {value!r}")
    return color


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024.0 ** 2
    except Exception:
        pass
    try:
        kernel32 = ctypes.windll.kernel32

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = kernel32.GetCurrentProcess()
        if kernel32.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024.0 ** 2
    except Exception:
        pass
    return 0.0


def _peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024.0 ** 2 if hasattr(
            psutil.Process().memory_info(), "peak_wset"
        ) else psutil.Process().memory_info().rss / 1024.0 ** 2
    except Exception:
        pass
    try:
        kernel32 = ctypes.windll.kernel32

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        if kernel32.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 1024.0 ** 2
    except Exception:
        pass
    return 0.0


def _display_to_image_scale(viewport_size, image_size):
    view_w, view_h = viewport_size
    image_w, image_h = image_size
    scale = min(view_w / image_w, view_h / image_h)
    offset_x = (view_w - image_w * scale) / 2
    offset_y = (view_h - image_h * scale) / 2
    return scale, offset_x, offset_y


_SEL_COVERAGE = 0.5     # fraction of a selection box inside an island bbox
_Y_LAYER_GAP = 8        # y gap below which selection boxes are one layer
_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _selection_in_island(island_bbox, box) -> bool:
    """Whether most of a selection box lies inside the island bbox.

    The island bbox is the whole highlight region (it can be much larger
    than one OCR block), so a plain IoU would never reach a useful
    threshold; the box-side coverage is the meaningful overlap measure.
    """
    ax1, ay1, ax2, ay2 = island_bbox
    bx1, by1, bx2, by2 = box
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    box_area = (bx2 - bx1) * (by2 - by1)
    return box_area > 0 and inter / box_area > _SEL_COVERAGE


def _cluster_by_y_layers(
    boxes: Sequence[tuple[int, int, int, int]], y_gap: int = _Y_LAYER_GAP
) -> list[list[tuple[int, int, int, int]]]:
    """Cluster boxes by y overlap (gap within *y_gap* = same layer)."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            _, ay1, _, ay2 = boxes[i]
            _, by1, _, by2 = boxes[j]
            if min(ay2, by2) - max(ay1, by1) >= -y_gap:
                union(i, j)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(find(index), []).append(box)
    return [members for members in groups.values()]


def _merge_contained_islands(islands: list[dict]) -> list[dict]:
    """Drop islands whose bbox is fully inside another island's bbox.

    Glyph-aware expansion adds text pixels to the mask; disconnected glyph
    strokes can become tiny standalone components that lie inside the box
    of their owning island (e.g. a column of dots below the box).  Those
    fragments belong to the containing island and must not survive as
    separate islands.  A real neighbouring island only partially overlaps
    (its bbox sticks out) and stays independent.
    """
    kept: list[dict] = []
    for island in islands:
        box = island["display_bbox"]
        swallowed = any(
            other is not island
            and other["display_bbox"][0] <= box[0]
            and other["display_bbox"][1] <= box[1]
            and other["display_bbox"][2] >= box[2]
            and other["display_bbox"][3] >= box[3]
            for other in islands
        )
        if not swallowed:
            kept.append(island)
    return kept


def extract_islands(
    page_dir: str | Path,
    image_size,
    timing: dict,
    contours: bool = False,
    min_island_area: float = 800.0,
    selection_blocks: Sequence = None,
) -> dict:
    """Diff-based island extraction with quality refinements.

    Refinements (all optional, activated by default):
    * dark-panel repair: the undimmed interior of a bright seed box belongs
      to the island even when the panel itself is dark;
    * glyph-aware bbox: each island bbox expands to cover its own text
      pixels (local-contrast mask, nearest-island claim via distance
      transform) so bboxes stop truncating glyphs;
    * selection-box splitting: when the Ctrl+A blue selection boxes inside
      an island form several y layers, the island is split into one island
      per layer (Photos merges vertically separate text blocks into one
      highlight box);
    * micro-island filter: islands below *min_island_area* (image px^2) are
      dropped and reported.

    ``selection_blocks`` is a sequence of display-space ``OverlayBlock``
    boxes (Photos' atomic OCR units); it is optional.
    """
    page_dir = Path(page_dir)
    t0 = time.perf_counter()
    base = np.array(
        Image.open(page_dir / "photos_base.png").convert("RGB")
    ).astype(np.float32)
    active = np.array(
        Image.open(page_dir / "photos_ocr.png").convert("RGB")
    ).astype(np.float32)
    if base.shape != active.shape:
        raise ValueError(f"{page_dir.name}: base/active frame sizes differ")
    delta = np.mean(np.abs(active - base), axis=2)
    signed = np.mean(active - base, axis=2)
    dimmed = (delta > _DIM_DELTA) & (signed < _DIM_SIGNED)
    dim_fraction = float(np.mean(dimmed))
    status = "ok" if dim_fraction >= _OVERLAY_FRACTION else "no-overlay"

    base_bright = np.mean(base, axis=2)
    # Bright seeds: undimmed and bright (box outlines, light boxes).  Every
    # seed's undimmed box interior is then claimed, repairing dark panels
    # whose interior fails the absolute brightness guard.  Page-scale
    # undimmed bright areas (window chrome, page borders) are not highlight
    # boxes: their seed bbox is page-sized and skipped.
    page_area = dimmed.shape[0] * dimmed.shape[1]
    seed = ((~dimmed) & (base_bright > 145)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _K3)
    _, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed)
    island_region = np.zeros(dimmed.shape, dtype=bool)
    for index in range(1, seed_stats.shape[0]):
        x, y, w, h, area = seed_stats[index]
        # Seeds are anchors only: thin box outlines are valid, so only the
        # area gate applies here; size gates run on the final islands.
        if area < _MIN_AREA:
            continue
        if w * h > page_area * 0.10:
            continue
        island_region[y:y + h, x:x + w] |= ~dimmed[y:y + h, x:x + w]
    mask8 = (island_region.astype(np.uint8)) * 255
    mask8 = cv2.morphologyEx(mask8, cv2.MORPH_CLOSE, _K3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask8)

    # Component bboxes are the island geometry: Photos' highlight boxes are
    # the ground truth boundary and are NOT expanded.  Truncated glyphs at
    # the box edge are handled downstream by the REVIEW path, which uses the
    # manga reading directly (lazy switch, manga is the text authority).
    islands: list[dict] = []
    for index in range(1, n):
        x, y, w, h, area = stats[index]
        if area < _MIN_AREA or w < _MIN_WIDTH or h < _MIN_HEIGHT:
            continue
        islands.append({
            "display_bbox": [int(x), int(y), int(x + w), int(y + h)],
            "component_index": index,
        })
    islands = _merge_contained_islands(islands)
    dropped_tiny: list[dict] = []
    splits: list[dict] = []
    scale, offset_x, offset_y = _display_to_image_scale(
        (active.shape[1], active.shape[0]), image_size
    )

    # Blue-box driven islands: Photos' atomic OCR selection boxes are the
    # ground truth for text positions (the diff mask alone misses text in
    # large undimmed regions, e.g. afterword pages).  Every box is padded
    # 2px, aggregated into sentence-level groups, and each group becomes an
    # island (bbox = padded box union; mask = undimmed pixels inside the
    # group).  Diff islands that no group covers stay as fallback
    # (artwork / SFX pages where Photos selected nothing).
    if selection_blocks:
        from photos_capture import OverlayBlock, aggregate_selection_boxes

        # page_scale selections (Photos selected the whole page, e.g. a
        # cover) are not atomic text units: they must not drive islands.
        atomic = [
            block for block in selection_blocks
            if not getattr(block, "page_scale", False)
        ]
        padded = [
            OverlayBlock(
                max(0, block.x - 2),
                max(0, block.y - 2),
                block.width + 4,
                block.height + 4,
            )
            for block in atomic
        ]
        groups = aggregate_selection_boxes(padded, x_gap=24, y_pad=8)
        group_boxes = [
            [
                max(0, int(x1)),
                max(0, int(y1)),
                min(active.shape[1], int(x2)),
                min(active.shape[0], int(y2)),
            ]
            for (x1, y1, x2, y2), _ in groups
        ]
        final: list[dict] = []
        for index, gbox in enumerate(group_boxes, 1):
            gx1, gy1, gx2, gy2 = gbox
            undim = ~dimmed[gy1:gy2, gx1:gx2]
            sub_area = float(undim.sum()) / (scale * scale)
            if sub_area < min_island_area:
                dropped_tiny.append({
                    "display_bbox": list(gbox),
                    "image_area": round(sub_area, 1),
                    "reason": f"selection group area {sub_area:.1f} < {min_island_area}",
                })
                continue
            # group pixels are part of the island mask (filled masks keep
            # the original pixels even when the diff seed was filtered)
            island_region[gy1:gy2, gx1:gx2] |= undim
            final.append({
                "display_bbox": list(gbox),
                "component_index": index,
                "from_selection_group": True,
            })
        # Diff islands that overlap any selection group are replaced by the
        # group islands (the blue boxes are the text ground truth; the diff
        # island often spans several boxes as one large region).  Diff
        # islands without any blue-box overlap stay as fallback (artwork /
        # SFX pages where Photos selected nothing).
        for island in islands:
            box = island["display_bbox"]
            bx1, by1, bx2, by2 = box
            overlapping = any(
                bx1 < gx2 and gx1 < bx2 and by1 < gy2 and gy1 < by2
                for gx1, gy1, gx2, gy2 in group_boxes
            )
            if overlapping:
                continue
            final.append(island)
        islands = sorted(
            final,
            key=lambda item: (item["display_bbox"][1], -item["display_bbox"][2]),
        )
        splits.append({
            "selection_groups": len(groups),
            "group_islands": sum(
                1 for item in final if item.get("from_selection_group")
            ),
            "fallback_diff_islands": sum(
                1 for item in final if not item.get("from_selection_group")
            ),
        })

    # Build the final island records (display + image coordinates).
    completed: list[dict] = []
    for island in islands:
        x, y, x2, y2 = (int(value) for value in island["display_bbox"])
        width, height = x2 - x, y2 - y
        area = int(island_region[y:y2, x:x2].sum())
        if area < _MIN_AREA or width < _MIN_WIDTH or height < _MIN_HEIGHT:
            dropped_tiny.append({
                "display_bbox": [x, y, x2, y2],
                "image_area": round(area / (scale * scale), 1),
                "reason": f"tiny display area {area} < {_MIN_AREA}",
            })
            continue
        image_area = area / (scale * scale)
        if image_area < min_island_area:
            dropped_tiny.append({
                "display_bbox": [x, y, x2, y2],
                "image_area": round(image_area, 1),
                "reason": f"image_area {image_area:.1f} < min_island_area={min_island_area}",
            })
            continue
        image_bbox = [
            round((x - offset_x) / scale),
            round((y - offset_y) / scale),
            round((x2 - offset_x) / scale),
            round((y2 - offset_y) / scale),
        ]
        image_bbox[0] = max(0, min(image_size[0] - 1, image_bbox[0]))
        image_bbox[2] = max(image_bbox[0] + 1, min(image_size[0], image_bbox[2]))
        image_bbox[1] = max(0, min(image_size[1] - 1, image_bbox[1]))
        image_bbox[3] = max(image_bbox[1] + 1, min(image_size[1], image_bbox[3]))
        record = {
            "id": f"island_{len(completed) + 1:03d}",
            "display_bbox": [x, y, x2, y2],
            "display_center": [x + width // 2, y + height // 2],
            "display_area": area,
            "image_bbox": image_bbox,
            "image_center": [
                (image_bbox[0] + image_bbox[2]) // 2,
                (image_bbox[1] + image_bbox[3]) // 2,
            ],
            "image_area": round(area / (scale * scale), 1),
        }
        if island.get("from_selection_group"):
            record["from_selection_group"] = True
        if contours:
            component = (island_region[y:y2, x:x2]).astype(np.uint8) * 255
            contours_found, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_found:
                contour = max(contours_found, key=cv2.contourArea)
                approx = cv2.approxPolyDP(contour, 2.0, True)
                display_polygon = [
                    [
                        min(active.shape[1], max(0, px + x)),
                        min(active.shape[0], max(0, py + y)),
                    ]
                    for px, py in approx.reshape(-1, 2).tolist()
                ]
                record["display_polygon"] = display_polygon
                record["image_polygon"] = [
                    [
                        min(image_size[0], max(0, round((px - offset_x) / scale))),
                        min(image_size[1], max(0, round((py - offset_y) / scale))),
                    ]
                    for px, py in display_polygon
                ]
        completed.append(record)
    islands = completed

    elapsed = time.perf_counter() - t0
    timing["dim_fraction"] = round(dim_fraction, 4)
    timing["status"] = status
    timing["island_count"] = len(islands)
    timing["dropped_tiny_count"] = len(dropped_tiny)
    timing["split_count"] = len(splits)
    timing["processing_seconds"] = round(elapsed, 3)
    timing["memory_rss_mb"] = round(_rss_mb(), 2)
    return {
        "masked": None,
        "island_mask": island_region,
        "islands": islands,
        "timing": timing,
        "dropped_tiny": dropped_tiny,
        "splits": splits,
    }


def build_fallback_dataset(
    original_path: str | Path,
    page_dir: str | Path,
    image_size,
    min_island_area: float = 800.0,
    device: str = "cpu",
) -> dict:
    """Build a minimal islands.json for a failed-capture page.

    Text is located headlessly: winrt line boxes first, Paddle whole-page
    boxes as a fallback.  Boxes are aggregated into islands (the same
    sentence grouping used for Photos blue boxes) and ``islands.json`` plus
    a copy of the original image are written.  No filled masks are produced
    -- the erasure/typeset path only needs islands.json + original.jpg.
    """
    from photos_capture import OverlayBlock, aggregate_selection_boxes

    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    image_w, image_h = image_size

    boxes: list[OverlayBlock] = []
    try:
        from windows_ocr import windows_ocr_predictions

        predictions = windows_ocr_predictions(str(original_path)) or []
        for prediction in predictions:
            points = np.asarray(
                prediction.get("polygon") or [], dtype=np.float64
            ).reshape(-1, 2)
            if points.size == 0:
                continue
            x1, y1 = int(points[:, 0].min()), int(points[:, 1].min())
            x2, y2 = int(points[:, 0].max()), int(points[:, 1].max())
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                boxes.append(OverlayBlock(x1, y1, x2 - x1, y2 - y1))
    except Exception:
        boxes = []

    if not boxes:
        # Paddle whole-page boxes as the fallback locator
        import subprocess
        import tempfile

        root = Path(__file__).resolve().parent
        paddle_python = root / "paddle_venv" / "Scripts" / "python.exe"
        worker = root / "paddle_ocr_worker.py"
        if paddle_python.is_file() and worker.is_file():
            with tempfile.TemporaryDirectory(prefix="fallback_paddle_") as temporary:
                output = os.path.join(temporary, "paddle.json")
                completed = subprocess.run(
                    [
                        str(paddle_python),
                        str(worker),
                        str(original_path),
                        output,
                        "--device",
                        device,
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=600,
                )
                if completed.returncode == 0 and os.path.isfile(output):
                    payload = json.load(open(output, encoding="utf-8"))
                    pages = payload.get("pages") or []
                    if pages:
                        for prediction in pages[0].get("predictions", []):
                            points = np.asarray(
                                prediction.get("polygon") or [], dtype=np.float64
                            ).reshape(-1, 2)
                            if points.size == 0:
                                continue
                            x1, y1 = int(points[:, 0].min()), int(points[:, 1].min())
                            x2, y2 = int(points[:, 0].max()), int(points[:, 1].max())
                            if x2 - x1 >= 8 and y2 - y1 >= 8:
                                boxes.append(
                                    OverlayBlock(x1, y1, x2 - x1, y2 - y1)
                                )

    islands: list[dict] = []
    for (x1, y1, x2, y2), members in aggregate_selection_boxes(
        boxes, x_gap=24, y_pad=8
    ):
        x1 = max(0, min(image_w, int(x1)))
        y1 = max(0, min(image_h, int(y1)))
        x2 = max(x1 + 1, min(image_w, int(x2)))
        y2 = max(y1 + 1, min(image_h, int(y2)))
        area = float((x2 - x1) * (y2 - y1))
        if area < min_island_area:
            continue
        islands.append({
            "id": f"island_{len(islands) + 1:03d}",
            "display_bbox": [x1, y1, x2, y2],
            "display_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "display_area": int(area),
            "image_bbox": [x1, y1, x2, y2],
            "image_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "image_area": round(area, 1),
        })
    if not islands:
        raise ValueError("fallback locator found no text boxes")

    payload = {
        "schema_version": 1,
        "page": page_dir.name,
        "image_size": [image_w, image_h],
        "contours": False,
        "fill_mode": "bbox",
        "min_island_area": min_island_area,
        "fallback": True,
        "islands": islands,
        "dropped_tiny": [],
        "splits": [],
        "timing": {"fallback": True, "island_count": len(islands)},
    }
    (page_dir / "islands.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(original_path, page_dir / "original.jpg")
    return payload


def process_page(
    page_dir: str | Path,
    image_size,
    timing: dict,
    fill_color=(0, 0, 0),
    contours: bool = False,
    min_island_area: float = 800.0,
    selection_blocks: Sequence = None,
) -> dict:
    """Extract islands for one page; returns a report dict (back-compat).

    ``contours`` (default off) additionally extracts each island's precise
    outline polygon (approxPolyDP on the largest external contour) in both
    display and image coordinates.
    """
    result = extract_islands(
        page_dir,
        image_size,
        timing,
        contours=contours,
        min_island_area=min_island_area,
        selection_blocks=selection_blocks,
    )
    base = np.array(
        Image.open(Path(page_dir) / "photos_base.png").convert("RGB")
    ).astype(np.float32)
    active = np.array(
        Image.open(Path(page_dir) / "photos_ocr.png").convert("RGB")
    ).astype(np.float32)
    fill = np.asarray(fill_color, dtype=np.float32).reshape(1, 1, 3)
    result["masked"] = np.where(
        result["island_mask"][..., None], active, fill
    ).astype(np.uint8)
    result["fill_color"] = fill_color
    return result


def load_selection_blocks(page_dir: str | Path):
    """Load Photos' atomic OCR selection boxes (display space) if present."""
    from photos_capture import OverlayBlock

    manifest_path = Path(page_dir) / "capture.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        blocks = manifest.get("capture", {}).get("selection_blocks") or []
        return [
            OverlayBlock(
                int(box[0]),
                int(box[1]),
                int(box[2]) - int(box[0]),
                int(box[3]) - int(box[1]),
            )
            for item in blocks
            if len(item.get("display_bbox") or []) == 4
            for box in [item["display_bbox"]]
        ]
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Background diff-based island extraction")
    parser.add_argument("--input", required=True, help="capture dir with per-page photos_base/photos_ocr")
    parser.add_argument("--output", required=True, help="output dir for masked.png + islands.json")
    parser.add_argument("--pages", help="comma-separated page stems or numeric ranges")
    parser.add_argument(
        "--fill-color",
        default="0,0,0",
        help="background fill color as R,G,B or #RRGGBB (default black)",
    )
    parser.add_argument(
        "--contours",
        action="store_true",
        help="extract precise island outline polygons (default off)",
    )
    parser.add_argument(
        "--min-island-area",
        type=float,
        default=800.0,
        help="drop islands below this image-area (px^2, default 800)",
    )
    args = parser.parse_args(argv)
    fill_color = parse_color(args.fill_color)

    def _parse_pages(value):
        if not value:
            return None
        result = set()
        for raw in value.split(","):
            item = raw.strip()
            if "-" in item:
                left, right = item.split("-", 1)
                result.update(str(number) for number in range(int(left), int(right) + 1))
            else:
                result.add(str(int(item)) if item.isdigit() else item)
        return result

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _parse_pages(args.pages)
    pages = sorted(
        (item for item in input_root.iterdir() if item.is_dir()),
        key=lambda item: int(item.name) if item.name.isdigit() else item.name.lower(),
    )
    start_wall = time.perf_counter()
    memory_before = _rss_mb()
    rows = []
    for page_dir in pages:
        if selected is not None and page_dir.name not in selected:
            continue
        manifest_path = page_dir / "capture.json"
        image_size = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                source = manifest.get("source") or {}
                image_size = (int(source["width"]), int(source["height"]))
            except Exception:
                image_size = None
        row = {"page": page_dir.name, "status": "failed", "error": ""}
        try:
            if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
                raise ValueError("missing valid source dimensions")
            timing = {"memory_before_mb": round(memory_before, 2)}
            selection_blocks = load_selection_blocks(page_dir)
            result = process_page(
                page_dir,
                image_size,
                timing,
                fill_color,
                args.contours,
                min_island_area=args.min_island_area,
                selection_blocks=selection_blocks,
            )
            out_page = output_root / page_dir.name
            out_page.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result["masked"]).save(out_page / "masked.png")
            with open(out_page / "islands.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "page": page_dir.name,
                        "image_size": list(image_size),
                        "fill_color": list(result["fill_color"]),
                        "contours": args.contours,
                        "min_island_area": args.min_island_area,
                        "islands": result["islands"],
                        "dropped_tiny": result["dropped_tiny"],
                        "splits": result["splits"],
                        "timing": result["timing"],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            row.update({
                "status": "ok",
                "dim_fraction": result["timing"]["dim_fraction"],
                "island_count": result["timing"]["island_count"],
                "processing_seconds": result["timing"]["processing_seconds"],
                "memory_rss_mb": result["timing"]["memory_rss_mb"],
            })
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)

    wall_elapsed = time.perf_counter() - start_wall
    summary = {
        "schema_version": 1,
        "artifact": "diff-based-island-extraction",
        "input": str(input_root),
        "output": str(output_root),
        "fill_color": list(fill_color),
        "contours": args.contours,
        "wall_seconds": round(wall_elapsed, 2),
        "memory_before_mb": round(memory_before, 2),
        "memory_peak_mb": round(_peak_rss_mb(), 2),
        "pages": rows,
        "complete": bool(rows) and all(item["status"] == "ok" for item in rows),
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for row in rows:
        status = "ok" if row["status"] == "ok" else f"FAIL {row['error']}"
        extra = (
            f" dim={row['dim_fraction']:.2f} islands={row['island_count']} "
            f"time={row['processing_seconds']}s rss={row['memory_rss_mb']}MB"
            if row["status"] == "ok"
            else ""
        )
        print(f"page {row['page']}: {status}{extra}")
    print(f"wall={wall_elapsed:.2f}s peak_rss={_peak_rss_mb():.1f}MB")
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
