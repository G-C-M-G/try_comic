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


def process_page(
    page_dir: str | Path,
    image_size,
    timing: dict,
    fill_color=(0, 0, 0),
    contours: bool = False,
) -> dict:
    """Extract islands for one page; returns a report dict.

    ``contours`` (default off) additionally extracts each island's precise
    outline polygon (approxPolyDP on the largest external contour) in both
    display and image coordinates.
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

    # Islands are the undimmed regions Photos keeps at original brightness;
    # they are bright white box interiors.  The brightness guard removes dark
    # undimmed specks (antialiasing/art noise) that are not highlight boxes.
    base_bright = np.mean(base, axis=2)
    island_region = (~dimmed) & (base_bright > 145)
    fill = np.asarray(fill_color, dtype=np.float32).reshape(1, 1, 3)
    masked = np.where(island_region[..., None], active, fill).astype(np.uint8)

    bright = (island_region.astype(np.uint8)) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    scale, offset_x, offset_y = _display_to_image_scale(
        (active.shape[1], active.shape[0]), image_size
    )
    islands = []
    for index, (x, y, width, height, area) in enumerate(
        (int(v) for v in item) for item in stats[1:]
    ):
        if area < _MIN_AREA or width < _MIN_WIDTH or height < _MIN_HEIGHT:
            continue
        display_bbox = [x, y, x + width, y + height]
        image_bbox = [
            round((x - offset_x) / scale),
            round((y - offset_y) / scale),
            round((x + width - offset_x) / scale),
            round((y + height - offset_y) / scale),
        ]
        image_bbox[0] = max(0, min(image_size[0] - 1, image_bbox[0]))
        image_bbox[2] = max(image_bbox[0] + 1, min(image_size[0], image_bbox[2]))
        image_bbox[1] = max(0, min(image_size[1] - 1, image_bbox[1]))
        image_bbox[3] = max(image_bbox[1] + 1, min(image_size[1], image_bbox[3]))
        island = {
            "id": f"island_{len(islands) + 1:03d}",
            "display_bbox": display_bbox,
            "display_center": [x + width // 2, y + height // 2],
            "display_area": area,
            "image_bbox": image_bbox,
            "image_center": [
                (image_bbox[0] + image_bbox[2]) // 2,
                (image_bbox[1] + image_bbox[3]) // 2,
            ],
            "image_area": round(area / (scale * scale), 1),
        }
        if contours:
            component = (labels == index + 1).astype(np.uint8) * 255
            contours_found, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_found:
                contour = max(contours_found, key=cv2.contourArea)
                approx = cv2.approxPolyDP(contour, 2.0, True)
                display_polygon = approx.reshape(-1, 2).tolist()
                island["display_polygon"] = display_polygon
                island["image_polygon"] = [
                    [
                        round((px - offset_x) / scale),
                        round((py - offset_y) / scale),
                    ]
                    for px, py in display_polygon
                ]
        islands.append(island)

    elapsed = time.perf_counter() - t0
    timing["dim_fraction"] = round(dim_fraction, 4)
    timing["status"] = status
    timing["island_count"] = len(islands)
    timing["processing_seconds"] = round(elapsed, 3)
    timing["memory_rss_mb"] = round(_rss_mb(), 2)
    return {
        "masked": masked,
        "island_mask": island_region,
        "islands": islands,
        "timing": timing,
        "fill_color": fill_color,
    }


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
            result = process_page(page_dir, image_size, timing, fill_color, args.contours)
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
                        "islands": result["islands"],
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
