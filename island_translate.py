"""Translate pages from diff-based island masks.

Pipeline: photos_base/photos_ocr diff -> blue-filled island mask -> PaddleOCR
on the masked image -> column clustering -> Sakura full-sentence translation
-> erase + typeset -> translated image + state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from island_extract import parse_color, process_page
from photos_capture import _group_paddle_columns, _run_paddle_whole_page
from windows_ocr import windows_ocr_predictions
import photos_translate as pt


async def translate_page(
    page: str,
    source_image: str,
    capture_page_dir: str | Path,
    output_path: str,
    state_path: str | None,
    fill_color,
    engine: str = "windows-ocr",
    paddle_device: str = "cpu",
) -> dict:
    image_size = Image.open(source_image).size
    timing = {}
    t0 = time.perf_counter()
    result = process_page(capture_page_dir, image_size, timing, fill_color)
    t_mask = time.perf_counter()
    # The OCR input must live in ORIGINAL image space: the display frame is
    # ~0.67x smaller, and Paddle polygons measured there would be treated as
    # source coordinates, scaling every typeset position.  Keep the islands'
    # original pixels on a fill-colored background at full resolution.
    image_arr = np.array(Image.open(source_image).convert("RGB"))
    fill = np.asarray(fill_color, dtype=np.uint8).reshape(1, 1, 3)
    ocr_masked = np.full(image_arr.shape, fill, dtype=np.uint8)
    for island in result["islands"]:
        x1, y1, x2, y2 = island["image_bbox"]
        ocr_masked[y1:y2, x1:x2] = image_arr[y1:y2, x1:x2]
    Image.fromarray(ocr_masked).save(
        os.path.splitext(output_path)[0] + ".ocr-masked.png"
    )
    with tempfile.TemporaryDirectory(prefix="island_translate_") as temporary:
        masked_path = os.path.join(temporary, "masked.png")
        Image.fromarray(ocr_masked).save(masked_path)
        if engine == "windows-ocr":
            predictions = await asyncio.to_thread(windows_ocr_predictions, masked_path)
            if predictions:
                text_source = "windows-ocr-masked"
            else:
                # Stylized SFX pages (31/34) defeat the Windows recognizer;
                # PaddleOCR covers them.
                predictions = _run_paddle_whole_page(masked_path, paddle_device)
                text_source = "paddle-masked"
        else:
            predictions = _run_paddle_whole_page(masked_path, paddle_device)
            text_source = "paddle-masked"
    t_paddle = time.perf_counter()
    columns = _group_paddle_columns(predictions, [], image_size, text_source)
    manifest = {
        "schema_version": 2,
        "backend": "Diff islands + PaddleOCR",
        "source": {
            "path": os.path.abspath(source_image),
            "width": image_size[0],
            "height": image_size[1],
        },
        "capture": {
            "method": "island-diff-mask",
            "fill_color": list(fill_color),
            "dim_fraction": timing.get("dim_fraction"),
            "island_count": timing.get("island_count"),
            "paddle_predictions": len(predictions),
            "timing": {
                "mask_seconds": round(t_mask - t0, 3),
                "paddle_seconds": round(t_paddle - t_mask, 3),
            },
        },
        "columns": columns,
    }
    await pt.translate_manifest(manifest, output_path, state_path)
    t_done = time.perf_counter()
    sentence_count = 0
    if state_path and os.path.isfile(state_path):
        try:
            state = json.load(open(state_path, encoding="utf-8"))
            sentence_count = len(state.get("sentences", []))
        except Exception:
            sentence_count = 0
    return {
        "page": page,
        "columns": len(columns),
        "sentences": sentence_count,
        "engine_used": text_source,
        "mask_seconds": round(t_mask - t0, 2),
        "paddle_seconds": round(t_paddle - t_mask, 2),
        "translate_render_seconds": round(t_done - t_paddle, 2),
        "total_seconds": round(t_done - t0, 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate pages via diff island masks")
    parser.add_argument("--images", required=True, help="source image directory")
    parser.add_argument("--capture", required=True, help="capture dir with photos_base/photos_ocr per page")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--pages", help="comma-separated page stems or numeric ranges")
    parser.add_argument("--fill-color", default="30,144,255", help="mask fill color R,G,B or #RRGGBB")
    parser.add_argument("--engine", choices=["windows-ocr", "paddle"], default="windows-ocr")
    parser.add_argument("--paddle-device", default="cpu")
    args = parser.parse_args(argv)

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

    fill_color = parse_color(args.fill_color)
    images_root = Path(args.images).resolve()
    capture_root = Path(args.capture).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _parse_pages(args.pages)
    image_paths = sorted(
        (
            path for path in images_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and (selected is None or path.stem in selected)
        ),
        key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem.lower(),
    )
    if not image_paths:
        raise SystemExit(f"no images selected in {images_root}")

    start_wall = time.perf_counter()
    rows = []
    for image_path in image_paths:
        page = image_path.stem
        capture_page_dir = capture_root / page
        output_path = str(output_root / f"{page}.translated.png")
        state_path = str(output_root / f"{page}.state.json")
        try:
            row = asyncio.run(
                translate_page(
                    page,
                    str(image_path),
                    capture_page_dir,
                    output_path,
                    state_path,
                    fill_color,
                    args.engine,
                    args.paddle_device,
                )
            )
            row["status"] = "ok"
            rows.append(row)
            print(
                f"page {page}: OK cols={row['columns']} sentences={row['sentences']} "
                f"mask={row['mask_seconds']}s paddle={row['paddle_seconds']}s "
                f"translate+render={row['translate_render_seconds']}s total={row['total_seconds']}s",
                flush=True,
            )
        except Exception as error:
            rows.append({"page": page, "status": "failed", "error": str(error)})
            print(f"page {page}: FAIL {error}", flush=True)

    wall = time.perf_counter() - start_wall
    report = {
        "schema_version": 1,
        "artifact": "island-diff-mask-translate",
        "images": str(images_root),
        "capture": str(capture_root),
        "output": str(output_root),
        "fill_color": list(fill_color),
        "wall_seconds": round(wall, 2),
        "pages": rows,
        "complete": bool(rows) and all(item["status"] == "ok" for item in rows),
    }
    with open(output_root / "batch.report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"wall={wall:.2f}s")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
