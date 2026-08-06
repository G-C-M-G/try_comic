"""Build the handoff island test dataset.

For every page this produces, from existing captures (no UI, no re-capture):

* ``filled_black.png`` / ``filled_blue.png`` - island mask with the background
  filled black / blue and the highlight islands kept at original pixels.
  Default: simple bounding-box fills.  With ``--contours`` (default off) the
  fills are shape-faithful: the non-background pixels are the exact island
  contours (the precise island-outline annotation feature);
* ``islands.json`` - per-island bbox / center / area (display + image coords),
  plus outline polygons when ``--contours`` is enabled;
* ``original.jpg`` - the source image;
* ``photos_ocr.png`` - the Photos scan-text overlay frame (reference);
* ``islands_annotated.png`` - the original image with each island outline
  drawn (only with ``--contours``).

The dataset is the input contract for the next project: translation text must
stay inside the islands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import cv2
from PIL import Image, ImageDraw

from island_extract import (
    extract_islands,
    load_selection_blocks,
    parse_color,
    process_page,
)

_ANNOTATION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#808000", "#000075",
]


def _parse_pages(value: str | None):
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


def _filled_bbox(source_image: str, islands: list[dict], fill_color) -> np.ndarray:
    """Simple full-resolution fill: each island's bounding box keeps pixels."""
    image = np.array(Image.open(source_image).convert("RGB"))
    fill = np.asarray(fill_color, dtype=np.uint8).reshape(1, 1, 3)
    filled = np.full(image.shape, fill, dtype=np.uint8)
    for island in islands:
        x1, y1, x2, y2 = island["image_bbox"]
        filled[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return filled


def _filled_shape(
    source_image: str,
    island_mask_display: np.ndarray,
    image_size,
    fill_color,
) -> np.ndarray:
    """PRECISE contour-annotation fill (--contours, default off).

    The non-background pixels ARE the exact island shapes: the undimmed
    highlight regions Photos keeps at original brightness, scaled from the
    display frame to the source image with nearest-neighbour so the outline
    stays pixel-faithful.  This is the "精准标注高亮岛轮廓" feature.
    """
    image = np.array(Image.open(source_image).convert("RGB"))
    fill = np.asarray(fill_color, dtype=np.uint8).reshape(1, 1, 3)
    display_mask = island_mask_display.astype(np.uint8) * 255
    full_mask = (
        cv2.resize(
            display_mask,
            (int(image_size[0]), int(image_size[1])),
            interpolation=cv2.INTER_NEAREST,
        )
        > 127
    )
    return np.where(full_mask[..., None], image, fill).astype(np.uint8)


def _annotate(source_image: str, islands: list[dict], output_path: str) -> None:
    canvas = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, island in enumerate(islands):
        polygon = island.get("image_polygon")
        if not polygon or len(polygon) < 3:
            continue
        color = _ANNOTATION_COLORS[index % len(_ANNOTATION_COLORS)]
        draw.polygon(polygon, outline=color, width=2)
        x, y = polygon[0]
        draw.text((x, max(0, y - 14)), island["id"], fill=color)
    canvas.save(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build handoff island test dataset")
    parser.add_argument("--images", required=True, help="source image directory")
    parser.add_argument("--capture", required=True, help="capture dir with photos_base/photos_ocr per page")
    parser.add_argument("--output", required=True, help="dataset output directory")
    parser.add_argument("--pages", help="comma-separated page stems or numeric ranges")
    parser.add_argument("--fill-black", default="0,0,0", help="black fill color")
    parser.add_argument("--fill-blue", default="30,144,255", help="blue fill color")
    parser.add_argument(
        "--contours",
        action="store_true",
        help="precise island contour annotation (default off): shape-faithful "
        "filled masks + outline polygons + islands_annotated.png",
    )
    parser.add_argument(
        "--min-island-area",
        type=float,
        default=800.0,
        help="drop islands below this image-area (px^2, default 800)",
    )
    args = parser.parse_args(argv)

    fill_black = parse_color(args.fill_black)
    fill_blue = parse_color(args.fill_blue)
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
    start_wall = time.perf_counter()
    rows = []
    for image_path in image_paths:
        page = image_path.stem
        capture_page_dir = capture_root / page
        out_page = output_root / page
        row = {"page": page, "status": "failed", "error": ""}
        try:
            image_size = Image.open(image_path).size
            timing = {}
            selection_blocks = load_selection_blocks(capture_page_dir)
            result = extract_islands(
                capture_page_dir,
                image_size,
                timing,
                contours=args.contours,
                min_island_area=args.min_island_area,
                selection_blocks=selection_blocks,
            )
            islands = result["islands"]
            island_mask = result["island_mask"]
            out_page.mkdir(parents=True, exist_ok=True)
            if args.contours:
                # Precise contour-annotation fills (default off): the
                # non-background pixels are the exact island shapes.
                Image.fromarray(
                    _filled_shape(str(image_path), island_mask, image_size, fill_black)
                ).save(out_page / "filled_black.png")
                Image.fromarray(
                    _filled_shape(str(image_path), island_mask, image_size, fill_blue)
                ).save(out_page / "filled_blue.png")
            else:
                # Default: simple bounding-box fills.
                Image.fromarray(_filled_bbox(str(image_path), islands, fill_black)).save(
                    out_page / "filled_black.png"
                )
                Image.fromarray(_filled_bbox(str(image_path), islands, fill_blue)).save(
                    out_page / "filled_blue.png"
                )
            shutil.copyfile(image_path, out_page / ("original" + image_path.suffix.lower()))
            overlay = capture_page_dir / "photos_ocr.png"
            if overlay.is_file():
                shutil.copyfile(overlay, out_page / "photos_ocr.png")
            selected = capture_page_dir / "photos_selected.png"
            if selected.is_file():
                shutil.copyfile(selected, out_page / "photos_selected.png")
            selection_json = capture_page_dir / "capture.json"
            if selection_json.is_file():
                try:
                    manifest = json.loads(selection_json.read_text(encoding="utf-8-sig"))
                    capture = manifest.get("capture", {})
                    selection_blocks_out = {
                        "schema_version": 1,
                        "page": page,
                        "image_size": list(image_size),
                        "selection_blocks": capture.get("selection_blocks") or [],
                        "selection_groups": capture.get("selection_groups") or [],
                        "selection_boxes_status": capture.get("selection_boxes_status"),
                    }
                    with open(out_page / "selection_blocks.json", "w", encoding="utf-8") as handle:
                        json.dump(selection_blocks_out, handle, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            with open(out_page / "islands.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "page": page,
                        "image_size": list(image_size),
                        "fill_black": list(fill_black),
                        "fill_blue": list(fill_blue),
                        "contours": args.contours,
                        "fill_mode": "shape" if args.contours else "bbox",
                        "min_island_area": args.min_island_area,
                        "islands": islands,
                        "dropped_tiny": result["dropped_tiny"],
                        "splits": result["splits"],
                        "timing": result["timing"],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            if args.contours:
                _annotate(str(image_path), islands, str(out_page / "islands_annotated.png"))
            row.update({
                "status": "ok",
                "island_count": len(islands),
                "dim_fraction": result["timing"]["dim_fraction"],
                "dropped_tiny_count": result["timing"]["dropped_tiny_count"],
                "split_count": result["timing"]["split_count"],
                "contours": args.contours,
            })
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)

    wall = time.perf_counter() - start_wall
    summary = {
        "schema_version": 1,
        "artifact": "island-handoff-dataset",
        "images": str(images_root),
        "capture": str(capture_root),
        "output": str(output_root),
        "fill_black": list(fill_black),
        "fill_blue": list(fill_blue),
        "contours": args.contours,
        "fill_mode": "shape" if args.contours else "bbox",
        "wall_seconds": round(wall, 2),
        "pages": rows,
        "complete": bool(rows) and all(item["status"] == "ok" for item in rows),
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for row in rows:
        if row["status"] == "ok":
            print(
                f"page {row['page']}: ok islands={row['island_count']} "
                f"dim={row['dim_fraction']:.2f} dropped={row['dropped_tiny_count']} "
                f"splits={row['split_count']} contours={row['contours']}"
            )
        else:
            print(f"page {row['page']}: FAIL {row['error']}")
    print(f"wall={wall:.2f}s output={output_root}")
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
