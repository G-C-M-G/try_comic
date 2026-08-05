"""Island dry-run annotation image.

Draws per-island polygon outlines, semi-transparent mask tints, id labels,
the hard island bbox boundary, and (optionally) OCR line boxes with their
recognized text on top of the original image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .island_contract import IslandDataset, IslandInfo
from .island_masks import build_island_masks
from .ocr_contract import BBox, Polygon


@dataclass(frozen=True)
class OcrLine:
    polygon: Polygon
    text: str
    confidence: Optional[float] = None


_ISLAND_COLORS = (
    (255, 51, 51, 90),
    (255, 128, 0, 90),
    (255, 204, 0, 90),
    (102, 255, 51, 90),
    (51, 204, 255, 90),
    (51, 102, 255, 90),
    (204, 51, 255, 90),
    (255, 51, 153, 90),
)


def _default_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _integer_points(polygon: Polygon) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in polygon]


def render_island_dry_run(
    image: Image.Image,
    dataset: IslandDataset,
    *,
    ocr_lines: Optional[Sequence[OcrLine]] = None,
    max_layout_overflow: int = 0,
    stroke_color: Optional[tuple[int, int, int]] = (255, 215, 0),
    stroke_width: int = 6,
) -> Image.Image:
    """Draw the island review diagram on a copy of *image*.

    ``stroke_color``/``stroke_width`` draw the default thick yellow outline
    on every island bbox edge (highlight-island default style); pass
    ``stroke_color=None`` to disable.
    """
    canvas = image.convert("RGB").copy()
    mask_set = build_island_masks(dataset)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _default_font(18)
    small_font = _default_font(13)

    color_lookup = {}
    for index, island in enumerate(dataset.islands):
        color = _ISLAND_COLORS[index % len(_ISLAND_COLORS)]
        polygon = island.polygon
        color_lookup[island.id] = color[:3]
        if polygon is not None:
            draw.polygon(_integer_points(polygon), fill=color)
            draw.line(_integer_points(polygon) + [_integer_points(polygon)[0]], fill=color[:3], width=2)
        else:
            x1, y1, x2, y2 = island.image_bbox
            draw.rectangle((x1, y1, x2, y2), fill=color)
            draw.rectangle((x1, y1, x2, y2), outline=color[:3], width=2)

    # Hard boundary = island bbox (0% default).  --max-layout-overflow 5/20 only
    # adds the optional relaxed outline for review; nothing here expands a mask.
    for index, island in enumerate(dataset.islands):
        x1, y1, x2, y2 = island.image_bbox
        if max_layout_overflow > 0:
            pad = round(max((x2 - x1), (y2 - y1)) * max_layout_overflow / 100)
            draw.rectangle(
                (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad),
                outline=(255, 0, 0),
                width=1,
            )
        if stroke_color is not None:
            draw.rectangle((x1, y1, x2, y2), outline=stroke_color, width=stroke_width)
        draw.text(
            (x1, max(0, y1 - 20)),
            f"{index + 1:02d} {island.id}",
            fill=color_lookup[island.id],
            font=font,
        )

    if ocr_lines:
        for line in ocr_lines:
            points = _integer_points(line.polygon)
            draw.line(points + [points[0]], fill=(0, 102, 255), width=2)
            text = str(line.text or "").strip()
            if text:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                draw.text((min(xs), max(0, min(ys) - 15)), text, fill=(0, 102, 255), font=small_font)

    legend = "tint=island mask, colored line=polygon, red rect=relaxed boundary, blue=OCR line"
    draw.text((8, 8), legend, fill=(0, 0, 0), font=small_font)
    return canvas
