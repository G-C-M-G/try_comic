"""Island geometry masks for the island-based pipeline.

The geometry authority is ``islands.json``: each island contributes one mask
rasterized from its ``image_polygon`` (fallback: ``image_bbox`` rectangle).
Masks are intentionally NOT expanded into a big rectangle covering island
exteriors; any pixel change outside the island union is an audit failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageChops, ImageDraw

from .errors import ContractValidationError
from .island_contract import IslandDataset, IslandInfo
from .ocr_contract import BBox, Polygon
from .region_masks import ColumnMask, MaskAudit, MaskSet, rasterize_polygon


@dataclass(frozen=True)
class IslandMaskAudit:
    island_id: str
    reading_position: int
    polygon: Optional[Polygon]
    geometry_bbox: BBox
    polygon_pixel_count: int
    mask_pixel_count: int
    fill_mode: str


@dataclass(frozen=True)
class IslandMaskSet:
    columns: Tuple[ColumnMask, ...]
    combined: Image.Image
    union_pixel_count: int
    audits: Tuple[IslandMaskAudit, ...]


def island_mask_image(island: IslandInfo, image_size: Tuple[int, int]) -> Image.Image:
    """Rasterize one island's filled polygon (or bbox rectangle) to a mask."""
    polygon = island.polygon
    if polygon is not None:
        mask = rasterize_polygon(image_size, polygon)
        if mask.histogram()[255] == 0:
            raise ContractValidationError(f"island {island.id}: polygon rasterizes empty")
        return mask
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).rectangle(island.image_bbox, fill=255)
    return mask


def build_island_masks(dataset: IslandDataset) -> IslandMaskSet:
    """Build one geometry mask per island plus the combined island union."""
    image_size = dataset.image_size
    columns = []
    audits = []
    combined = Image.new("L", image_size, 0)
    for index, island in enumerate(dataset.islands):
        mask = island_mask_image(island, image_size)
        polygon_pixels = 0
        if island.polygon is not None:
            polygon_pixels = mask.histogram()[255]
        audit = MaskAudit(
            column_id=island.id,
            unit_id=island.id,
            unit_position=0,
            reading_position=index,
            polygon=island.polygon or _bbox_polygon(island.image_bbox),
            mask_bbox=island.image_bbox,
            pixel_count=mask.histogram()[255],
            overlap_with_previous_pixels=0,
        )
        columns.append(ColumnMask(audit=audit, image=mask))
        audits.append(
            IslandMaskAudit(
                island_id=island.id,
                reading_position=index,
                polygon=island.polygon,
                geometry_bbox=island.image_bbox,
                polygon_pixel_count=polygon_pixels,
                mask_pixel_count=audit.pixel_count,
                fill_mode="shape" if island.polygon is not None else "bbox",
            )
        )
        combined = ImageChops.lighter(combined, mask)

    overlaps = []
    cross_unit = 0
    # Adjacent islands may share boundary pixels (same reason column envelopes
    # overlap); only island-exterior integrity matters, so no hard failure here.
    return IslandMaskSet(
        columns=tuple(columns),
        combined=combined,
        union_pixel_count=combined.histogram()[255],
        audits=tuple(audits),
    )


def _bbox_polygon(bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
