from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Mapping, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .errors import ContractValidationError
from .ocr_contract import BBox, OCRContract, Polygon


@dataclass(frozen=True)
class MaskAudit:
    column_id: str
    unit_id: str
    unit_position: int
    reading_position: int
    polygon: Polygon
    mask_bbox: BBox
    pixel_count: int
    overlap_with_previous_pixels: int


@dataclass(frozen=True)
class ColumnMask:
    audit: MaskAudit
    image: Image.Image


@dataclass(frozen=True)
class MaskOverlap:
    first_column_id: str
    second_column_id: str
    first_unit_id: str
    second_unit_id: str
    pixel_count: int

    @property
    def crosses_translation_units(self) -> bool:
        return self.first_unit_id != self.second_unit_id


@dataclass(frozen=True)
class MaskSet:
    columns: Tuple[ColumnMask, ...]
    combined: Image.Image
    overlaps: Tuple[MaskOverlap, ...]
    union_pixel_count: int
    overlapping_pixel_count: int
    cross_unit_overlap_pixel_count: int


@dataclass(frozen=True)
class ErasureMaskConfig:
    glyph_threshold_margin: int = 24
    glyph_dilation: int = 3
    artwork_margin: int = 2
    artwork_search_margin: int = 3
    artwork_dark_threshold: int = 100
    artwork_light_threshold: int = 225
    artwork_outline_contact_radius: int = 5
    artwork_core_contact_radius: int = 7
    artwork_growth_radius: int = 3
    minimum_glyph_pixels: int = 12


@dataclass(frozen=True)
class ErasureMaskAudit:
    column_id: str
    unit_id: str
    reading_position: int
    kind: str
    geometry_bbox: BBox
    mask_bbox: BBox
    geometry_pixel_count: int
    pixel_count: int
    threshold: Optional[int]
    dilation_pixels: int
    contains_geometry_envelope: bool
    white_outline_pixels: int


@dataclass(frozen=True)
class ErasureColumnMask:
    audit: ErasureMaskAudit
    image: Image.Image


@dataclass(frozen=True)
class ErasureMaskSet:
    columns: Tuple[ErasureColumnMask, ...]
    combined: Image.Image
    union_pixel_count: int


def build_column_masks(contract: OCRContract) -> MaskSet:
    """Rasterize one exact polygon envelope per retained OCR column.

    These are B-stage audit masks, not yet the glyph-sensitive masks used by the
    eraser in stage C. They deliberately preserve column identity even when a union
    is also produced for visualization.
    """

    size = contract.image_size
    region_map = {}
    unit_map = {}
    for unit in contract.translation_units:
        for index, region in enumerate(unit.layout_regions):
            region_map[region.column_id] = region
            unit_map[region.column_id] = (unit.id, index)

    combined = Image.new("L", size, 0)
    column_masks = []
    total_overlap = 0
    for reading_position, column_id in enumerate(contract.reading_order):
        if column_id not in region_map:
            raise ContractValidationError(
                f"cannot build mask: {column_id} has no translation layout region"
            )
        region = region_map[column_id]
        mask = rasterize_polygon(size, region.polygon)
        raw_bbox = mask.getbbox()
        if raw_bbox is None:
            raise ContractValidationError(f"polygon mask for {column_id} is empty")
        overlap = ImageChops.darker(combined, mask).histogram()[255]
        combined = ImageChops.lighter(combined, mask)
        unit_id, unit_position = unit_map[column_id]
        audit = MaskAudit(
            column_id=column_id,
            unit_id=unit_id,
            unit_position=unit_position,
            reading_position=reading_position,
            polygon=region.polygon,
            mask_bbox=tuple(int(value) for value in raw_bbox),  # type: ignore[arg-type]
            pixel_count=mask.histogram()[255],
            overlap_with_previous_pixels=overlap,
        )
        column_masks.append(ColumnMask(audit=audit, image=mask))
        total_overlap += overlap
    overlaps = []
    for first_index, first in enumerate(column_masks):
        for second in column_masks[first_index + 1 :]:
            pixel_count = ImageChops.darker(first.image, second.image).histogram()[255]
            if pixel_count:
                overlaps.append(
                    MaskOverlap(
                        first_column_id=first.audit.column_id,
                        second_column_id=second.audit.column_id,
                        first_unit_id=first.audit.unit_id,
                        second_unit_id=second.audit.unit_id,
                        pixel_count=pixel_count,
                    )
                )
    return MaskSet(
        columns=tuple(column_masks),
        combined=combined,
        overlaps=tuple(overlaps),
        union_pixel_count=combined.histogram()[255],
        overlapping_pixel_count=total_overlap,
        cross_unit_overlap_pixel_count=sum(
            item.pixel_count for item in overlaps if item.crosses_translation_units
        ),
    )


def build_erasure_masks(
    image: Image.Image,
    geometry_masks: MaskSet,
    mask_modes: Mapping[str, str],
    config: ErasureMaskConfig = ErasureMaskConfig(),
) -> ErasureMaskSet:
    """Build glyph-sensitive masks while preserving one mask per OCR column.

    Solid backgrounds use a dark-glyph mask so bubble whitespace and borders are
    left alone. Artwork masks retain the full polygon envelope and add a small
    margin so white text outlines are included before local inpainting.
    """

    if image.size != geometry_masks.combined.size:
        raise ContractValidationError("image size differs from OCR mask geometry")
    if config.glyph_dilation < 0 or config.artwork_margin < 0:
        raise ContractValidationError("mask dilation values must be non-negative")
    if config.minimum_glyph_pixels < 1:
        raise ContractValidationError("minimum_glyph_pixels must be positive")

    gray = image.convert("L")
    combined = Image.new("L", image.size, 0)
    columns = []
    for geometry in geometry_masks.columns:
        column_id = geometry.audit.column_id
        mode = mask_modes.get(column_id)
        if mode == "solid_fill":
            mask, threshold, fallback = _dark_glyph_mask(
                gray,
                geometry.image,
                threshold_margin=config.glyph_threshold_margin,
                dilation=config.glyph_dilation,
                minimum_pixels=config.minimum_glyph_pixels,
            )
            kind = "polygon-fallback" if fallback else "dark-glyph-threshold"
            dilation = config.glyph_dilation
            contains_geometry = fallback
        elif mode == "inpaint":
            mask, outline_pixels, fallback = _outlined_artwork_mask(
                gray,
                geometry.image,
                config=config,
            )
            threshold = None
            kind = "polygon-fallback" if fallback else "outlined-glyph-contrast"
            dilation = config.artwork_margin
            contains_geometry = fallback
        else:
            raise ContractValidationError(
                f"missing or unsupported mask mode for {column_id}: {mode!r}"
            )
        raw_bbox = mask.getbbox()
        if raw_bbox is None:
            raise ContractValidationError(f"erasure mask for {column_id} is empty")
        pixel_count = mask.histogram()[255]
        audit = ErasureMaskAudit(
            column_id=column_id,
            unit_id=geometry.audit.unit_id,
            reading_position=geometry.audit.reading_position,
            kind=kind,
            geometry_bbox=geometry.audit.mask_bbox,
            mask_bbox=tuple(int(value) for value in raw_bbox),  # type: ignore[arg-type]
            geometry_pixel_count=geometry.audit.pixel_count,
            pixel_count=pixel_count,
            threshold=threshold,
            dilation_pixels=dilation,
            contains_geometry_envelope=contains_geometry,
            white_outline_pixels=(0 if mode == "solid_fill" else outline_pixels),
        )
        columns.append(ErasureColumnMask(audit=audit, image=mask))
        combined = ImageChops.lighter(combined, mask)
    return ErasureMaskSet(
        columns=tuple(columns),
        combined=combined,
        union_pixel_count=combined.histogram()[255],
    )


def rasterize_polygon(size: Tuple[int, int], polygon: Polygon) -> Image.Image:
    mask = Image.new("L", size, 0)
    points = [(_pixel(point[0]), _pixel(point[1])) for point in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def iter_independent_masks(mask_set: MaskSet) -> Iterator[ColumnMask]:
    yield from mask_set.columns


def _dark_glyph_mask(
    gray: Image.Image,
    geometry_mask: Image.Image,
    *,
    threshold_margin: int,
    dilation: int,
    minimum_pixels: int,
) -> Tuple[Image.Image, int, bool]:
    bbox = geometry_mask.getbbox()
    if bbox is None:
        raise ContractValidationError("cannot segment an empty geometry mask")
    sample_ring = _mask_ring(geometry_mask, max(4, dilation + 2))
    ring_histogram = gray.histogram(mask=sample_ring)
    background = _histogram_median(ring_histogram, default=255)
    threshold = max(0, min(230, background - threshold_margin))
    dark = gray.point(lambda value: 255 if value <= threshold else 0, mode="L")
    search_envelope = _dilate_mask(geometry_mask, dilation)
    glyphs = ImageChops.darker(dark, search_envelope)
    if dilation:
        glyphs = _dilate_mask(glyphs, dilation)
        glyphs = ImageChops.darker(
            glyphs,
            _dilate_mask(geometry_mask, dilation * 2),
        )
    if glyphs.histogram()[255] < minimum_pixels:
        return _dilate_mask(geometry_mask, dilation), threshold, True
    return glyphs, threshold, False


def _outlined_artwork_mask(
    gray: Image.Image,
    geometry_mask: Image.Image,
    *,
    config: ErasureMaskConfig,
) -> Tuple[Image.Image, int, bool]:
    search_envelope = _dilate_mask(geometry_mask, config.artwork_search_margin)
    dark = ImageChops.darker(
        gray.point(
            lambda value: 255 if value <= config.artwork_dark_threshold else 0,
            mode="L",
        ),
        search_envelope,
    )
    bright = ImageChops.darker(
        gray.point(
            lambda value: 255 if value >= config.artwork_light_threshold else 0,
            mode="L",
        ),
        search_envelope,
    )
    outline = ImageChops.darker(
        bright,
        _dilate_mask(dark, config.artwork_outline_contact_radius),
    )
    core = ImageChops.darker(
        dark,
        _dilate_mask(outline, config.artwork_core_contact_radius),
    )
    seed = ImageChops.lighter(outline, core)
    contrast = ImageChops.lighter(dark, bright)
    text = ImageChops.darker(
        contrast,
        _dilate_mask(seed, config.artwork_growth_radius),
    )
    text = ImageChops.lighter(text, seed)
    mask = ImageChops.darker(
        _dilate_mask(text, config.artwork_margin),
        _dilate_mask(
            geometry_mask,
            config.artwork_search_margin + config.artwork_margin + 1,
        ),
    )
    if mask.histogram()[255] < config.minimum_glyph_pixels:
        fallback = _dilate_mask(geometry_mask, config.artwork_margin)
        return fallback, ImageChops.darker(fallback, bright).histogram()[255], True
    return mask, ImageChops.darker(mask, outline).histogram()[255], False


def _mask_ring(mask: Image.Image, radius: int) -> Image.Image:
    return ImageChops.subtract(_dilate_mask(mask, radius), mask)


def _dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    bbox = mask.getbbox()
    if bbox is None:
        return mask.copy()
    x1, y1, x2, y2 = bbox
    crop_box = (
        max(0, x1 - radius),
        max(0, y1 - radius),
        min(mask.width, x2 + radius),
        min(mask.height, y2 + radius),
    )
    local = mask.crop(crop_box).filter(ImageFilter.MaxFilter(radius * 2 + 1))
    result = Image.new("L", mask.size, 0)
    result.paste(local, crop_box[:2])
    return result


def _histogram_median(histogram: list[int], *, default: int) -> int:
    total = sum(histogram)
    if not total:
        return default
    midpoint = (total + 1) // 2
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= midpoint:
            return value
    return default


def _pixel(value: float) -> int:
    return int(math.floor(float(value) + 0.5))
