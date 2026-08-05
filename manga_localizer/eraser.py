from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Mapping, Optional, Protocol, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .errors import ErasureError
from .ocr_contract import BBox, OCRContract
from .region_masks import (
    ColumnMask,
    ErasureColumnMask,
    ErasureMaskConfig,
    ErasureMaskSet,
    MaskSet,
    build_column_masks,
    build_erasure_masks,
)


RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class BackgroundMetrics:
    sample_pixels: int
    mean_luminance: float
    luminance_stddev: float
    white_ratio: float
    dark_ratio: float


@dataclass(frozen=True)
class EraseStyleDecision:
    column_id: str
    strategy: str
    classifier: str
    reason: str
    metrics: BackgroundMetrics


class EraseStyleClassifier(Protocol):
    name: str

    def classify(self, image: Image.Image, geometry: ColumnMask) -> EraseStyleDecision:
        ...


class Inpainter(Protocol):
    name: str

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        ...


@dataclass(frozen=True)
class LocalStatisticsStyleClassifier:
    ring_radius: int = 8
    solid_min_luminance: float = 225.0
    solid_min_white_ratio: float = 0.75
    name: str = "local-background-statistics-v1"

    def classify(self, image: Image.Image, geometry: ColumnMask) -> EraseStyleDecision:
        metrics = measure_background(image, geometry.image, self.ring_radius)
        solid = (
            metrics.mean_luminance >= self.solid_min_luminance
            and metrics.white_ratio >= self.solid_min_white_ratio
        )
        strategy = "solid_fill" if solid else "inpaint"
        reason = (
            "bright-low-complexity-background"
            if solid
            else "line-art-tone-or-complex-background"
        )
        return EraseStyleDecision(
            column_id=geometry.audit.column_id,
            strategy=strategy,
            classifier=self.name,
            reason=reason,
            metrics=metrics,
        )


@dataclass(frozen=True)
class DirectionalInpainter:
    name: str = "directional-linear-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        fallback = sample_ring_color(image, mask, max(2, padding))
        filled = _directional_fill(local_image, local_mask, fallback, blend=True)
        local_result = Image.composite(filled, local_image, local_mask)
        result = image.copy()
        result.paste(local_result, box[:2])
        return result


@dataclass(frozen=True)
class NearestBoundaryInpainter:
    name: str = "nearest-boundary-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        fallback = sample_ring_color(image, mask, max(2, padding))
        filled = _directional_fill(local_image, local_mask, fallback, blend=False)
        local_result = Image.composite(filled, local_image, local_mask)
        result = image.copy()
        result.paste(local_result, box[:2])
        return result


@dataclass(frozen=True)
class OpenCVFSRInpainter:
    name: str = "opencv-fsr-fast-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise ErasureError(
                "opencv-contrib-python-headless and numpy are required for FSR inpainting"
            ) from error
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        source = np.asarray(local_image)
        erase_mask = np.asarray(local_mask)
        valid_mask = np.where(erase_mask > 0, 0, 255).astype(np.uint8)
        destination = np.empty_like(source)
        try:
            cv2.xphoto.inpaint(
                source,
                valid_mask,
                destination,
                cv2.xphoto.INPAINT_FSR_FAST,
            )
        except cv2.error as error:
            raise ErasureError(f"OpenCV FSR inpainting failed: {error}") from error
        local_result = Image.fromarray(destination)
        result = image.copy()
        result.paste(local_result, box[:2], local_mask)
        return result


@dataclass(frozen=True)
class ErasureConfig:
    style_override: str = "auto"
    mask: ErasureMaskConfig = ErasureMaskConfig()
    fill_ring_radius: int = 6
    inpaint_padding: int = 8


@dataclass(frozen=True)
class ColumnErasureAudit:
    column_id: str
    unit_id: str
    reading_position: int
    strategy: str
    classifier: str
    classifier_reason: str
    background: BackgroundMetrics
    mask_kind: str
    mask_bbox: BBox
    mask_pixels: int
    geometry_pixels: int
    threshold: Optional[int]
    dilation_pixels: int
    contains_geometry_envelope: bool
    white_outline_pixels: int
    backend: str
    fill_color: Optional[RGBColor]
    changed_pixels: int


@dataclass(frozen=True)
class PixelDifferenceAudit:
    allowed_mask_pixels: int
    changed_pixels: int
    changed_inside_mask_pixels: int
    changed_outside_mask_pixels: int
    unchanged_mask_pixels: int
    outside_mask_unchanged: bool
    difference_bbox: Optional[BBox]
    max_channel_delta: int
    sum_absolute_difference: int


@dataclass(frozen=True)
class ErasureResult:
    image: Image.Image
    difference: Image.Image
    changed_mask: Image.Image
    geometry_masks: MaskSet
    erasure_masks: ErasureMaskSet
    decisions: Mapping[str, EraseStyleDecision]
    columns: Tuple[ColumnErasureAudit, ...]
    pixels: PixelDifferenceAudit
    erasure_seconds: float
    pixel_audit_seconds: float


def erase_image(
    image: Image.Image,
    contract: OCRContract,
    *,
    config: ErasureConfig = ErasureConfig(),
    classifier: Optional[EraseStyleClassifier] = None,
    inpainter: Optional[Inpainter] = None,
) -> ErasureResult:
    erasure_started = time.perf_counter()
    if image.size != contract.image_size:
        raise ErasureError(
            f"image size {image.size} differs from OCR contract {contract.image_size}"
        )
    if config.style_override not in {"auto", "solid_fill", "inpaint"}:
        raise ErasureError(f"unsupported erase strategy: {config.style_override}")
    if config.fill_ring_radius < 1 or config.inpaint_padding < 1:
        raise ErasureError("erasure sampling and inpaint radii must be positive")

    source = image.convert("RGB")
    geometry_masks = build_column_masks(contract)
    active_classifier = classifier or LocalStatisticsStyleClassifier()
    decisions: Dict[str, EraseStyleDecision] = {}
    for geometry in geometry_masks.columns:
        automatic = active_classifier.classify(source, geometry)
        if config.style_override == "auto":
            decision = automatic
        else:
            decision = EraseStyleDecision(
                column_id=automatic.column_id,
                strategy=config.style_override,
                classifier="cli-override",
                reason=f"forced-{config.style_override}",
                metrics=automatic.metrics,
            )
        decisions[decision.column_id] = decision

    erasure_masks = build_erasure_masks(
        source,
        geometry_masks,
        {column_id: decision.strategy for column_id, decision in decisions.items()},
        config.mask,
    )
    active_inpainter = inpainter or OpenCVFSRInpainter()
    working = source.copy()
    execution: Dict[str, Tuple[str, Optional[RGBColor]]] = {}
    inpaint_groups: Dict[str, Tuple[Inpainter, Image.Image]] = {}
    for column_mask in erasure_masks.columns:
        decision = decisions[column_mask.audit.column_id]
        if decision.strategy == "solid_fill":
            color = sample_ring_color(
                source,
                column_mask.image,
                config.fill_ring_radius,
            )
            working.paste(color, (0, 0, working.width, working.height), column_mask.image)
            execution[column_mask.audit.column_id] = ("restricted-ring-median-fill-v1", color)
        else:
            backend = active_inpainter
            group = inpaint_groups.get(backend.name)
            if group is None:
                group = (backend, Image.new("L", source.size, 0))
            inpaint_groups[backend.name] = (
                group[0],
                ImageChops.lighter(group[1], column_mask.image),
            )
            execution[column_mask.audit.column_id] = (backend.name, None)
    for backend, grouped_mask in inpaint_groups.values():
        if grouped_mask.getbbox() is not None:
            working = backend.inpaint(working, grouped_mask, config.inpaint_padding)

    # This final composite is the hard pixel-integrity boundary. Even if a future
    # backend touches a larger crop internally, pixels outside the audited union
    # cannot reach the output.
    # Keep the pixel-integrity audit on its own timer. Erasure and future
    # typesetting work may be expensive, but only this verification budget is
    # used for the one-minute gate.
    pixel_audit_started = time.perf_counter()
    erased = Image.composite(working, source, erasure_masks.combined)
    difference = ImageChops.difference(source, erased)
    changed_mask = _changed_pixel_mask(difference)
    pixels = _pixel_difference_audit(difference, changed_mask, erasure_masks.combined)
    pixel_audit_seconds = time.perf_counter() - pixel_audit_started
    if not pixels.outside_mask_unchanged:
        raise ErasureError(
            f"eraser changed {pixels.changed_outside_mask_pixels} pixels outside the mask"
        )

    mask_map = {item.audit.column_id: item for item in erasure_masks.columns}
    column_audits = []
    for geometry in geometry_masks.columns:
        column_id = geometry.audit.column_id
        column_mask = mask_map[column_id]
        decision = decisions[column_id]
        backend, fill_color = execution[column_id]
        changed = ImageChops.darker(changed_mask, column_mask.image).histogram()[255]
        column_audits.append(
            ColumnErasureAudit(
                column_id=column_id,
                unit_id=column_mask.audit.unit_id,
                reading_position=column_mask.audit.reading_position,
                strategy=decision.strategy,
                classifier=decision.classifier,
                classifier_reason=decision.reason,
                background=decision.metrics,
                mask_kind=column_mask.audit.kind,
                mask_bbox=column_mask.audit.mask_bbox,
                mask_pixels=column_mask.audit.pixel_count,
                geometry_pixels=column_mask.audit.geometry_pixel_count,
                threshold=column_mask.audit.threshold,
                dilation_pixels=column_mask.audit.dilation_pixels,
                contains_geometry_envelope=column_mask.audit.contains_geometry_envelope,
                white_outline_pixels=column_mask.audit.white_outline_pixels,
                backend=backend,
                fill_color=fill_color,
                changed_pixels=changed,
            )
        )
    return ErasureResult(
        image=erased,
        difference=difference,
        changed_mask=changed_mask,
        geometry_masks=geometry_masks,
        erasure_masks=erasure_masks,
        decisions=decisions,
        columns=tuple(column_audits),
        pixels=pixels,
        erasure_seconds=time.perf_counter() - erasure_started,
        pixel_audit_seconds=pixel_audit_seconds,
    )


def measure_background(
    image: Image.Image,
    mask: Image.Image,
    ring_radius: int,
) -> BackgroundMetrics:
    gray = image.convert("L")
    ring = _mask_ring(mask, max(1, ring_radius))
    histogram = gray.histogram(mask=ring)
    sample_pixels = sum(histogram)
    if not sample_pixels:
        return BackgroundMetrics(0, 255.0, 0.0, 1.0, 0.0)
    mean = sum(value * count for value, count in enumerate(histogram)) / sample_pixels
    variance = (
        sum(((value - mean) ** 2) * count for value, count in enumerate(histogram))
        / sample_pixels
    )
    return BackgroundMetrics(
        sample_pixels=sample_pixels,
        mean_luminance=mean,
        luminance_stddev=math.sqrt(variance),
        white_ratio=sum(histogram[240:]) / sample_pixels,
        dark_ratio=sum(histogram[:80]) / sample_pixels,
    )


def sample_ring_color(image: Image.Image, mask: Image.Image, radius: int) -> RGBColor:
    rgb = image.convert("RGB")
    ring = _mask_ring(mask, max(1, radius))
    return tuple(_histogram_median(band.histogram(mask=ring), 255) for band in rgb.split())  # type: ignore[return-value]


def _directional_fill(
    image: Image.Image,
    mask: Image.Image,
    fallback: RGBColor,
    *,
    blend: bool,
) -> Image.Image:
    width, height = image.size
    pixels = _flattened_pixels(image)
    masked = [value > 0 for value in _flattened_pixels(mask)]
    red = [0.0] * len(pixels)
    green = [0.0] * len(pixels)
    blue = [0.0] * len(pixels)
    weights = [0.0] * len(pixels)

    def add(index: int, color: RGBColor, weight: float) -> None:
        red[index] += color[0] * weight
        green[index] += color[1] * weight
        blue[index] += color[2] * weight
        weights[index] += weight

    for y in range(height):
        x = 0
        while x < width:
            index = y * width + x
            if not masked[index]:
                x += 1
                continue
            start = x
            while x < width and masked[y * width + x]:
                x += 1
            end = x
            left = pixels[y * width + start - 1] if start > 0 else None
            right = pixels[y * width + end] if end < width else None
            span = end - start
            weight = 1.0 / max(1, span)
            for offset, current_x in enumerate(range(start, end), 1):
                if left is not None and right is not None:
                    fraction = offset / (span + 1)
                    color = (
                        tuple(
                            round(left[channel] * (1.0 - fraction) + right[channel] * fraction)
                            for channel in range(3)
                        )
                        if blend
                        else left
                        if offset * 2 <= span + 1
                        else right
                    )
                else:
                    color = left or right or fallback
                add(y * width + current_x, color, weight)  # type: ignore[arg-type]

    for x in range(width):
        y = 0
        while y < height:
            index = y * width + x
            if not masked[index]:
                y += 1
                continue
            start = y
            while y < height and masked[y * width + x]:
                y += 1
            end = y
            top = pixels[(start - 1) * width + x] if start > 0 else None
            bottom = pixels[end * width + x] if end < height else None
            span = end - start
            weight = 1.0 / max(1, span)
            for offset, current_y in enumerate(range(start, end), 1):
                if top is not None and bottom is not None:
                    fraction = offset / (span + 1)
                    color = (
                        tuple(
                            round(top[channel] * (1.0 - fraction) + bottom[channel] * fraction)
                            for channel in range(3)
                        )
                        if blend
                        else top
                        if offset * 2 <= span + 1
                        else bottom
                    )
                else:
                    color = top or bottom or fallback
                add(current_y * width + x, color, weight)  # type: ignore[arg-type]

    output = list(pixels)
    for index, is_masked in enumerate(masked):
        if not is_masked:
            continue
        if weights[index]:
            output[index] = (
                round(red[index] / weights[index]),
                round(green[index] / weights[index]),
                round(blue[index] / weights[index]),
            )
        else:
            output[index] = fallback
    result = Image.new("RGB", image.size)
    result.putdata(output)
    return result


def _pixel_difference_audit(
    difference: Image.Image,
    changed_mask: Image.Image,
    allowed_mask: Image.Image,
) -> PixelDifferenceAudit:
    outside = ImageOps.invert(allowed_mask)
    changed_inside = ImageChops.darker(changed_mask, allowed_mask).histogram()[255]
    changed_outside = ImageChops.darker(changed_mask, outside).histogram()[255]
    changed_pixels = changed_mask.histogram()[255]
    allowed_pixels = allowed_mask.histogram()[255]
    bbox = changed_mask.getbbox()
    histograms = [band.histogram() for band in difference.convert("RGB").split()]
    max_delta = max(
        (value for histogram in histograms for value, count in enumerate(histogram) if count),
        default=0,
    )
    absolute_sum = sum(
        value * count for histogram in histograms for value, count in enumerate(histogram)
    )
    return PixelDifferenceAudit(
        allowed_mask_pixels=allowed_pixels,
        changed_pixels=changed_pixels,
        changed_inside_mask_pixels=changed_inside,
        changed_outside_mask_pixels=changed_outside,
        unchanged_mask_pixels=allowed_pixels - changed_inside,
        outside_mask_unchanged=changed_outside == 0,
        difference_bbox=(
            tuple(int(value) for value in bbox) if bbox is not None else None  # type: ignore[arg-type]
        ),
        max_channel_delta=max_delta,
        sum_absolute_difference=absolute_sum,
    )


def _changed_pixel_mask(difference: Image.Image) -> Image.Image:
    bands = [band.point(lambda value: 255 if value else 0, mode="L") for band in difference.split()]
    changed = bands[0]
    for band in bands[1:]:
        changed = ImageChops.lighter(changed, band)
    return changed


def _mask_ring(mask: Image.Image, radius: int) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", mask.size, 0)
    x1, y1, x2, y2 = bbox
    box = _expanded_bbox(bbox, radius, mask.size)
    local = mask.crop(box).filter(ImageFilter.MaxFilter(radius * 2 + 1))
    dilated = Image.new("L", mask.size, 0)
    dilated.paste(local, box[:2])
    return ImageChops.subtract(dilated, mask)


def _expanded_bbox(bbox: BBox, padding: int, size: Tuple[int, int]) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(size[0], x2 + padding),
        min(size[1], y2 + padding),
    )


def _histogram_median(histogram: list[int], default: int) -> int:
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


def _flattened_pixels(image: Image.Image):
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened()) if flattened is not None else list(image.getdata())
