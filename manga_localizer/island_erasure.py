"""Island-restricted erasure with a zero-change guarantee outside islands.

Reuses the audited eraser machinery (glyph segmentation, style decisions,
restricted-ring fill / local FSR inpaint) with island geometry.  The final
composite only copies pixels inside the erasure masks, and two independent
pixel audits run against the erasure union and the island union; any change
outside the islands raises.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Protocol, Tuple

from PIL import Image, ImageChops

from .eraser import (
    ColumnErasureAudit,
    EraseStyleDecision,
    ErasureConfig,
    ErasureResult,
    Inpainter,
    LocalStatisticsStyleClassifier,
    OpenCVFSRInpainter,
    PixelDifferenceAudit,
    RGBColor,
    sample_ring_color,
    _changed_pixel_mask,
    _pixel_difference_audit,
)
from .errors import ErasureError
from .island_contract import IslandDataset
from .island_masks import IslandMaskSet, build_island_masks
from .region_masks import build_erasure_masks


@dataclass(frozen=True)
class IslandErasureResult:
    image: Image.Image
    difference: Image.Image
    changed_mask: Image.Image
    island_masks: IslandMaskSet
    decisions: Dict[str, EraseStyleDecision]
    columns: Tuple[ColumnErasureAudit, ...]
    pixels: PixelDifferenceAudit
    island_pixels: PixelDifferenceAudit
    dropped_tiny: Tuple[str, ...]
    erasure_seconds: float
    pixel_audit_seconds: float


def erase_island_image(
    image: Image.Image,
    dataset: IslandDataset,
    *,
    config: ErasureConfig = ErasureConfig(),
    classifier: Optional[LocalStatisticsStyleClassifier] = None,
    inpainter: Optional[Inpainter] = None,
    min_island_area: float = 0.0,
) -> IslandErasureResult:
    from .island_contract import filter_tiny_islands

    active_dataset, dropped = filter_tiny_islands(dataset, min_island_area)
    if active_dataset is not dataset:
        dataset = active_dataset
    if image.size != dataset.image_size:
        raise ErasureError(
            f"image size {image.size} differs from islands.json {dataset.image_size}"
        )
    if config.style_override not in {"auto", "solid_fill", "inpaint"}:
        raise ErasureError(f"unsupported erase strategy: {config.style_override}")

    started = time.perf_counter()
    source = image.convert("RGB")
    island_masks = build_island_masks(dataset)
    active_classifier = classifier or LocalStatisticsStyleClassifier()
    decisions: Dict[str, EraseStyleDecision] = {}
    for geometry in island_masks.columns:
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
        island_masks,
        {island_id: decision.strategy for island_id, decision in decisions.items()},
        config.mask,
    )
    # Glyph detection dilates the island geometry (up to a few pixels) to
    # catch strokes that touch the boundary.  That dilation must never leak
    # outside the island: every erasure mask is clipped to its island, so the
    # island-exterior zero-change guarantee is structural, not just audited.
    geometry_by_id = {column.audit.column_id: column.image for column in island_masks.columns}
    clipped_masks = []
    combined = Image.new("L", source.size, 0)
    for column_mask in erasure_masks.columns:
        clipped = ImageChops.darker(column_mask.image, geometry_by_id[column_mask.audit.column_id])
        if clipped.histogram()[255] > 0:
            clipped_masks.append((column_mask.audit, clipped))
            combined = ImageChops.lighter(combined, clipped)
    if not clipped_masks:
        raise ErasureError("all island erasure masks are empty after island clipping")
    from .region_masks import ErasureColumnMask, ErasureMaskSet

    erasure_masks = ErasureMaskSet(
        columns=tuple(ErasureColumnMask(audit=audit, image=mask) for audit, mask in clipped_masks),
        combined=combined,
        union_pixel_count=combined.histogram()[255],
    )
    active_inpainter = inpainter or OpenCVFSRInpainter()
    working = source.copy()
    execution: Dict[str, Tuple[str, Optional[RGBColor]]] = {}
    inpaint_groups: Dict[str, Tuple[Inpainter, Image.Image]] = {}
    for column_mask in erasure_masks.columns:
        island_id = column_mask.audit.column_id
        decision = decisions[island_id]
        if decision.strategy == "solid_fill":
            color = sample_ring_color(
                source,
                column_mask.image,
                config.fill_ring_radius,
            )
            working.paste(color, (0, 0, working.width, working.height), column_mask.image)
            execution[island_id] = ("restricted-ring-median-fill-v1", color)
        else:
            backend = active_inpainter
            group = inpaint_groups.get(backend.name)
            if group is None:
                group = (backend, Image.new("L", source.size, 0))
            inpaint_groups[backend.name] = (
                group[0],
                ImageChops.lighter(group[1], column_mask.image),
            )
            execution[island_id] = (backend.name, None)
    for backend, grouped_mask in inpaint_groups.values():
        if grouped_mask.getbbox() is not None:
            working = backend.inpaint(working, grouped_mask, config.inpaint_padding)

    pixel_audit_started = time.perf_counter()
    erased = Image.composite(working, source, erasure_masks.combined)
    difference = ImageChops.difference(source, erased)
    changed_mask = _changed_pixel_mask(difference)
    pixels = _pixel_difference_audit(difference, changed_mask, erasure_masks.combined)
    island_pixels = _pixel_difference_audit(difference, changed_mask, island_masks.combined)
    pixel_audit_seconds = time.perf_counter() - pixel_audit_started
    if not pixels.outside_mask_unchanged:
        raise ErasureError(
            f"eraser changed {pixels.changed_outside_mask_pixels} pixels outside the erasure masks"
        )
    if not island_pixels.outside_mask_unchanged:
        raise ErasureError(
            f"eraser changed {island_pixels.changed_outside_mask_pixels} pixels outside the islands"
        )

    mask_map = {item.audit.column_id: item for item in erasure_masks.columns}
    audits = []
    for geometry in island_masks.columns:
        island_id = geometry.audit.column_id
        column_mask = mask_map[island_id]
        decision = decisions[island_id]
        backend, fill_color = execution[island_id]
        changed = ImageChops.darker(changed_mask, column_mask.image).histogram()[255]
        audits.append(
            ColumnErasureAudit(
                column_id=island_id,
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

    return IslandErasureResult(
        image=erased,
        difference=difference,
        changed_mask=changed_mask,
        island_masks=island_masks,
        decisions=decisions,
        columns=tuple(audits),
        pixels=pixels,
        island_pixels=island_pixels,
        dropped_tiny=tuple(island.id for island in dropped),
        erasure_seconds=time.perf_counter() - started,
        pixel_audit_seconds=pixel_audit_seconds,
    )
