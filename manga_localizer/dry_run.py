from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .layout_bounds import compute_layout_bbox
from .ocr_contract import OCRContract, sha256_file
from .region_masks import MaskSet, build_column_masks
from .review import ReviewApplication


UNIT_COLORS = (
    (0, 166, 118),
    (35, 116, 225),
    (225, 82, 82),
    (146, 92, 190),
    (0, 145, 168),
    (219, 126, 0),
    (96, 125, 45),
)
STATUS_COLORS = {
    "pending": (235, 105, 0),
    "bypassed": (153, 74, 196),
    "approved": (0, 153, 102),
    "corrected": (0, 116, 217),
    "not_required": (0, 153, 102),
}


@dataclass(frozen=True)
class DryRunResult:
    output_path: Path
    audit_path: Path
    output_sha256: str
    status: str
    column_count: int
    pending_count: int
    bypassed_count: int


def render_dry_run(
    image_path: Path,
    application: ReviewApplication,
    output_path: Path,
    *,
    audit_path: Optional[Path] = None,
    max_layout_overflow: int = 10,
) -> DryRunResult:
    contract = application.contract
    image_path = Path(image_path).resolve()
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("dry-run output must use .png for lossless deterministic auditing")
    audit_output = (
        Path(audit_path).resolve()
        if audit_path is not None
        else output.with_name(output.stem + ".audit.json")
    )
    original_hash_before = sha256_file(image_path)
    with Image.open(image_path) as opened:
        base = opened.convert("RGBA")
    masks = build_column_masks(contract)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    unit_indexes = {unit.id: index for index, unit in enumerate(contract.translation_units)}
    review_statuses = {
        item.column_id: item.status for item in contract.manual_review_queue
    }

    for column_mask in masks.columns:
        unit_index = unit_indexes[column_mask.audit.unit_id]
        color = UNIT_COLORS[unit_index % len(UNIT_COLORS)]
        overlay.paste((*color, 58), (0, 0), column_mask.image)
    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)
    font = _audit_font(composed.size)
    column_map = contract.columns_by_id

    column_audits = []
    for column_mask in masks.columns:
        mask_audit = column_mask.audit
        column = column_map[mask_audit.column_id]
        unit_index = unit_indexes[mask_audit.unit_id]
        unit_color = UNIT_COLORS[unit_index % len(UNIT_COLORS)]
        status = review_statuses.get(column.id, column.review_status)
        outline_color = STATUS_COLORS.get(status, unit_color)
        polygon = [_integer_point(point) for point in column.polygon]
        draw.line(polygon + [polygon[0]], fill=(*outline_color, 255), width=3, joint="curve")
        _outlined_rectangle(draw, column.image_bbox, (255, 255, 255, 255))
        _dashed_rectangle(draw, column.layout_bbox_10pct, (255, 203, 0, 255), 5, 4, 2)
        layout_20 = compute_layout_bbox(column.image_bbox, 20, contract.image_size)
        if max_layout_overflow == 20:
            _dashed_rectangle(draw, layout_20, (230, 50, 50, 255), 2, 5, 2)
        label = _label(mask_audit.reading_position, unit_index, status)
        _draw_label(draw, label, mask_audit.mask_bbox, font, outline_color, composed.size)
        column_audits.append(
            {
                "reading_position": mask_audit.reading_position + 1,
                "column_id": column.id,
                "unit_id": mask_audit.unit_id,
                "unit_position": mask_audit.unit_position + 1,
                "source_text": column.text,
                "review_status": status,
                "review_reasons": list(column.review_reasons),
                "polygon": _number_polygon(column.polygon),
                "image_bbox": list(column.image_bbox),
                "layout_bbox_10pct": list(column.layout_bbox_10pct),
                "layout_bbox_20pct": list(layout_20),
                "active_layout_preview": list(
                    column.layout_bbox_10pct if max_layout_overflow == 10 else layout_20
                ),
                "layout_preview_level": max_layout_overflow,
                "layout_20_requires_manual_review": max_layout_overflow == 20,
                "mask": {
                    "kind": "polygon-envelope-audit-mask",
                    "bbox": list(mask_audit.mask_bbox),
                    "pixel_count": mask_audit.pixel_count,
                    "overlap_with_previous_pixels": mask_audit.overlap_with_previous_pixels,
                },
            }
        )

    ignored_audits = []
    for index, ignored in enumerate(contract.ignored_regions, 1):
        polygon = ignored.get("polygon")
        if polygon:
            points = [_integer_point(point) for point in polygon]
            draw.line(points + [points[0]], fill=(220, 30, 45, 255), width=3)
            bbox = _polygon_bbox(points)
            draw.line((bbox[0], bbox[1], bbox[2], bbox[3]), fill=(220, 30, 45, 255), width=2)
            draw.line((bbox[0], bbox[3], bbox[2], bbox[1]), fill=(220, 30, 45, 255), width=2)
            _draw_label(draw, f"X{index}", bbox, font, (220, 30, 45), composed.size)
        ignored_audits.append(
            {
                "index": index,
                "reason": ignored.get("reason", "unspecified"),
                "text": ignored.get("text", ""),
                "polygon": ignored.get("polygon"),
                "image_bbox": ignored.get("image_bbox"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output, format="PNG", compress_level=6, optimize=False)
    output_hash = sha256_file(output)
    original_hash_after = sha256_file(image_path)
    gate_status = (
        "blocked"
        if contract.manual_review.translation_blocked
        else "bypassed"
        if contract.manual_review.bypassed_count
        else "ready"
    )
    audit = _build_audit(
        image_path=image_path,
        original_hash=original_hash_before,
        original_unchanged=original_hash_before == original_hash_after,
        application=application,
        masks=masks,
        output=output,
        output_hash=output_hash,
        gate_status=gate_status,
        max_layout_overflow=max_layout_overflow,
        column_audits=column_audits,
        ignored_audits=ignored_audits,
    )
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return DryRunResult(
        output_path=output,
        audit_path=audit_output,
        output_sha256=output_hash,
        status=gate_status,
        column_count=len(contract.columns),
        pending_count=contract.manual_review.pending_count,
        bypassed_count=contract.manual_review.bypassed_count,
    )


def _build_audit(
    *,
    image_path: Path,
    original_hash: str,
    original_unchanged: bool,
    application: ReviewApplication,
    masks: MaskSet,
    output: Path,
    output_hash: str,
    gate_status: str,
    max_layout_overflow: int,
    column_audits: Sequence[Dict[str, Any]],
    ignored_audits: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    contract = application.contract
    queue = []
    for item in contract.manual_review_queue:
        queue.append(
            {
                "column_id": item.column_id,
                "status": item.status,
                "reason_codes": list(item.reason_codes),
                "current_text": item.current_text,
                "mangaocr_readings": list(item.mangaocr_readings),
                "mangaocr_confidence": item.mangaocr_confidence,
                "paddleocr_text": item.paddleocr_text,
                "paddleocr_confidence": item.paddleocr_confidence,
                "paddle_box_overlap": item.paddle_box_overlap,
                "resolved_text": item.resolved_text,
                "resolved_polygon": (
                    _number_polygon(item.resolved_polygon)
                    if item.resolved_polygon is not None
                    else None
                ),
                "resolved_image_bbox": (
                    list(item.resolved_image_bbox)
                    if item.resolved_image_bbox is not None
                    else None
                ),
                "resolved_layout_bbox_10pct": (
                    list(item.resolved_layout_bbox_10pct)
                    if item.resolved_layout_bbox_10pct is not None
                    else None
                ),
                "polygon": _number_polygon(item.polygon),
                "image_bbox": list(item.image_bbox),
            }
        )
    return {
        "artifact": "manga-localizer-dry-run",
        "artifact_version": 1,
        "dry_run": True,
        "does_not_erase_or_translate": True,
        "status": gate_status,
        "source": {
            "image": str(image_path),
            "image_sha256": original_hash,
            "image_unchanged": original_unchanged,
            "width": contract.source.width,
            "height": contract.source.height,
            "ocr": str(contract.path),
            "ocr_sha256": contract.source_sha256,
            "effective_contract_sha256": contract.effective_sha256,
            "schema": str(contract.schema_path),
            "schema_sha256": contract.schema_sha256,
        },
        "review": {
            "mode": application.mode,
            "decision_file": str(application.decision_path) if application.decision_path else None,
            "decision_sha256": application.decision_sha256,
            "enabled": contract.manual_review.enabled,
            "translation_blocked": contract.manual_review.translation_blocked,
            "candidate_count": contract.manual_review.candidate_count,
            "pending_count": contract.manual_review.pending_count,
            "bypassed_count": contract.manual_review.bypassed_count,
            "reviewed_count": contract.manual_review.reviewed_count,
            "queue": queue,
        },
        "layout_policy": {
            "default_overflow_level": 10,
            "max_layout_overflow": max_layout_overflow,
            "layout_10_is_hard_default": True,
            "layout_20_is_absolute_ceiling": True,
            "layout_20_computed_from_complete_image_bbox": True,
            "compound_expansion_forbidden": True,
            "layout_20_actual_use_requires_reason_and_manual_review": True,
        },
        "summary": {
            "retained_columns": len(contract.columns),
            "translation_units": len(contract.translation_units),
            "ignored_regions": len(contract.ignored_regions),
            "independent_masks": len(masks.columns),
            "mask_union_pixels": masks.union_pixel_count,
            "overlapping_mask_pixels": masks.overlapping_pixel_count,
            "cross_unit_overlap_pixels": masks.cross_unit_overlap_pixel_count,
            "all_retained_columns_mapped": len(masks.columns) == len(contract.columns),
        },
        "mask_overlaps": [
            {
                "first_column_id": item.first_column_id,
                "second_column_id": item.second_column_id,
                "first_unit_id": item.first_unit_id,
                "second_unit_id": item.second_unit_id,
                "pixel_count": item.pixel_count,
                "crosses_translation_units": item.crosses_translation_units,
            }
            for item in masks.overlaps
        ],
        "translation_units": [
            {
                "unit_id": unit.id,
                "column_ids": list(unit.column_ids),
                "source_text": unit.source_text,
                "translation_query": unit.translation_query,
                "panel_index": unit.panel_index,
            }
            for unit in contract.translation_units
        ],
        "columns": list(column_audits),
        "ignored_regions": list(ignored_audits),
        "output": {"path": str(output), "sha256": output_hash},
    }


def _audit_font(image_size: Tuple[int, int]) -> ImageFont.ImageFont:
    size = max(11, min(16, round(min(image_size) * 0.014)))
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label(reading_index: int, unit_index: int, status: str) -> str:
    del unit_index  # Unit identity is encoded by fill color and recorded in audit JSON.
    status_code = {
        "pending": "P",
        "bypassed": "B",
        "approved": "A",
        "corrected": "C",
    }.get(status, "")
    return f"{status_code}{reading_index + 1:02d}"


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor_bbox: Sequence[int],
    font: ImageFont.ImageFont,
    color: Tuple[int, int, int],
    image_size: Tuple[int, int],
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = max(0, min(int(anchor_bbox[0]), image_size[0] - text_width - 4))
    y = max(0, min(int(anchor_bbox[1]) - text_height - 3, image_size[1] - text_height - 4))
    draw.rectangle((x, y, x + text_width + 4, y + text_height + 3), fill=(*color, 235))
    brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    foreground = (0, 0, 0, 255) if brightness > 150 else (255, 255, 255, 255)
    draw.text((x + 2, y + 1), text, font=font, fill=foreground)


def _outlined_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[int],
    color: Tuple[int, int, int, int],
) -> None:
    draw.rectangle(tuple(bbox), outline=(0, 0, 0, 220), width=3)
    draw.rectangle(tuple(bbox), outline=color, width=1)


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[int],
    color: Tuple[int, int, int, int],
    dash: int,
    gap: int,
    width: int,
) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    for start in range(x1, x2, dash + gap):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)
    for start in range(y1, y2, dash + gap):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def _integer_point(point: Sequence[float]) -> Tuple[int, int]:
    return round(float(point[0])), round(float(point[1]))


def _number_polygon(polygon: Sequence[Sequence[float]]) -> list[list[Any]]:
    result = []
    for point in polygon:
        result.append([_clean_number(point[0]), _clean_number(point[1])])
    return result


def _clean_number(value: float) -> Any:
    number = float(value)
    return int(number) if number.is_integer() else number


def _polygon_bbox(points: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
