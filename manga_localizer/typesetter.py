from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .errors import TypesettingError
from .layout_bounds import compute_layout_bbox, measured_overflow_percent
from .ocr_contract import BBox, OCRContract
from .pixel_audit import RegionChangeAudit, audit_region_change


RGBColor = Tuple[int, int, int]
TERMINAL_PUNCTUATION = frozenset("。！？!?…．.")


@dataclass(frozen=True)
class LayoutConfig:
    max_overflow_level: int = 10
    min_font_size: int = 10
    initial_font_ratio: float = 0.88
    character_spacing: int = 0
    line_spacing: int = 2
    outline_width: int = 2
    bubble_text_color: RGBColor = (0, 0, 0)
    artwork_text_color: RGBColor = (0, 0, 0)
    artwork_outline_color: RGBColor = (255, 255, 255)
    font_path: Optional[Path] = None
    terminal_punctuation_font_ratio: float = 0.68
    preserve_unit_center: bool = True
    center_dimension_ratio: float = 0.2
    sample_stride: int = 2


@dataclass(frozen=True)
class LayoutPlacement:
    unit_id: str
    column_id: str
    text_segment: str
    direction: str
    source_bbox: BBox
    bbox: BBox
    layout_bbox_10pct: BBox
    layout_bbox_20pct: BBox
    overflow_level: int
    overflow_reason: Optional[str]
    manual_review_required: bool
    font_path: str
    font_size: int
    character_spacing: int
    line_spacing: int
    text_color: RGBColor
    outline_color: RGBColor
    outline_width: int
    style: str
    status: str
    minimum_layout_percent: float
    layout_width_percent_of_source: float
    layout_height_percent_of_source: float
    terminal_punctuation_font_ratio: float
    overflow_width_percent: float
    overflow_height_percent: float


@dataclass(frozen=True)
class PlacementAudit:
    placement: LayoutPlacement
    region_change: Optional[RegionChangeAudit]


@dataclass(frozen=True)
class TypesetResult:
    image: Image.Image
    placements: Tuple[LayoutPlacement, ...]
    audits: Tuple[PlacementAudit, ...]
    status: str
    typesetting_seconds: float
    pixel_audit_seconds: float


@dataclass(frozen=True)
class TranslationArtifact:
    path: Path
    data: Mapping[str, Any]
    translations_by_unit: Mapping[str, str]
    segments_by_column: Mapping[str, str]


def load_translation_artifact(
    path: Path,
    contract: OCRContract,
) -> TranslationArtifact:
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise TypesettingError(f"translation artifact does not exist: {artifact_path}")
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TypesettingError(f"cannot read translation artifact: {error}") from error
    if not isinstance(data, dict) or data.get("stage") != "D":
        raise TypesettingError("translation artifact must be a stage D JSON")
    if data.get("status") != "translated":
        raise TypesettingError(
            f"translation artifact is not ready for typesetting: {data.get('status')}"
        )
    source = data.get("source") or {}
    if source.get("effective_contract_sha256") != contract.effective_sha256:
        raise TypesettingError("translation artifact does not match this OCR contract")
    policy = data.get("layout_policy") or {}
    if (
        policy.get("soft_constraint_overflow_level") != 10
        or policy.get("hard_constraint_overflow_level") != 20
        or policy.get("compound_expansion_forbidden") is not True
    ):
        raise TypesettingError("translation artifact has an invalid 10%/20% layout policy")
    units = data.get("units")
    if not isinstance(units, list):
        raise TypesettingError("translation artifact units must be an array")
    translations: Dict[str, str] = {}
    for item in units:
        if not isinstance(item, dict):
            raise TypesettingError("translation artifact contains an invalid unit")
        unit_id = str(item.get("unit_id") or "")
        text = item.get("translated_text")
        if not unit_id or not isinstance(text, str) or not text.strip():
            raise TypesettingError(f"unit {unit_id or '<missing>'} has no translation")
        if item.get("translation_status") != "translated":
            raise TypesettingError(f"unit {unit_id} is not translated")
        if unit_id in translations:
            raise TypesettingError(f"duplicate translation unit {unit_id}")
        translations[unit_id] = text
    expected_units = {unit.id for unit in contract.translation_units}
    if set(translations) != expected_units:
        raise TypesettingError("translation artifact does not cover every OCR unit")
    columns = data.get("columns")
    expected_columns = set(contract.columns_by_id)
    if not isinstance(columns, list) or {
        str(item.get("column_id")) for item in columns if isinstance(item, dict)
    } != expected_columns:
        raise TypesettingError("translation artifact does not map every OCR column")
    aligned_segments: Dict[str, str] = {}
    for item in columns:
        if not isinstance(item, dict):
            continue
        segment = item.get("aligned_text_segment")
        if isinstance(segment, str) and segment:
            aligned_segments[str(item["column_id"])] = segment
    for unit in contract.translation_units:
        available = [column_id in aligned_segments for column_id in unit.column_ids]
        if any(available) and not all(available):
            raise TypesettingError(f"unit {unit.id} has partial column alignment")
        if all(available) and "".join(
            aligned_segments[column_id] for column_id in unit.column_ids
        ) != translations[unit.id]:
            raise TypesettingError(f"unit {unit.id} aligned segments change its translation")
    return TranslationArtifact(
        path=artifact_path,
        data=data,
        translations_by_unit=translations,
        segments_by_column=aligned_segments,
    )


def typeset_translation(
    base_image: Image.Image,
    contract: OCRContract,
    translation: TranslationArtifact,
    *,
    config: LayoutConfig = LayoutConfig(),
    style_by_column: Optional[Mapping[str, str]] = None,
) -> TypesetResult:
    typesetting_started = time.perf_counter()
    if base_image.size != contract.image_size:
        raise TypesettingError("base image size differs from OCR contract")
    if config.max_overflow_level not in (10, 20):
        raise TypesettingError("max_overflow_level must be 10 or 20")
    if config.min_font_size < 1 or config.outline_width < 0:
        raise TypesettingError("font size and outline width must be non-negative")
    if not (0 < config.initial_font_ratio <= 1):
        raise TypesettingError("initial_font_ratio must be in (0, 1]")
    if not (0 < config.terminal_punctuation_font_ratio <= 1):
        raise TypesettingError(
            "terminal_punctuation_font_ratio must be in (0, 1]"
        )

    styles = dict(style_by_column or {})
    placements = []
    working = base_image.convert("RGB")
    audits = []
    pixel_audit_seconds = 0.0
    for unit in contract.translation_units:
        text = translation.translations_by_unit[unit.id]
        aligned = (
            tuple(translation.segments_by_column[column_id] for column_id in unit.column_ids)
            if all(column_id in translation.segments_by_column for column_id in unit.column_ids)
            else None
        )
        unit_placements = _plan_unit(
            unit,
            text,
            contract,
            styles,
            config,
            aligned_segments=aligned,
        )
        placement_start = len(placements)
        placements.extend(unit_placements)
        for local_index, placement in enumerate(unit_placements):
            if placement.status != "layout_ready":
                audits.append(PlacementAudit(placement, None))
                continue
            before = working
            working = _draw_placement(working, placement, config)
            pixel_audit_started = time.perf_counter()
            change = audit_region_change(
                before,
                working,
                placement.bbox,
                center_dimension_ratio=config.center_dimension_ratio,
                sample_stride=config.sample_stride,
            )
            pixel_audit_seconds += time.perf_counter() - pixel_audit_started
            audited_placement = placement
            if not change.changed:
                audited_placement = _replace_placement(
                    placement,
                    status="translation_failed",
                    overflow_reason="rendered region has no sampled pixel change",
                )
            audits.append(PlacementAudit(audited_placement, change))
            if audited_placement is not placement:
                placements[placement_start + local_index] = audited_placement

    final_placements = tuple(placements)
    has_failure = any(item.placement.status != "layout_ready" for item in audits)
    needs_review = any(item.placement.manual_review_required for item in audits)
    status = (
        "typeset_failed"
        if has_failure
        else "typeset_pending_review"
        if needs_review
        else "typeset"
    )
    return TypesetResult(
        image=working,
        placements=final_placements,
        audits=tuple(audits),
        status=status,
        typesetting_seconds=time.perf_counter() - typesetting_started,
        pixel_audit_seconds=pixel_audit_seconds,
    )


def default_font_path() -> Path:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path(r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TypesettingError("no Chinese-capable font was found; pass --layout-font")


def styles_from_erasure_audit(path: Path) -> Mapping[str, str]:
    audit_path = Path(path).resolve()
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TypesettingError(f"cannot read erasure audit: {error}") from error
    styles = {}
    for item in data.get("columns", []):
        if not isinstance(item, dict) or not item.get("column_id"):
            continue
        styles[str(item["column_id"])] = (
            "bubble" if item.get("strategy") == "solid_fill" else "artwork"
        )
    return styles


def placements_to_json(result: TypesetResult) -> Dict[str, Any]:
    unit_font_sizes: Dict[str, set[int]] = {}
    unit_source_boxes: Dict[str, list[BBox]] = {}
    unit_layout_boxes: Dict[str, list[BBox]] = {}
    for item in result.placements:
        if item.status == "layout_ready":
            unit_font_sizes.setdefault(item.unit_id, set()).add(item.font_size)
            unit_source_boxes.setdefault(item.unit_id, []).append(item.source_bbox)
            unit_layout_boxes.setdefault(item.unit_id, []).append(item.bbox)
    return {
        "status": result.status,
        "unit_font_audit": [
            {
                "unit_id": unit_id,
                "font_sizes": sorted(font_sizes),
                "uniform_base_font_size": len(font_sizes) == 1,
                "terminal_punctuation_smaller_font_is_allowed": True,
                "source_bbox": list(_bbox_union(unit_source_boxes[unit_id])),
                "layout_bbox": list(_bbox_union(unit_layout_boxes[unit_id])),
                "source_center_2x": list(
                    _bbox_center_2x(_bbox_union(unit_source_boxes[unit_id]))
                ),
                "layout_center_2x": list(
                    _bbox_center_2x(_bbox_union(unit_layout_boxes[unit_id]))
                ),
                "center_offset_2x": [
                    layout - source
                    for layout, source in zip(
                        _bbox_center_2x(_bbox_union(unit_layout_boxes[unit_id])),
                        _bbox_center_2x(_bbox_union(unit_source_boxes[unit_id])),
                    )
                ],
                "center_preserved": (
                    _bbox_center_2x(_bbox_union(unit_layout_boxes[unit_id]))
                    == _bbox_center_2x(_bbox_union(unit_source_boxes[unit_id]))
                ),
            }
            for unit_id, font_sizes in sorted(unit_font_sizes.items())
        ],
        "placements": [
            {
                "unit_id": item.unit_id,
                "column_id": item.column_id,
                "text_segment": item.text_segment,
                "direction": item.direction,
                "source_bbox": list(item.source_bbox),
                "layout_bbox": list(item.bbox),
                "layout_bbox_10pct": list(item.layout_bbox_10pct),
                "layout_bbox_20pct": list(item.layout_bbox_20pct),
                "overflow_level": item.overflow_level,
                "overflow_reason": item.overflow_reason,
                "manual_review_required": item.manual_review_required,
                "font": item.font_path,
                "font_size": item.font_size,
                "character_spacing": item.character_spacing,
                "line_spacing": item.line_spacing,
                "text_color": list(item.text_color),
                "outline_color": list(item.outline_color),
                "outline_width": item.outline_width,
                "style": item.style,
                "status": item.status,
                "minimum_layout_percent": item.minimum_layout_percent,
                "layout_width_percent_of_source": round(
                    item.layout_width_percent_of_source, 6
                ),
                "layout_height_percent_of_source": round(
                    item.layout_height_percent_of_source, 6
                ),
                "terminal_punctuation_font_ratio": (
                    item.terminal_punctuation_font_ratio
                ),
                "overflow_width_percent": round(item.overflow_width_percent, 6),
                "overflow_height_percent": round(item.overflow_height_percent, 6),
            }
            for item in result.placements
        ],
        "pixel_checks": [
            {
                "column_id": item.placement.column_id,
                "status": item.placement.status,
                "bbox": list(item.region_change.bbox)
                if item.region_change is not None
                else None,
                "center_bbox": list(item.region_change.center_bbox)
                if item.region_change is not None
                else None,
                "center_sampled_pixels": (
                    item.region_change.center_sampled_pixels
                    if item.region_change is not None
                    else 0
                ),
                "center_changed_samples": (
                    item.region_change.center_changed_samples
                    if item.region_change is not None
                    else 0
                ),
                "full_sampled_pixels": (
                    item.region_change.full_sampled_pixels
                    if item.region_change is not None
                    else 0
                ),
                "full_changed_samples": (
                    item.region_change.full_changed_samples
                    if item.region_change is not None
                    else 0
                ),
                "changed": (
                    item.region_change.changed if item.region_change is not None else False
                ),
            }
            for item in result.audits
        ],
    }


def _plan_unit(
    unit,
    text: str,
    contract: OCRContract,
    styles: Mapping[str, str],
    config: LayoutConfig,
    *,
    aligned_segments: Optional[Tuple[str, ...]],
) -> Tuple[LayoutPlacement, ...]:
    columns = [contract.columns_by_id[column_id] for column_id in unit.column_ids]
    if not text.strip():
        return tuple(
            _failed_placement(
                unit.id,
                column,
                "empty translation",
                contract.image_size,
                config,
            )
            for column in columns
        )
    directions = {column.direction.lower() for column in columns}
    vertical = not directions.intersection({"horizontal", "h"})
    font_path = config.font_path or default_font_path()
    levels = (10,) if config.max_overflow_level == 10 else (10, 20)
    chosen = None
    for level in levels:
        boxes = {
            column.id: compute_layout_bbox(column.image_bbox, level, contract.image_size)
            for column in columns
        }
        if config.preserve_unit_center:
            centered = _center_unit_boxes(columns, boxes, minimum_percent=80.0)
            if centered is None:
                continue
            boxes = centered
        candidate = _find_fit(
            text,
            columns,
            boxes,
            font_path,
            vertical,
            config,
            aligned_segments,
        )
        if candidate is not None:
            chosen = (level, boxes, candidate)
            break
    if chosen is None:
        reason = (
            "translated sentence cannot fit complete reviewed OCR columns within the "
            f"{config.max_overflow_level}% hard selection"
        )
        return tuple(
            _failed_placement(unit.id, column, reason, contract.image_size, config)
            for column in columns
        )

    level, boxes, candidate = chosen
    font_size, segments = candidate
    placements = []
    manual_review = level == 20
    reason = (
        "10% soft boundary fit after font/layout adjustment"
        if level == 10
        else "10% soft boundary did not fit; direct 20% hard-boundary fallback"
    )
    for column, segment in zip(columns, segments):
        bbox10 = column.layout_bbox_10pct
        bbox20 = compute_layout_bbox(column.image_bbox, 20, contract.image_size)
        selected = boxes[column.id]
        source_width = column.image_bbox[2] - column.image_bbox[0]
        source_height = column.image_bbox[3] - column.image_bbox[1]
        layout_width_percent = (
            (selected[2] - selected[0]) / source_width * 100.0
        )
        layout_height_percent = (
            (selected[3] - selected[1]) / source_height * 100.0
        )
        if layout_width_percent < 80.0 or layout_height_percent < 80.0:
            raise TypesettingError(
                f"layout box for {column.id} is smaller than 80% of its reviewed OCR box"
            )
        style = styles.get(column.id, "artwork")
        if style not in {"bubble", "artwork"}:
            style = "artwork"
        fill = (
            config.bubble_text_color
            if style == "bubble"
            else config.artwork_text_color
        )
        outline = config.artwork_outline_color if style == "artwork" else fill
        stroke = config.outline_width if style == "artwork" else 0
        width_overflow, height_overflow = measured_overflow_percent(
            column.image_bbox,
            selected,
            contract.image_size,
        )
        placements.append(
            LayoutPlacement(
                unit_id=unit.id,
                column_id=column.id,
                text_segment=segment,
                direction="vertical" if vertical else "horizontal",
                source_bbox=column.image_bbox,
                bbox=selected,
                layout_bbox_10pct=bbox10,
                layout_bbox_20pct=bbox20,
                overflow_level=level,
                overflow_reason=reason,
                manual_review_required=manual_review,
                font_path=str(font_path),
                font_size=font_size,
                character_spacing=config.character_spacing,
                line_spacing=config.line_spacing,
                text_color=fill,
                outline_color=outline,
                outline_width=stroke,
                style=style,
                status="layout_ready",
                minimum_layout_percent=80.0,
                layout_width_percent_of_source=layout_width_percent,
                layout_height_percent_of_source=layout_height_percent,
                terminal_punctuation_font_ratio=(
                    config.terminal_punctuation_font_ratio
                ),
                overflow_width_percent=width_overflow,
                overflow_height_percent=height_overflow,
            )
        )
    return tuple(placements)


def _find_fit(
    text: str,
    columns,
    boxes: Mapping[str, BBox],
    font_path: Path,
    vertical: bool,
    config: LayoutConfig,
    aligned_segments: Optional[Tuple[str, ...]],
) -> Optional[Tuple[int, Tuple[str, ...]]]:
    widths = [boxes[column.id][2] - boxes[column.id][0] for column in columns]
    heights = [boxes[column.id][3] - boxes[column.id][1] for column in columns]
    initial = max(
        config.min_font_size,
        min(96, round((max(widths) if vertical else max(heights)) * config.initial_font_ratio)),
    )
    for font_size in range(initial, config.min_font_size - 1, -1):
        font = _load_font(font_path, font_size)
        if vertical:
            segments = _vertical_segments(
                text,
                columns,
                boxes,
                font,
                config,
                aligned_segments,
            )
        else:
            segments = _horizontal_segments(
                text,
                columns,
                boxes,
                font,
                config,
                aligned_segments,
            )
        if segments is not None:
            return font_size, segments
    return None


def _vertical_segments(
    text: str,
    columns,
    boxes: Mapping[str, BBox],
    font: ImageFont.FreeTypeFont,
    config: LayoutConfig,
    aligned_segments: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, ...]]:
    chars = _text_units(text)
    if aligned_segments is None and len(chars) < len(columns):
        return None
    capacities = []
    for column in columns:
        bbox = boxes[column.id]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        glyph_width, glyph_height = _max_glyph_size(
            font,
            chars,
            config.outline_width,
            config.terminal_punctuation_font_ratio,
        )
        if glyph_width > width or glyph_height > height:
            return None
        advance = glyph_height + config.character_spacing
        capacities.append(max(0, (height + config.character_spacing) // max(1, advance)))
    if sum(capacities) < len(chars) or any(value < 1 for value in capacities):
        return None
    if aligned_segments is not None:
        if (
            len(aligned_segments) != len(columns)
            or any(not segment for segment in aligned_segments)
            or "".join(aligned_segments) != text
            or any(
                len(_text_units(segment)) > capacity
                for segment, capacity in zip(aligned_segments, capacities)
            )
        ):
            return None
        return aligned_segments
    return _allocate(chars, capacities, [len(column.text) for column in columns])


def _horizontal_segments(
    text: str,
    columns,
    boxes: Mapping[str, BBox],
    font: ImageFont.FreeTypeFont,
    config: LayoutConfig,
    aligned_segments: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, ...]]:
    chars = _text_units(text)
    if aligned_segments is None and len(chars) < len(columns):
        return None
    _, glyph_height = _max_glyph_size(
        font,
        chars,
        config.outline_width,
        config.terminal_punctuation_font_ratio,
    )
    capacities = []
    for column in columns:
        bbox = boxes[column.id]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        glyph_width, _ = _max_glyph_size(
            font,
            chars,
            config.outline_width,
            config.terminal_punctuation_font_ratio,
        )
        if glyph_width > width or glyph_height > height:
            return None
        per_line = max(1, (width + config.character_spacing) // max(1, glyph_width + config.character_spacing))
        lines = max(1, (height + config.line_spacing) // max(1, glyph_height + config.line_spacing))
        capacities.append(per_line * lines)
    if sum(capacities) < len(chars) or any(value < 1 for value in capacities):
        return None
    if aligned_segments is not None:
        if (
            len(aligned_segments) != len(columns)
            or any(not segment for segment in aligned_segments)
            or "".join(aligned_segments) != text
            or any(
                len(_text_units(segment)) > capacity
                for segment, capacity in zip(aligned_segments, capacities)
            )
        ):
            return None
        return aligned_segments
    return _allocate(chars, capacities, [len(column.text) for column in columns])


def _draw_placement(
    image: Image.Image,
    placement: LayoutPlacement,
    config: LayoutConfig,
) -> Image.Image:
    font = _load_font(Path(placement.font_path), placement.font_size)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if placement.direction == "vertical":
        _draw_vertical(draw, placement, font, config)
    else:
        _draw_horizontal(draw, placement, font, config)
    clip = Image.new("L", image.size, 0)
    x1, y1, x2, y2 = placement.bbox
    ImageDraw.Draw(clip).rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    alpha = ImageChops.multiply(layer.getchannel("A"), clip)
    layer.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def _draw_vertical(
    draw,
    placement: LayoutPlacement,
    font: ImageFont.FreeTypeFont,
    config: LayoutConfig,
) -> None:
    x1, y1, x2, y2 = placement.bbox
    y = y1
    glyph_height = _max_glyph_size(
        font,
        _text_units(placement.text_segment),
        placement.outline_width,
        config.terminal_punctuation_font_ratio,
    )[1]
    advance = glyph_height + placement.character_spacing
    for unit in _text_units(placement.text_segment):
        unit_font, unit_stroke = _font_and_stroke_for_unit(
            font,
            unit,
            placement.outline_width,
            config.terminal_punctuation_font_ratio,
        )
        measured = draw.textbbox(
            (0, 0), unit, font=unit_font, stroke_width=unit_stroke
        )
        width = measured[2] - measured[0]
        height = measured[3] - measured[1]
        x = round((x1 + x2 - width) / 2) - measured[0]
        unit_y = y + max(0, (glyph_height - height) / 2)
        draw.text(
            (x, round(unit_y) - measured[1]),
            unit,
            font=unit_font,
            fill=placement.text_color,
            stroke_width=unit_stroke,
            stroke_fill=placement.outline_color,
        )
        y += advance


def _draw_horizontal(
    draw,
    placement: LayoutPlacement,
    font: ImageFont.FreeTypeFont,
    config: LayoutConfig,
) -> None:
    x1, y1, x2, y2 = placement.bbox
    chars = _text_units(placement.text_segment)
    glyph_width, glyph_height = _max_glyph_size(
        font,
        chars,
        placement.outline_width,
        config.terminal_punctuation_font_ratio,
    )
    per_line = max(1, (x2 - x1 + config.character_spacing) // max(1, glyph_width + config.character_spacing))
    advance_x = glyph_width + config.character_spacing
    advance_y = glyph_height + placement.line_spacing
    for index, unit in enumerate(chars):
        row, col = divmod(index, per_line)
        unit_font, unit_stroke = _font_and_stroke_for_unit(
            font,
            unit,
            placement.outline_width,
            config.terminal_punctuation_font_ratio,
        )
        measured = draw.textbbox(
            (0, 0), unit, font=unit_font, stroke_width=unit_stroke
        )
        height = measured[3] - measured[1]
        x = x1 + col * advance_x - measured[0]
        y = (
            y1
            + row * advance_y
            + max(0, (glyph_height - height) / 2)
            - measured[1]
        )
        draw.text(
            (round(x), round(y)),
            unit,
            font=unit_font,
            fill=placement.text_color,
            stroke_width=unit_stroke,
            stroke_fill=placement.outline_color,
        )


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size=size)
    except (OSError, ValueError) as error:
        raise TypesettingError(f"cannot load layout font {path}: {error}") from error


def _max_glyph_size(
    font: ImageFont.FreeTypeFont,
    chars: Sequence[str],
    stroke_width: int,
    terminal_punctuation_font_ratio: float = 0.68,
) -> Tuple[int, int]:
    sample = list(chars or ["国"])
    measured = [font.getbbox("国", stroke_width=stroke_width)]
    for unit in sample:
        unit_font, unit_stroke = _font_and_stroke_for_unit(
            font,
            unit,
            stroke_width,
            terminal_punctuation_font_ratio,
        )
        measured.append(unit_font.getbbox(unit, stroke_width=unit_stroke))
    return (
        max(item[2] - item[0] for item in measured),
        max(item[3] - item[1] for item in measured),
    )


def _text_units(text: str) -> Tuple[str, ...]:
    chars = [char for char in text if char not in "\r\n"]
    punctuation_start = len(chars)
    while (
        punctuation_start > 0
        and chars[punctuation_start - 1] in TERMINAL_PUNCTUATION
    ):
        punctuation_start -= 1
    if punctuation_start == len(chars):
        return tuple(chars)
    return tuple(chars[:punctuation_start]) + (
        "".join(chars[punctuation_start:]),
    )


def _font_and_stroke_for_unit(
    font: ImageFont.FreeTypeFont,
    unit: str,
    stroke_width: int,
    terminal_punctuation_font_ratio: float,
) -> Tuple[ImageFont.FreeTypeFont, int]:
    if not unit or not all(char in TERMINAL_PUNCTUATION for char in unit):
        return font, stroke_width
    scale = min(terminal_punctuation_font_ratio, 1.0 / len(unit))
    size = max(1, round(font.size * scale))
    path = getattr(font, "path", None)
    compact_font = _load_font(Path(path), size) if path else font
    compact_stroke = (
        max(1, round(stroke_width * scale)) if stroke_width else 0
    )
    return compact_font, compact_stroke


def _center_unit_boxes(
    columns,
    boxes: Mapping[str, BBox],
    *,
    minimum_percent: float,
) -> Optional[Dict[str, BBox]]:
    adjusted = {column.id: boxes[column.id] for column in columns}
    source_union = _bbox_union([column.image_bbox for column in columns])
    target_center = _bbox_center_2x(source_union)
    for axis in (0, 1):
        current_union = _bbox_union(list(adjusted.values()))
        current_center = _bbox_center_2x(current_union)[axis]
        difference = current_center - target_center[axis]
        if difference == 0:
            continue
        start_index = axis
        end_index = axis + 2
        if difference > 0:
            target_end = target_center[axis] - current_union[start_index]
            for column in columns:
                bbox = list(adjusted[column.id])
                if bbox[end_index] > target_end:
                    bbox[end_index] = target_end
                    adjusted[column.id] = tuple(bbox)
        else:
            target_start = target_center[axis] - current_union[end_index]
            for column in columns:
                bbox = list(adjusted[column.id])
                if bbox[start_index] < target_start:
                    bbox[start_index] = target_start
                    adjusted[column.id] = tuple(bbox)
        for column in columns:
            bbox = adjusted[column.id]
            source = column.image_bbox
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                return None
            width_percent = (bbox[2] - bbox[0]) / (source[2] - source[0]) * 100
            height_percent = (bbox[3] - bbox[1]) / (source[3] - source[1]) * 100
            if width_percent < minimum_percent or height_percent < minimum_percent:
                return None
    if _bbox_center_2x(_bbox_union(list(adjusted.values()))) != target_center:
        return None
    return adjusted


def _bbox_union(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise TypesettingError("cannot compute the union of no layout boxes")
    return (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )


def _bbox_center_2x(bbox: BBox) -> Tuple[int, int]:
    return bbox[0] + bbox[2], bbox[1] + bbox[3]


def _allocate(
    chars: Sequence[str],
    capacities: Sequence[int],
    weights: Sequence[int],
) -> Tuple[str, ...]:
    if len(capacities) != len(weights):
        raise TypesettingError("layout capacities and source weights differ")
    allocation = [1 for _ in capacities]
    remaining = len(chars) - len(allocation)
    normalized_weights = [max(1, value) for value in weights]
    while remaining > 0:
        candidates = [
            index
            for index, capacity in enumerate(capacities)
            if allocation[index] < capacity
        ]
        if not candidates:
            raise TypesettingError("layout capacity ended before the sentence")
        chosen = max(
            candidates,
            key=lambda index: (
                normalized_weights[index] / allocation[index],
                normalized_weights[index],
                -index,
            ),
        )
        allocation[chosen] += 1
        remaining -= 1
    result = []
    offset = 0
    for take in allocation:
        result.append("".join(chars[offset : offset + take]))
        offset += take
    if offset != len(chars):
        raise TypesettingError("internal text allocation did not consume the full sentence")
    return tuple(result)


def _failed_placement(
    unit_id: str,
    column,
    reason: str,
    image_size: Tuple[int, int],
    config: LayoutConfig,
) -> LayoutPlacement:
    bbox10 = column.layout_bbox_10pct
    bbox20 = compute_layout_bbox(column.image_bbox, 20, image_size)
    return LayoutPlacement(
        unit_id=unit_id,
        column_id=column.id,
        text_segment="",
        direction=column.direction,
        source_bbox=column.image_bbox,
        bbox=bbox10,
        layout_bbox_10pct=bbox10,
        layout_bbox_20pct=bbox20,
        overflow_level=10,
        overflow_reason=reason,
        manual_review_required=True,
        font_path=str(config.font_path or ""),
        font_size=0,
        character_spacing=config.character_spacing,
        line_spacing=config.line_spacing,
        text_color=config.bubble_text_color,
        outline_color=config.artwork_outline_color,
        outline_width=config.outline_width,
        style="unknown",
        status="layout_failed",
        minimum_layout_percent=80.0,
        layout_width_percent_of_source=(
            (bbox10[2] - bbox10[0])
            / (column.image_bbox[2] - column.image_bbox[0])
            * 100.0
        ),
        layout_height_percent_of_source=(
            (bbox10[3] - bbox10[1])
            / (column.image_bbox[3] - column.image_bbox[1])
            * 100.0
        ),
        terminal_punctuation_font_ratio=(
            config.terminal_punctuation_font_ratio
        ),
        overflow_width_percent=0.0,
        overflow_height_percent=0.0,
    )


def _replace_placement(placement: LayoutPlacement, **changes: Any) -> LayoutPlacement:
    values = placement.__dict__.copy()
    values.update(changes)
    return LayoutPlacement(**values)
