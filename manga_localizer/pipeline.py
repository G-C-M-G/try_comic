from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .eraser import (
    ErasureConfig,
    ErasureResult,
    LocalStatisticsStyleClassifier,
    erase_image,
)
from .layout_bounds import compute_layout_bbox
from .errors import ErasureError, TranslationError
from .ocr_contract import sha256_file
from .region_masks import build_column_masks
from .review import ReviewApplication, ensure_translation_allowed
from .state import (
    load_editable_state,
    update_state_after_reembed,
    write_pipeline_state,
)
from .translator import (
    Glossary,
    TranslationBundle,
    TranslationCache,
    Translator,
    translate_contract,
)
from .typesetter import (
    LayoutConfig,
    TranslationArtifact,
    TypesetResult,
    load_translation_artifact,
    placements_to_json,
    styles_from_erasure_audit,
    typeset_translation,
)


@dataclass(frozen=True)
class ErasePipelineResult:
    output_path: Path
    mask_path: Path
    difference_path: Path
    audit_path: Path
    output_sha256: str
    status: str
    column_count: int
    changed_pixels: int
    changed_outside_mask_pixels: int


@dataclass(frozen=True)
class TranslationPipelineResult:
    output_path: Path
    output_sha256: str
    status: str
    unit_count: int
    column_count: int
    failed_unit_count: int
    cache_hit_count: int


@dataclass(frozen=True)
class TypesetPipelineResult:
    output_path: Path
    audit_path: Path
    output_sha256: str
    status: str
    placement_count: int
    overflow_20_count: int
    failed_count: int
    hard_limit_outside_changed_pixels: int
    pixel_audit_seconds: float


@dataclass(frozen=True)
class FullPipelineResult:
    output_path: Path
    state_path: Path
    stage_directory: Path
    status: str
    erasure: ErasePipelineResult
    translation: TranslationPipelineResult
    typesetting: TypesetPipelineResult


def render_typesetting(
    image_path: Path,
    erased_image_path: Path,
    translation_path: Path,
    application: ReviewApplication,
    output_path: Path,
    *,
    audit_path: Optional[Path] = None,
    erasure_audit_path: Optional[Path] = None,
    config: LayoutConfig = LayoutConfig(),
) -> TypesetPipelineResult:
    ensure_translation_allowed(application)
    contract = application.contract
    source_path = Path(image_path).resolve()
    erased_path = Path(erased_image_path).resolve()
    output = _png_path(output_path, "typeset output")
    audit_output = (
        Path(audit_path).resolve()
        if audit_path is not None
        else output.with_name(output.stem + ".typeset.audit.json")
    )
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    with Image.open(erased_path) as opened:
        erased = opened.convert("RGB")
    if source.size != contract.image_size or erased.size != contract.image_size:
        raise ValueError("source and erased image must match the OCR dimensions")
    translation = load_translation_artifact(translation_path, contract)
    styles = (
        styles_from_erasure_audit(erasure_audit_path)
        if erasure_audit_path is not None
        else _infer_column_styles(source, contract)
    )
    if set(styles) != set(contract.columns_by_id):
        raise ValueError("erasure style mapping must cover every retained OCR column")
    return _render_typesetting_artifact(
        source_path,
        erased_path,
        translation,
        application,
        output,
        audit_output,
        styles,
        config,
    )


def render_state_typesetting(
    state_path: Path,
    application: ReviewApplication,
    output_path: Path,
    *,
    image_path: Optional[Path] = None,
    erased_image_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
    state_output_path: Optional[Path] = None,
    config: LayoutConfig = LayoutConfig(),
) -> tuple[TypesetPipelineResult, Path, tuple[str, ...]]:
    ensure_translation_allowed(application)
    editable = load_editable_state(
        state_path,
        application.contract,
        explicit_image=image_path,
        explicit_erased_image=erased_image_path,
    )
    output = _png_path(output_path, "typeset output")
    audit_output = (
        Path(audit_path).resolve()
        if audit_path is not None
        else output.with_name(output.stem + ".typeset.audit.json")
    )
    result = _render_typesetting_artifact(
        editable.image_path,
        editable.erased_image_path,
        editable.translation,
        application,
        output,
        audit_output,
        editable.styles_by_column,
        config,
    )
    updated_state = update_state_after_reembed(
        editable,
        typeset_audit_path=result.audit_path,
        output_path=result.output_path,
        state_output_path=state_output_path,
    )
    return result, updated_state, editable.invalidated_alignment_units


def _render_typesetting_artifact(
    source_path: Path,
    erased_path: Path,
    translation: TranslationArtifact,
    application: ReviewApplication,
    output: Path,
    audit_output: Path,
    styles: Mapping[str, str],
    config: LayoutConfig,
) -> TypesetPipelineResult:
    contract = application.contract
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    with Image.open(erased_path) as opened:
        erased = opened.convert("RGB")
    if source.size != contract.image_size or erased.size != contract.image_size:
        raise ValueError("source and erased image must match the OCR dimensions")
    if set(styles) != set(contract.columns_by_id):
        raise ValueError("erasure style mapping must cover every retained OCR column")
    typeset = typeset_translation(
        erased,
        contract,
        translation,
        config=config,
        style_by_column=styles,
    )

    hard_audit_started = time.perf_counter()
    hard_mask = _hard_layout_union(contract)
    difference = ImageChops.difference(erased, typeset.image)
    changed_mask = _binary_difference_mask(difference)
    outside_hard = ImageChops.darker(changed_mask, ImageOps.invert(hard_mask))
    outside_hard_pixels = outside_hard.histogram()[255]
    hard_audit_seconds = time.perf_counter() - hard_audit_started
    pixel_audit_seconds = typeset.pixel_audit_seconds + hard_audit_seconds
    status = typeset.status
    if outside_hard_pixels or pixel_audit_seconds > 60.0:
        status = "typeset_failed"

    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    typeset.image.save(output, format="PNG", compress_level=6, optimize=False)
    payload = _typeset_audit(
        image_path=source_path,
        erased_path=erased_path,
        translation_path=translation.path,
        application=application,
        typeset=typeset,
        config=config,
        output=output,
        status=status,
        outside_hard_pixels=outside_hard_pixels,
        pixel_audit_seconds=pixel_audit_seconds,
    )
    audit_output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return TypesetPipelineResult(
        output_path=output,
        audit_path=audit_output,
        output_sha256=sha256_file(output),
        status=status,
        placement_count=len(typeset.placements),
        overflow_20_count=sum(
            item.overflow_level == 20 for item in typeset.placements
        ),
        failed_count=sum(item.status != "layout_ready" for item in typeset.placements),
        hard_limit_outside_changed_pixels=outside_hard_pixels,
        pixel_audit_seconds=pixel_audit_seconds,
    )


def render_full_pipeline(
    image_path: Path,
    application: ReviewApplication,
    output_path: Path,
    *,
    translators: tuple[Translator, ...],
    glossary: Glossary,
    cache: Optional[TranslationCache] = None,
    max_attempts_per_translator: int = 2,
    state_path: Optional[Path] = None,
    stage_directory: Optional[Path] = None,
    erasure_config: ErasureConfig = ErasureConfig(),
    layout_config: LayoutConfig = LayoutConfig(),
) -> FullPipelineResult:
    ensure_translation_allowed(application)
    source = Path(image_path).resolve()
    output = _png_path(output_path, "final output")
    stages = (
        Path(stage_directory).resolve()
        if stage_directory is not None
        else output.parent / f"{output.stem}.stages"
    )
    state_output = (
        Path(state_path).resolve()
        if state_path is not None
        else output.with_name(output.stem + ".state.json")
    )
    stages.mkdir(parents=True, exist_ok=True)
    erasure = render_erasure(
        source,
        application,
        stages / "erased.png",
        mask_path=stages / "mask.png",
        difference_path=stages / "erase.diff.png",
        audit_path=stages / "erase.audit.json",
        config=erasure_config,
    )
    if erasure.status == "erasure_failed":
        raise ErasureError(
            f"full pipeline stopped after erasure failure: {erasure.audit_path}"
        )
    translation = render_translation(
        application,
        stages / "translation.json",
        translators=translators,
        glossary=glossary,
        cache=cache,
        max_attempts_per_translator=max_attempts_per_translator,
    )
    if translation.status != "translated":
        raise TranslationError(
            f"full pipeline stopped after translation failure: {translation.output_path}"
        )
    typesetting = render_typesetting(
        source,
        erasure.output_path,
        translation.output_path,
        application,
        output,
        audit_path=stages / "typeset.audit.json",
        erasure_audit_path=erasure.audit_path,
        config=layout_config,
    )
    written_state = write_pipeline_state(
        state_output,
        application,
        image_path=source,
        erased_image_path=erasure.output_path,
        mask_path=erasure.mask_path,
        erasure_audit_path=erasure.audit_path,
        translation_path=translation.output_path,
        typeset_audit_path=typesetting.audit_path,
        output_path=typesetting.output_path,
    )
    return FullPipelineResult(
        output_path=typesetting.output_path,
        state_path=written_state,
        stage_directory=stages,
        status=typesetting.status,
        erasure=erasure,
        translation=translation,
        typesetting=typesetting,
    )
def render_translation(
    application: ReviewApplication,
    output_path: Path,
    *,
    translators: tuple[Translator, ...],
    glossary: Glossary,
    cache: Optional[TranslationCache] = None,
    max_attempts_per_translator: int = 2,
) -> TranslationPipelineResult:
    ensure_translation_allowed(application)
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("translation output must use .json")
    bundle = translate_contract(
        application.contract,
        translators,
        glossary=glossary,
        cache=cache,
        max_attempts_per_translator=max_attempts_per_translator,
    )
    artifact = _translation_artifact(application, bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return TranslationPipelineResult(
        output_path=output,
        output_sha256=sha256_file(output),
        status=bundle.status,
        unit_count=len(bundle.units),
        column_count=len(bundle.columns),
        failed_unit_count=len(bundle.failed_units),
        cache_hit_count=bundle.cache_hit_count,
    )


def render_erasure(
    image_path: Path,
    application: ReviewApplication,
    output_path: Path,
    *,
    mask_path: Optional[Path] = None,
    difference_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
    config: ErasureConfig = ErasureConfig(),
) -> ErasePipelineResult:
    ensure_translation_allowed(application)
    image_source = Path(image_path).resolve()
    output = _png_path(output_path, "erased output")
    mask_output = _png_path(
        mask_path or output.with_name(output.stem + ".mask.png"),
        "mask output",
    )
    difference_output = _png_path(
        difference_path or output.with_name(output.stem + ".diff.png"),
        "difference output",
    )
    audit_output = (
        Path(audit_path).resolve()
        if audit_path is not None
        else output.with_name(output.stem + ".erase.audit.json")
    )
    source_hash_before = sha256_file(image_source)
    with Image.open(image_source) as opened:
        source = opened.convert("RGB")
    erasure = erase_image(source, application.contract, config=config)

    for path in (output, mask_output, difference_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    erasure.image.save(output, format="PNG", compress_level=6, optimize=False)
    erasure.erasure_masks.combined.save(
        mask_output,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    erasure.difference.save(
        difference_output,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    source_hash_after = sha256_file(image_source)
    failed_columns = [item.column_id for item in erasure.columns if not item.changed_pixels]
    status = (
        "erasure_failed"
        if failed_columns or erasure.pixel_audit_seconds > 60.0
        else "erased_with_review_bypass"
        if application.bypassed_count
        else "erased"
    )
    audit = _erasure_audit(
        image_path=image_source,
        image_sha256=source_hash_before,
        source_unchanged=source_hash_before == source_hash_after,
        application=application,
        erasure=erasure,
        config=config,
        output=output,
        mask_output=mask_output,
        difference_output=difference_output,
        status=status,
    )
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ErasePipelineResult(
        output_path=output,
        mask_path=mask_output,
        difference_path=difference_output,
        audit_path=audit_output,
        output_sha256=sha256_file(output),
        status=status,
        column_count=len(erasure.columns),
        changed_pixels=erasure.pixels.changed_pixels,
        changed_outside_mask_pixels=erasure.pixels.changed_outside_mask_pixels,
    )


def _erasure_audit(
    *,
    image_path: Path,
    image_sha256: str,
    source_unchanged: bool,
    application: ReviewApplication,
    erasure: ErasureResult,
    config: ErasureConfig,
    output: Path,
    mask_output: Path,
    difference_output: Path,
    status: str,
) -> Dict[str, Any]:
    contract = application.contract
    return {
        "artifact": "manga-localizer-erasure",
        "artifact_version": 1,
        "stage": "C",
        "erase_only": True,
        "translated": False,
        "status": status,
        "source": {
            "image": str(image_path),
            "image_sha256": image_sha256,
            "image_unchanged": source_unchanged,
            "width": contract.source.width,
            "height": contract.source.height,
            "ocr": str(contract.path),
            "ocr_sha256": contract.source_sha256,
            "effective_contract_sha256": contract.effective_sha256,
            "schema_sha256": contract.schema_sha256,
        },
        "review": {
            "mode": application.mode,
            "decision_file": (
                str(application.decision_path) if application.decision_path else None
            ),
            "decision_sha256": application.decision_sha256,
            "approved_count": application.approved_count,
            "corrected_count": application.corrected_count,
            "rejected_count": application.rejected_count,
            "pending_count": application.pending_count,
            "bypassed_count": application.bypassed_count,
        },
        "config": {
            "style_override": config.style_override,
            "fill_ring_radius": config.fill_ring_radius,
            "inpaint_padding": config.inpaint_padding,
            "mask": {
                "glyph_threshold_margin": config.mask.glyph_threshold_margin,
                "glyph_dilation": config.mask.glyph_dilation,
                "artwork_margin": config.mask.artwork_margin,
                "artwork_search_margin": config.mask.artwork_search_margin,
                "artwork_dark_threshold": config.mask.artwork_dark_threshold,
                "artwork_light_threshold": config.mask.artwork_light_threshold,
                "artwork_outline_contact_radius": (
                    config.mask.artwork_outline_contact_radius
                ),
                "artwork_core_contact_radius": config.mask.artwork_core_contact_radius,
                "artwork_growth_radius": config.mask.artwork_growth_radius,
                "minimum_glyph_pixels": config.mask.minimum_glyph_pixels,
            },
        },
        "summary": {
            "retained_columns": len(contract.columns),
            "independent_geometry_masks": len(erasure.geometry_masks.columns),
            "independent_erasure_masks": len(erasure.erasure_masks.columns),
            "mask_union_pixels": erasure.erasure_masks.union_pixel_count,
            "solid_fill_columns": sum(
                item.strategy == "solid_fill" for item in erasure.columns
            ),
            "inpaint_columns": sum(item.strategy == "inpaint" for item in erasure.columns),
            "all_retained_columns_processed": len(erasure.columns) == len(contract.columns),
            "failed_columns": [
                item.column_id for item in erasure.columns if not item.changed_pixels
            ],
            "erasure_seconds": round(erasure.erasure_seconds, 6),
            "pixel_audit_seconds": round(erasure.pixel_audit_seconds, 6),
            "pixel_audit_time_budget_seconds": 60,
            "within_pixel_audit_time_budget": erasure.pixel_audit_seconds <= 60.0,
            "time_budget_applies_to": "pixel_audit_only",
        },
        "pixel_difference": {
            "allowed_mask_pixels": erasure.pixels.allowed_mask_pixels,
            "changed_pixels": erasure.pixels.changed_pixels,
            "changed_inside_mask_pixels": erasure.pixels.changed_inside_mask_pixels,
            "changed_outside_mask_pixels": erasure.pixels.changed_outside_mask_pixels,
            "unchanged_mask_pixels": erasure.pixels.unchanged_mask_pixels,
            "outside_mask_unchanged": erasure.pixels.outside_mask_unchanged,
            "difference_bbox": (
                list(erasure.pixels.difference_bbox)
                if erasure.pixels.difference_bbox is not None
                else None
            ),
            "max_channel_delta": erasure.pixels.max_channel_delta,
            "sum_absolute_difference": erasure.pixels.sum_absolute_difference,
        },
        "downstream_translation_gate": {
            "required": True,
            "rule": "sampled center core or sampled full layout region must change",
            "center_core_ratio": 0.2,
            "sample_stride_pixels": 2,
            "full_region_is_sampled_not_exhaustive": True,
            "zero_changed_pixels_status": "translation_failed",
            "recovery_order": [
                "retry_current_translator",
                "try_alternate_translator_or_render_strategy",
                "manual_review",
            ],
        },
        "columns": [
            {
                "reading_position": item.reading_position + 1,
                "column_id": item.column_id,
                "unit_id": item.unit_id,
                "source_text": contract.columns_by_id[item.column_id].text,
                "strategy": item.strategy,
                "classifier": item.classifier,
                "classifier_reason": item.classifier_reason,
                "background": {
                    "sample_pixels": item.background.sample_pixels,
                    "mean_luminance": round(item.background.mean_luminance, 4),
                    "luminance_stddev": round(item.background.luminance_stddev, 4),
                    "white_ratio": round(item.background.white_ratio, 6),
                    "dark_ratio": round(item.background.dark_ratio, 6),
                },
                "mask": {
                    "kind": item.mask_kind,
                    "bbox": list(item.mask_bbox),
                    "pixel_count": item.mask_pixels,
                    "geometry_pixel_count": item.geometry_pixels,
                    "threshold": item.threshold,
                    "dilation_pixels": item.dilation_pixels,
                    "contains_geometry_envelope": item.contains_geometry_envelope,
                    "white_outline_pixels": item.white_outline_pixels,
                    "raw_sha256": _image_bytes_sha256(
                        next(
                            mask.image
                            for mask in erasure.erasure_masks.columns
                            if mask.audit.column_id == item.column_id
                        )
                    ),
                },
                "eraser": {
                    "backend": item.backend,
                    "fill_color": list(item.fill_color) if item.fill_color else None,
                },
                "changed_pixels": item.changed_pixels,
                "status": "erased" if item.changed_pixels else "erase_failed",
            }
            for item in erasure.columns
        ],
        "outputs": {
            "erased": {"path": str(output), "sha256": sha256_file(output)},
            "mask": {"path": str(mask_output), "sha256": sha256_file(mask_output)},
            "difference": {
                "path": str(difference_output),
                "sha256": sha256_file(difference_output),
            },
        },
    }


def _translation_artifact(
    application: ReviewApplication,
    bundle: TranslationBundle,
) -> Dict[str, Any]:
    contract = application.contract
    units_by_id = {item.unit_id: item for item in bundle.units}
    columns_by_id = {item.column_id: item for item in bundle.columns}
    return {
        "artifact": "manga-localizer-translation",
        "artifact_version": 1,
        "stage": "D",
        "status": bundle.status,
        "target_language": bundle.target_language,
        "prompt_version": bundle.prompt_version,
        "translated_as_complete_units": True,
        "per_column_translation_forbidden": True,
        "source": {
            "ocr": str(contract.path),
            "ocr_sha256": contract.source_sha256,
            "effective_contract_sha256": contract.effective_sha256,
            "schema_sha256": contract.schema_sha256,
            "width": contract.source.width,
            "height": contract.source.height,
        },
        "review": {
            "mode": application.mode,
            "decision_file": (
                str(application.decision_path) if application.decision_path else None
            ),
            "decision_sha256": application.decision_sha256,
            "approved_count": application.approved_count,
            "corrected_count": application.corrected_count,
            "rejected_count": application.rejected_count,
            "pending_count": application.pending_count,
            "bypassed_count": application.bypassed_count,
        },
        "glossary_sha256": bundle.glossary_sha256,
        "context_policy": {
            "scope": "full_page",
            "current_input": "translation_units.source_text",
            "column_markers": "translation_units.translation_query",
            "column_markers_are_not_independent_translation_requests": True,
        },
        "layout_policy": {
            "basis": "reviewed_complete_ocr_image_bbox",
            "soft_constraint_overflow_level": 10,
            "hard_constraint_overflow_level": 20,
            "soft_constraint_actions": [
                "reduce_font_size",
                "adjust_lines_or_columns",
                "adjust_character_spacing",
            ],
            "hard_limit_fallback": (
                "compute layout_bbox_20pct directly from image_bbox; record reason "
                "and require manual review"
            ),
            "compound_expansion_forbidden": True,
            "pixels_outside_20pct_forbidden": True,
        },
        "summary": {
            "translation_units": len(bundle.units),
            "retained_columns": len(bundle.columns),
            "translated_units": sum(
                item.status == "translated" for item in bundle.units
            ),
            "failed_units": [
                item.unit_id for item in bundle.units if item.status != "translated"
            ],
            "all_columns_mapped": len(bundle.columns) == len(contract.columns),
        },
        "units": [
            {
                "unit_id": unit.id,
                "panel_index": unit.panel_index,
                "column_ids": list(unit.column_ids),
                "source_text": unit.source_text,
                "translation_query": unit.translation_query,
                "translated_text": units_by_id[unit.id].translated_text,
                "aligned_column_segments": (
                    [
                        {"column_id": column_id, "text": segment}
                        for column_id, segment in zip(
                            unit.column_ids,
                            units_by_id[unit.id].column_segments,
                        )
                    ]
                    if len(units_by_id[unit.id].column_segments) == len(unit.column_ids)
                    else []
                ),
                "translation_status": units_by_id[unit.id].status,
                "backend": units_by_id[unit.id].backend,
                "model": units_by_id[unit.id].model,
                "cache_key": units_by_id[unit.id].cache_key,
                "failure_reason": units_by_id[unit.id].failure_reason,
                "failed_attempts": (
                    [
                        {
                            "backend": attempt.backend,
                            "model": attempt.model,
                            "attempt": attempt.attempt,
                            "status": attempt.status,
                            "error": attempt.error,
                        }
                        for attempt in units_by_id[unit.id].attempts
                    ]
                    if units_by_id[unit.id].status != "translated"
                    else []
                ),
            }
            for unit in contract.translation_units
        ],
        "columns": [
            {
                "column_id": column_id,
                "unit_id": columns_by_id[column_id].unit_id,
                "unit_column_index": columns_by_id[column_id].unit_column_index,
                "unit_column_count": columns_by_id[column_id].unit_column_count,
                "source_text": columns_by_id[column_id].source_text,
                "translated_complete_unit_text": (
                    columns_by_id[column_id].translated_text
                ),
                "aligned_text_segment": columns_by_id[column_id].aligned_text_segment,
                "translation_status": columns_by_id[column_id].status,
                "image_bbox": list(contract.columns_by_id[column_id].image_bbox),
                "layout_bbox_10pct_soft": list(
                    contract.columns_by_id[column_id].layout_bbox_10pct
                ),
                "layout_bbox_20pct_hard": list(
                    compute_layout_bbox(
                        contract.columns_by_id[column_id].image_bbox,
                        20,
                        contract.image_size,
                    )
                ),
                "selected_overflow_level": None,
                "selected_layout_bbox": None,
                "layout_status": "not_typeset",
            }
            for column_id in contract.reading_order
        ],
    }


def _typeset_audit(
    *,
    image_path: Path,
    erased_path: Path,
    translation_path: Path,
    application: ReviewApplication,
    typeset: TypesetResult,
    config: LayoutConfig,
    output: Path,
    status: str,
    outside_hard_pixels: int,
    pixel_audit_seconds: float,
) -> Dict[str, Any]:
    contract = application.contract
    details = placements_to_json(typeset)
    unit_font_audit = details["unit_font_audit"]
    overflow_20 = [
        item.column_id for item in typeset.placements if item.overflow_level == 20
    ]
    failed = [
        item.column_id for item in typeset.placements if item.status != "layout_ready"
    ]
    return {
        "artifact": "manga-localizer-typeset",
        "artifact_version": 1,
        "stage": "E",
        "status": status,
        "source": {
            "image": str(image_path),
            "image_sha256": sha256_file(image_path),
            "erased_image": str(erased_path),
            "erased_image_sha256": sha256_file(erased_path),
            "translation": str(translation_path),
            "translation_sha256": sha256_file(translation_path),
            "ocr": str(contract.path),
            "ocr_sha256": contract.source_sha256,
            "effective_contract_sha256": contract.effective_sha256,
            "width": contract.source.width,
            "height": contract.source.height,
        },
        "review": {
            "mode": application.mode,
            "decision_file": (
                str(application.decision_path) if application.decision_path else None
            ),
            "decision_sha256": application.decision_sha256,
            "pending_count": application.pending_count,
            "bypassed_count": application.bypassed_count,
        },
        "layout_policy": {
            "soft_constraint_overflow_level": 10,
            "hard_constraint_overflow_level": 20,
            "selected_max_overflow_level": config.max_overflow_level,
            "basis": "reviewed_complete_ocr_image_bbox",
            "minimum_layout_percent_of_source_bbox": 80,
            "preserve_complete_unit_center": config.preserve_unit_center,
            "compound_expansion_forbidden": True,
            "overflow_20_requires_reason": True,
            "overflow_20_requires_manual_review": True,
            "pixels_outside_20pct_forbidden": True,
        },
        "config": {
            "font": str(config.font_path) if config.font_path else None,
            "min_font_size": config.min_font_size,
            "initial_font_ratio": config.initial_font_ratio,
            "character_spacing": config.character_spacing,
            "line_spacing": config.line_spacing,
            "outline_width": config.outline_width,
            "terminal_punctuation_font_ratio": (
                config.terminal_punctuation_font_ratio
            ),
            "terminal_punctuation_max_layout_units": 1,
            "preserve_unit_center": config.preserve_unit_center,
            "center_dimension_ratio": config.center_dimension_ratio,
            "sample_stride": config.sample_stride,
        },
        "summary": {
            "retained_columns": len(contract.columns),
            "placements": len(typeset.placements),
            "overflow_10_columns": sum(
                item.overflow_level == 10 for item in typeset.placements
            ),
            "overflow_20_columns": overflow_20,
            "manual_review_required": bool(overflow_20),
            "failed_columns": failed,
            "all_columns_mapped": len(typeset.placements) == len(contract.columns),
            "all_columns_changed": all(
                item.region_change is not None and item.region_change.changed
                for item in typeset.audits
            ),
            "all_layout_boxes_at_least_80pct": all(
                item.layout_width_percent_of_source >= 80.0
                and item.layout_height_percent_of_source >= 80.0
                for item in typeset.placements
            ),
            "all_complete_units_use_uniform_base_font_size": all(
                item["uniform_base_font_size"] for item in unit_font_audit
            ),
            "all_complete_unit_centers_preserved": all(
                item["center_preserved"] for item in unit_font_audit
            ),
        },
        "timing": {
            "typesetting_seconds": round(typeset.typesetting_seconds, 6),
            "pixel_audit_seconds": round(pixel_audit_seconds, 6),
            "pixel_audit_time_budget_seconds": 60,
            "within_pixel_audit_time_budget": pixel_audit_seconds <= 60.0,
            "time_budget_applies_to": "pixel_audit_only",
        },
        "hard_limit_pixel_audit": {
            "changed_outside_layout_bbox_20pct_union": outside_hard_pixels,
            "hard_limit_preserved": outside_hard_pixels == 0,
        },
        "placements": details["placements"],
        "unit_font_audit": unit_font_audit,
        "pixel_checks": details["pixel_checks"],
        "outputs": {
            "typeset": {
                "path": str(output),
                "sha256": sha256_file(output),
            }
        },
    }


def _infer_column_styles(image: Image.Image, contract) -> Dict[str, str]:
    classifier = LocalStatisticsStyleClassifier()
    geometry = build_column_masks(contract)
    return {
        column.audit.column_id: (
            "bubble"
            if classifier.classify(image, column).strategy == "solid_fill"
            else "artwork"
        )
        for column in geometry.columns
    }


def _hard_layout_union(contract) -> Image.Image:
    mask = Image.new("L", contract.image_size, 0)
    draw = ImageDraw.Draw(mask)
    for column in contract.columns:
        x1, y1, x2, y2 = compute_layout_bbox(
            column.image_bbox,
            20,
            contract.image_size,
        )
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    return mask


def _binary_difference_mask(difference: Image.Image) -> Image.Image:
    bands = [
        band.point(lambda value: 255 if value else 0, mode="L")
        for band in difference.convert("RGB").split()
    ]
    changed = bands[0]
    for band in bands[1:]:
        changed = ImageChops.lighter(changed, band)
    return changed


def _png_path(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() != ".png":
        raise ValueError(f"{label} must use .png for lossless pixel auditing")
    return resolved


def _image_bytes_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()
