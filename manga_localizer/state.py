from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from PIL import Image

from .errors import StateError
from .ocr_contract import OCRContract, canonical_sha256, sha256_file
from .review import ReviewApplication
from .typesetter import TranslationArtifact


STATE_ARTIFACT = "manga-localizer-state"
STATE_VERSION = 1


@dataclass(frozen=True)
class EditableState:
    path: Path
    data: Mapping[str, Any]
    image_path: Path
    erased_image_path: Path
    translation: TranslationArtifact
    styles_by_column: Mapping[str, str]
    invalidated_alignment_units: Tuple[str, ...]


def write_pipeline_state(
    state_path: Path,
    application: ReviewApplication,
    *,
    image_path: Path,
    erased_image_path: Path,
    mask_path: Path,
    erasure_audit_path: Path,
    translation_path: Path,
    typeset_audit_path: Path,
    output_path: Path,
) -> Path:
    output = _state_json_path(state_path)
    erasure = _read_json(erasure_audit_path, "erasure audit")
    translation = _read_json(translation_path, "translation artifact")
    typeset = _read_json(typeset_audit_path, "typeset audit")
    if erasure.get("stage") != "C" or translation.get("stage") != "D":
        raise StateError("state inputs must be stage C and stage D artifacts")
    if typeset.get("stage") != "E":
        raise StateError("state input must include a stage E typeset audit")

    contract = application.contract
    erasure_by_column = {
        str(item["column_id"]): item for item in erasure.get("columns", [])
    }
    translation_by_column = {
        str(item["column_id"]): item for item in translation.get("columns", [])
    }
    placement_by_column = {
        str(item["column_id"]): item for item in typeset.get("placements", [])
    }
    expected_columns = set(contract.columns_by_id)
    if not all(
        set(mapping) == expected_columns
        for mapping in (
            erasure_by_column,
            translation_by_column,
            placement_by_column,
        )
    ):
        raise StateError("stage artifacts do not cover the same retained OCR columns")

    units = []
    for item in translation.get("units", []):
        units.append(
            {
                "unit_id": item["unit_id"],
                "column_ids": list(item["column_ids"]),
                "source_text": item["source_text"],
                "translation_query": item["translation_query"],
                "translated_text": item["translated_text"],
                "column_segments": [
                    {
                        "column_id": segment["column_id"],
                        "text": segment["text"],
                    }
                    for segment in item.get("aligned_column_segments", [])
                ],
                "translation": {
                    "backend": item.get("backend"),
                    "model": item.get("model"),
                    "cache_key": item.get("cache_key"),
                    "status": item.get("translation_status"),
                    "failure_reason": item.get("failure_reason"),
                },
            }
        )

    payload: Dict[str, Any] = {
        "artifact": STATE_ARTIFACT,
        "artifact_version": STATE_VERSION,
        "stage": "F",
        "status": typeset.get("status"),
        "editable_fields": [
            "units[].translated_text",
            "units[].column_segments",
        ],
        "source": {
            "ocr": _path_reference(contract.path, output.parent),
            "ocr_sha256": contract.source_sha256,
            "effective_contract_sha256": contract.effective_sha256,
            "image": _path_reference(image_path, output.parent),
            "image_sha256": sha256_file(Path(image_path)),
            "width": contract.source.width,
            "height": contract.source.height,
        },
        "review": {
            "mode": application.mode,
            "decision_file": (
                _path_reference(application.decision_path, output.parent)
                if application.decision_path
                else None
            ),
            "decision_sha256": application.decision_sha256,
            "pending_count": application.pending_count,
            "bypassed_count": application.bypassed_count,
        },
        "policies": {
            "complete_unit_translation": True,
            "uniform_base_font_size_per_complete_unit": True,
            "terminal_punctuation_max_layout_units": 1,
            "terminal_punctuation_smaller_font_is_allowed": True,
            "minimum_layout_percent_of_source_bbox": 80,
            "soft_layout_overflow_percent": 10,
            "hard_layout_overflow_percent": 20,
            "preserve_complete_unit_center": bool(
                typeset.get("layout_policy", {}).get(
                    "preserve_complete_unit_center", True
                )
            ),
        },
        "layout_config": dict(typeset.get("config", {})),
        "unit_audits": list(typeset.get("unit_font_audit", [])),
        "translation_revision_sha256": _translation_revision(units),
        "units": units,
        "columns": [],
        "assets": {
            "erased": _asset(erased_image_path, output.parent),
            "mask": _asset(mask_path, output.parent),
            "erasure_audit": _asset(erasure_audit_path, output.parent),
            "translation": _asset(translation_path, output.parent),
            "typeset_audit": _asset(typeset_audit_path, output.parent),
            "final": _asset(output_path, output.parent),
        },
        "history": [
            {
                "action": "full_pipeline",
                "status": typeset.get("status"),
                "translation_revision_sha256": _translation_revision(units),
            }
        ],
    }
    for column_id in contract.reading_order:
        column = contract.columns_by_id[column_id]
        erased = erasure_by_column[column_id]
        translated = translation_by_column[column_id]
        placement = placement_by_column[column_id]
        payload["columns"].append(
            {
                "column_id": column_id,
                "unit_id": translated["unit_id"],
                "source_text": column.text,
                "translated_text_segment": placement.get("text_segment"),
                "polygon": [list(point) for point in column.polygon],
                "mask": {
                    **dict(erased.get("mask", {})),
                    "combined_mask_asset": "assets.mask",
                },
                "erasure": {
                    "strategy": erased.get("strategy"),
                    "backend": erased.get("eraser", {}).get("backend"),
                    "status": erased.get("status"),
                    "changed_pixels": erased.get("changed_pixels"),
                },
                "layout": {
                    "source_bbox": list(column.image_bbox),
                    "layout_bbox_10pct": placement.get("layout_bbox_10pct"),
                    "layout_bbox_20pct": placement.get("layout_bbox_20pct"),
                    "selected_bbox": placement.get("layout_bbox"),
                    "minimum_layout_percent": placement.get(
                        "minimum_layout_percent", 80
                    ),
                    "layout_width_percent_of_source": placement.get(
                        "layout_width_percent_of_source"
                    ),
                    "layout_height_percent_of_source": placement.get(
                        "layout_height_percent_of_source"
                    ),
                    "overflow_level": placement.get("overflow_level"),
                    "overflow_width_percent": placement.get(
                        "overflow_width_percent"
                    ),
                    "overflow_height_percent": placement.get(
                        "overflow_height_percent"
                    ),
                    "overflow_reason": placement.get("overflow_reason"),
                    "direction": placement.get("direction"),
                    "font": placement.get("font"),
                    "font_size": placement.get("font_size"),
                    "character_spacing": placement.get("character_spacing"),
                    "line_spacing": placement.get("line_spacing"),
                    "text_color": placement.get("text_color"),
                    "outline_color": placement.get("outline_color"),
                    "outline_width": placement.get("outline_width"),
                    "style": placement.get("style"),
                    "terminal_punctuation_font_ratio": placement.get(
                        "terminal_punctuation_font_ratio",
                        typeset.get("config", {}).get(
                            "terminal_punctuation_font_ratio", 0.68
                        ),
                    ),
                },
                "processing_status": placement.get("status"),
                "failure_reason": (
                    placement.get("overflow_reason")
                    if placement.get("status") != "layout_ready"
                    else None
                ),
            }
        )
    _write_json(output, payload)
    return output


def load_editable_state(
    state_path: Path,
    contract: OCRContract,
    *,
    explicit_image: Optional[Path] = None,
    explicit_erased_image: Optional[Path] = None,
) -> EditableState:
    path = _state_json_path(state_path, must_exist=True)
    data = _read_json(path, "state")
    if (
        data.get("artifact") != STATE_ARTIFACT
        or data.get("artifact_version") != STATE_VERSION
    ):
        raise StateError("unsupported state artifact")
    source = data.get("source") or {}
    if source.get("ocr_sha256") != contract.source_sha256:
        raise StateError("state OCR file hash does not match --ocr")
    if source.get("effective_contract_sha256") != contract.effective_sha256:
        raise StateError("state effective OCR contract does not match --ocr/review")
    image_path = (
        Path(explicit_image).resolve()
        if explicit_image is not None
        else _resolve_reference(source.get("image"), path.parent, "source image")
    )
    _verify_asset(
        image_path,
        source.get("image_sha256"),
        "source image",
    )
    erased_asset = (data.get("assets") or {}).get("erased") or {}
    erased_path = (
        Path(explicit_erased_image).resolve()
        if explicit_erased_image is not None
        else _resolve_reference(erased_asset.get("path"), path.parent, "erased image")
    )
    _verify_asset(erased_path, erased_asset.get("sha256"), "erased image")
    for label, candidate in (("source image", image_path), ("erased image", erased_path)):
        try:
            with Image.open(candidate) as opened:
                if opened.size != contract.image_size:
                    raise StateError(f"{label} dimensions differ from the OCR contract")
        except OSError as error:
            raise StateError(f"cannot read {label}: {error}") from error

    expected_units = {unit.id: unit for unit in contract.translation_units}
    raw_units = data.get("units")
    if not isinstance(raw_units, list) or {
        str(item.get("unit_id")) for item in raw_units if isinstance(item, dict)
    } != set(expected_units):
        raise StateError("state does not contain every OCR translation unit")
    translations: Dict[str, str] = {}
    segments_by_column: Dict[str, str] = {}
    invalidated = []
    for item in raw_units:
        unit_id = str(item["unit_id"])
        unit = expected_units[unit_id]
        translated = item.get("translated_text")
        if not isinstance(translated, str) or not translated.strip():
            raise StateError(f"state unit {unit_id} has an empty translation")
        translations[unit_id] = translated
        raw_segments = item.get("column_segments") or []
        segment_map = {
            str(segment.get("column_id")): str(segment.get("text") or "")
            for segment in raw_segments
            if isinstance(segment, dict)
        }
        if (
            set(segment_map) == set(unit.column_ids)
            and all(segment_map[column_id] for column_id in unit.column_ids)
            and "".join(segment_map[column_id] for column_id in unit.column_ids)
            == translated
        ):
            segments_by_column.update(segment_map)
        elif raw_segments:
            invalidated.append(unit_id)

    raw_columns = data.get("columns")
    if not isinstance(raw_columns, list) or {
        str(item.get("column_id"))
        for item in raw_columns
        if isinstance(item, dict)
    } != set(contract.columns_by_id):
        raise StateError("state does not contain every retained OCR column")
    styles = {}
    for item in raw_columns:
        style = str((item.get("layout") or {}).get("style") or "artwork")
        styles[str(item["column_id"])] = (
            style if style in {"bubble", "artwork"} else "artwork"
        )
    translation = TranslationArtifact(
        path=path,
        data=data,
        translations_by_unit=translations,
        segments_by_column=segments_by_column,
    )
    return EditableState(
        path=path,
        data=data,
        image_path=image_path,
        erased_image_path=erased_path,
        translation=translation,
        styles_by_column=styles,
        invalidated_alignment_units=tuple(invalidated),
    )


def update_state_after_reembed(
    editable: EditableState,
    *,
    typeset_audit_path: Path,
    output_path: Path,
    state_output_path: Optional[Path] = None,
) -> Path:
    target = _state_json_path(state_output_path or editable.path)
    data = json.loads(json.dumps(editable.data, ensure_ascii=False))
    audit = _read_json(typeset_audit_path, "typeset audit")
    if target.parent != editable.path.parent:
        _rebase_state_paths(data, editable.path.parent, target.parent)
    placement_by_column = {
        str(item["column_id"]): item for item in audit.get("placements", [])
    }
    for unit in data["units"]:
        if unit["unit_id"] in editable.invalidated_alignment_units:
            unit["column_segments"] = []
    for column in data["columns"]:
        placement = placement_by_column[str(column["column_id"])]
        column["translated_text_segment"] = placement.get("text_segment")
        layout = column["layout"]
        layout.update(
            {
                "selected_bbox": placement.get("layout_bbox"),
                "overflow_level": placement.get("overflow_level"),
                "overflow_width_percent": placement.get("overflow_width_percent"),
                "overflow_height_percent": placement.get("overflow_height_percent"),
                "overflow_reason": placement.get("overflow_reason"),
                "font": placement.get("font"),
                "font_size": placement.get("font_size"),
                "character_spacing": placement.get("character_spacing"),
                "line_spacing": placement.get("line_spacing"),
                "text_color": placement.get("text_color"),
                "outline_color": placement.get("outline_color"),
                "outline_width": placement.get("outline_width"),
                "style": placement.get("style"),
            }
        )
        column["processing_status"] = placement.get("status")
        column["failure_reason"] = (
            placement.get("overflow_reason")
            if placement.get("status") != "layout_ready"
            else None
        )
    data["status"] = audit.get("status")
    data["layout_config"] = dict(audit.get("config", {}))
    data["unit_audits"] = list(audit.get("unit_font_audit", []))
    data["policies"]["preserve_complete_unit_center"] = bool(
        audit.get("layout_policy", {}).get(
            "preserve_complete_unit_center", True
        )
    )
    data["translation_revision_sha256"] = _translation_revision(data["units"])
    data["assets"]["typeset_audit"] = _asset(typeset_audit_path, target.parent)
    data["assets"]["final"] = _asset(output_path, target.parent)
    data.setdefault("history", []).append(
        {
            "action": "state_reembed",
            "status": audit.get("status"),
            "translation_revision_sha256": data["translation_revision_sha256"],
            "invalidated_alignment_units": list(
                editable.invalidated_alignment_units
            ),
        }
    )
    _write_json(target, data)
    return target


def _translation_revision(units) -> str:
    return canonical_sha256(
        {
            "units": [
                {
                    "unit_id": item["unit_id"],
                    "translated_text": item["translated_text"],
                    "column_segments": item.get("column_segments") or [],
                }
                for item in units
            ]
        }
    )


def _rebase_state_paths(data: Dict[str, Any], old_base: Path, new_base: Path) -> None:
    source = data.get("source") or {}
    for field in ("ocr", "image"):
        if source.get(field):
            resolved = _resolve_reference(source[field], old_base, field)
            source[field] = _path_reference(resolved, new_base)
    review = data.get("review") or {}
    if review.get("decision_file"):
        resolved = _resolve_reference(
            review["decision_file"], old_base, "review decision"
        )
        review["decision_file"] = _path_reference(resolved, new_base)
    for asset in (data.get("assets") or {}).values():
        if isinstance(asset, dict) and asset.get("path"):
            resolved = _resolve_reference(asset["path"], old_base, "state asset")
            asset["path"] = _path_reference(resolved, new_base)


def _asset(path: Path, relative_to: Path) -> Dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "path": _path_reference(resolved, relative_to),
        "sha256": sha256_file(resolved),
    }


def _verify_asset(path: Path, expected_sha256: Any, label: str) -> None:
    if not path.is_file():
        raise StateError(f"{label} does not exist: {path}")
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise StateError(f"{label} hash differs from state")


def _path_reference(path: Path, relative_to: Path) -> str:
    resolved = Path(path).resolve()
    base = Path(relative_to).resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(resolved)


def _resolve_reference(value: Any, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StateError(f"state has no {label} path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return candidate.resolve()


def _state_json_path(path: Path, *, must_exist: bool = False) -> Path:
    resolved = Path(path).resolve()
    if not resolved.name.endswith(".state.json"):
        raise StateError("state path must end with .state.json")
    if must_exist and not resolved.is_file():
        raise StateError(f"state does not exist: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    candidate = Path(path).resolve()
    try:
        data = json.loads(candidate.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"cannot read {label} {candidate}: {error}") from error
    if not isinstance(data, dict):
        raise StateError(f"{label} root must be an object")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
