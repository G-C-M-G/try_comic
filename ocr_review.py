from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from background_ocr import attach_manual_review_queue, expand_bbox_10pct, polygon_bbox


VALID_DECISIONS = {"pending", "approved", "corrected", "rejected"}
VALID_ROLES = {"text", "dialogue", "narration", "uncertain", "non_dialogue", "sfx", "ignore"}
_TERMINAL_RE = re.compile(r"(?:[。！？!?]+|…+|\.{2,})[」』）)]*\s*$")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_review_template(result: dict, source_ocr: Path) -> dict:
    items = []
    for item in result.get("manual_review_queue", []):
        items.append({
            **deepcopy(item),
            "decision": "pending",
            "corrected_text": None,
            "corrected_polygon": None,
            "corrected_image_bbox": None,
            "use_paddle_geometry": False,
            "note": "",
        })
    return {
        "schema_version": 1,
        "source_ocr": str(source_ocr.resolve()),
        "source_ocr_sha256": _sha256(source_ocr),
        "instructions": {
            "approved": "Keep the current text and geometry.",
            "corrected": "Set corrected_text and/or corrected geometry.",
            "rejected": "Remove this false-positive OCR region.",
        },
        "items": items,
    }


def build_review_template_v2(result: dict, source_ocr: Path) -> dict:
    queue = {item["column_id"]: item for item in result.get("manual_review_queue", [])}
    items = []
    for column in sorted(result.get("columns", []), key=lambda item: item["selection_order"]):
        risk = queue.get(column["id"], {})
        items.append({
            "column_id": column["id"],
            "decision": "pending" if risk.get("status") in {"pending", "bypassed"} else "approved",
            "current_text": column["text"],
            "current_polygon": column["polygon"],
            "current_image_bbox": column["image_bbox"],
            "corrected_text": None,
            "corrected_polygon": None,
            "corrected_image_bbox": None,
            "role": column.get("role", "text"),
            "container_id": column.get("container_id"),
            "break_after": bool(column.get("break_after", False)),
            "join_after": bool(column.get("join_after", False)),
            "selection_order": int(column["selection_order"]),
            "reason_codes": list(risk.get("reason_codes", [])),
            "mangaocr_readings": list(column.get("readings", [])),
            "paddleocr_candidates": list(column.get("paddle_validation", [])),
            "photos_text": column.get("photos_text"),
            "note": "",
        })
    recoveries = []
    for index, candidate in enumerate(result.get("recovery_candidates", []), 1):
        recoveries.append({
            "candidate_id": candidate.get("id") or f"recovery_{index:03d}",
            # A recovery candidate represents text seen by only one geometry or
            # inventory source.  It must never disappear without an explicit
            # human decision.
            "decision": "pending",
            "text": candidate.get("text", ""),
            "polygon": candidate.get("polygon"),
            "image_bbox": candidate.get("image_bbox"),
            "role": "uncertain",
            "container_id": None,
            "break_after": False,
            "join_after": False,
            "selection_order": None,
            "reason": candidate.get("reason", "recovery-candidate"),
            "note": "",
        })
    return {
        "schema_version": 2,
        "source_ocr": str(source_ocr.resolve()),
        "source_ocr_sha256": _sha256(source_ocr),
        "items": items,
        "recoveries": recoveries,
        "additions": [],
    }


def prepare_review(
    source: Path,
    output_ocr: Path,
    review_path: Path,
    schema_version: int = 2,
) -> None:
    result = attach_manual_review_queue(_read_json(source))
    _write_json(output_ocr, result)
    template = (
        build_review_template_v2(result, output_ocr)
        if schema_version == 2
        else build_review_template(result, output_ocr)
    )
    _write_json(review_path, template)


def _validated_bbox(value, width: int, height: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("corrected_image_bbox must contain four coordinates")
    x1, y1, x2, y2 = (int(number) for number in value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"corrected_image_bbox is outside the image: {value}")
    return [x1, y1, x2, y2]


def _validated_polygon(value, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError("corrected_polygon must contain at least four points")
    points = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("every corrected_polygon point must contain x and y")
        x, y = (int(number) for number in raw)
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"corrected_polygon point is outside the image: {raw}")
        points.append([x, y])
    return points


def _translation_query(columns: list[dict]) -> str:
    if len(columns) == 1:
        return columns[0]["text"]
    return "".join(f"<{index}>{column['text']}" for index, column in enumerate(columns, 1))


def _review_layout_region(column: dict) -> dict:
    return {
        "column_id": column["id"],
        "source_text": column["text"],
        "polygon": column["polygon"],
        "image_bbox": column["image_bbox"],
        "layout_bbox_10pct": column["layout_bbox_10pct"],
    }


def _rebuild_all_translation_units(result: dict) -> None:
    columns = sorted(result.get("columns", []), key=lambda item: int(item["selection_order"]))
    for index, column in enumerate(columns):
        column["selection_order"] = index
        column["panel_index"] = int(column.get("panel_index", 0) or 0)
    result["columns"] = columns
    result["reading_order"] = [column["id"] for column in columns]
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_key = None
    for column in columns:
        container = column.get("container_id") or f"unassigned:{column['id']}"
        key = (column["panel_index"], container)
        if current and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        current.append(column)
        if column.get("break_after") or (
            _TERMINAL_RE.search(column["text"]) and not column.get("join_after")
        ):
            groups.append(current)
            current = []
            current_key = None
    if current:
        groups.append(current)
    result["translation_units"] = [
        {
            "id": f"sentence_{index:03d}",
            "column_ids": [column["id"] for column in group],
            "source_text": "".join(column["text"] for column in group),
            "translation_query": _translation_query(group),
            "panel_index": group[0]["panel_index"],
            "container_id": group[0].get("container_id"),
            "grouping_basis": "container" if group[0].get("container_id") else "unassigned-column",
            "layout_regions": [_review_layout_region(column) for column in group],
        }
        for index, group in enumerate(groups, 1)
    ]


def _rebuild_translation_units(result: dict, rejected_ids: set[str]) -> None:
    columns = {item["id"]: item for item in result.get("columns", [])}
    result["reading_order"] = [
        column_id for column_id in result.get("reading_order", [])
        if column_id in columns and column_id not in rejected_ids
    ]
    rebuilt = []
    for unit in result.get("translation_units", []):
        unit_columns = [
            columns[column_id] for column_id in unit.get("column_ids", [])
            if column_id in columns and column_id not in rejected_ids
        ]
        if not unit_columns:
            continue
        item = dict(unit)
        item["column_ids"] = [column["id"] for column in unit_columns]
        item["source_text"] = "".join(column["text"] for column in unit_columns)
        item["translation_query"] = _translation_query(unit_columns)
        item["layout_regions"] = [{
            "column_id": column["id"],
            "source_text": column["text"],
            "polygon": column["polygon"],
            "image_bbox": column["image_bbox"],
            "layout_bbox_10pct": column["layout_bbox_10pct"],
        } for column in unit_columns]
        rebuilt.append(item)
    result["translation_units"] = rebuilt


def _correct_column_from_v2(
    column: dict,
    decision_item: dict,
    width: int,
    height: int,
) -> dict:
    updated = deepcopy(column)
    corrected_text = decision_item.get("corrected_text")
    if corrected_text is not None:
        corrected_text = str(corrected_text).strip()
        if not corrected_text:
            raise ValueError(f"corrected_text is empty for {column['id']}")
        updated["text"] = corrected_text
    corrected_polygon = decision_item.get("corrected_polygon")
    corrected_bbox = decision_item.get("corrected_image_bbox")
    if corrected_polygon is not None and corrected_bbox is not None:
        raise ValueError(f"set corrected_polygon or corrected_image_bbox, not both, for {column['id']}")
    if corrected_polygon is not None:
        updated["polygon"] = _validated_polygon(corrected_polygon, width, height)
        updated["image_bbox"] = _validated_bbox(polygon_bbox(updated["polygon"]), width, height)
    elif corrected_bbox is not None:
        updated["image_bbox"] = _validated_bbox(corrected_bbox, width, height)
        x1, y1, x2, y2 = updated["image_bbox"]
        updated["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    role = str(decision_item.get("role", updated.get("role", "text")))
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role for {column['id']}: {role}")
    updated["role"] = role
    updated["container_id"] = decision_item.get("container_id")
    updated["break_after"] = bool(decision_item.get("break_after", False))
    updated["join_after"] = bool(decision_item.get("join_after", False))
    if updated["break_after"] and updated["join_after"]:
        raise ValueError(f"break_after and join_after conflict for {column['id']}")
    updated["selection_order"] = int(decision_item.get("selection_order", updated.get("selection_order", 0)))
    updated["layout_bbox_10pct"] = expand_bbox_10pct(updated["image_bbox"], (width, height))
    return updated


def _candidate_column_v2(
    value: dict,
    column_id: str,
    width: int,
    height: int,
    fallback_order: int,
) -> dict:
    text = str(value.get("corrected_text") or value.get("text") or "").strip()
    if not text:
        raise ValueError(f"added column {column_id} requires text")
    polygon = value.get("corrected_polygon") or value.get("polygon")
    bbox = value.get("corrected_image_bbox") or value.get("image_bbox")
    if polygon is not None:
        polygon = _validated_polygon(polygon, width, height)
        bbox = _validated_bbox(polygon_bbox(polygon), width, height)
    elif bbox is not None:
        bbox = _validated_bbox(bbox, width, height)
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        raise ValueError(f"added column {column_id} requires polygon or image_bbox")
    role = str(value.get("role", "uncertain"))
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role for added column {column_id}: {role}")
    return {
        "id": column_id,
        "text": text,
        "polygon": polygon,
        "image_bbox": bbox,
        "layout_bbox_10pct": expand_bbox_10pct(bbox, (width, height)),
        "selection_order": int(value.get("selection_order", fallback_order) or fallback_order),
        "panel_index": int(value.get("panel_index", 0) or 0),
        "direction": value.get("direction", "vertical"),
        "role": role,
        "container_id": value.get("container_id"),
        "break_after": bool(value.get("break_after", False)),
        "join_after": bool(value.get("join_after", False)),
        "text_source": "manual-review-v2",
        "detection_sources": ["manual-review-v2"],
        "readings": [text],
        "recognition_confidence": 1.0,
        "cross_validation": {"decision": "manual-review-v2", "agrees": True},
        "review_required": False,
        "review_status": "corrected",
        "review_reasons": ["manually-added-column"],
    }


def _bbox_coverage(inner_value, outer_value) -> float:
    if not isinstance(inner_value, list) or len(inner_value) != 4:
        return 0.0
    if not isinstance(outer_value, list) or len(outer_value) != 4:
        return 0.0
    ix1, iy1, ix2, iy2 = (int(value) for value in inner_value)
    ox1, oy1, ox2, oy2 = (int(value) for value in outer_value)
    area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if area <= 0:
        return 0.0
    overlap = max(0, min(ix2, ox2) - max(ix1, ox1)) * max(
        0, min(iy2, oy2) - max(iy1, oy1)
    )
    return overlap / area


def _apply_review_v2(source: Path, review: dict, output: Path) -> None:
    if review.get("source_ocr_sha256") != _sha256(source):
        raise ValueError("review file does not match the OCR JSON hash")
    result = _read_json(source)
    width = int(result["source"]["width"])
    height = int(result["source"]["height"])
    original = {column["id"]: column for column in result.get("columns", [])}
    dialogue_only = bool(result.get("capture", {}).get("dialogue_only"))
    items = review.get("items", [])
    item_ids = [item.get("column_id") for item in items]
    if len(item_ids) != len(set(item_ids)) or set(item_ids) != set(original):
        raise ValueError("schema 2 review must contain every current column exactly once")
    retained = []
    ignored = list(result.get("ignored_regions", []))
    audit = []
    restore_targets: list[tuple[str, list[int]]] = []
    for item in items:
        column_id = item["column_id"]
        decision = item.get("decision", "pending")
        if decision not in VALID_DECISIONS or decision == "pending":
            raise ValueError(f"manual review is still pending or invalid for {column_id}")
        column = _correct_column_from_v2(original[column_id], item, width, height)
        if column["role"] == "uncertain":
            raise ValueError(f"manual review must resolve the dialogue role for {column_id}")
        if dialogue_only and column["role"] == "text":
            raise ValueError(
                f"dialogue-only review must classify {column_id} as dialogue, narration, or non-dialogue"
            )
        if decision == "rejected" or column["role"] in {"non_dialogue", "sfx", "ignore"}:
            ignored.append({**column, "reason": "manual-review-rejected-or-non-dialogue"})
            status = "rejected"
        else:
            status = "corrected" if decision == "corrected" else "approved"
            column["review_required"] = False
            column["review_status"] = status
            column["review_note"] = str(item.get("note", ""))
            retained.append(column)
            if item.get("restore_ignored"):
                restore_targets.append((column["id"], column["image_bbox"]))
        audit.append((column, status, item))

    recoveries = list(review.get("recoveries", []))
    for recovery in recoveries:
        decision = recovery.get("decision", "pending")
        candidate_id = recovery.get("candidate_id", "unknown-recovery")
        if decision == "pending":
            raise ValueError(f"manual review is still pending for {candidate_id}")
        if decision not in {"added", "rejected"}:
            raise ValueError(
                f"invalid recovery decision for {candidate_id}: {decision}; "
                "use added or rejected"
            )

    next_id = len(original) + 1
    additions = list(review.get("additions", []))
    additions.extend(
        item for item in recoveries if item.get("decision") == "added"
    )
    for value in additions:
        if value.get("decision", "added") not in {"added", "corrected", "approved"}:
            continue
        column_id = str(value.get("column_id") or f"column_added_{next_id:03d}")
        if column_id in original or any(column["id"] == column_id for column in retained):
            raise ValueError(f"duplicate added column id: {column_id}")
        column = _candidate_column_v2(value, column_id, width, height, len(retained))
        retained.append(column)
        if value.get("restore_ignored"):
            restore_targets.append((column["id"], column["image_bbox"]))
        audit.append((column, "corrected", value))
        next_id += 1

    retained.sort(key=lambda column: int(column["selection_order"]))
    restored_ignored = []
    retained_ignored = []
    for ignored_item in ignored:
        restored_by = next((
            column_id
            for column_id, bbox in restore_targets
            if _bbox_coverage(ignored_item.get("image_bbox"), bbox) >= 0.5
        ), None)
        if restored_by:
            restored_ignored.append({
                **ignored_item,
                "restored_by_column_id": restored_by,
                "restore_reason": "manual-review-v2",
            })
        else:
            retained_ignored.append(ignored_item)
    result["columns"] = retained
    result["ignored_regions"] = retained_ignored
    result["restored_ignored_regions"] = [
        *result.get("restored_ignored_regions", []),
        *restored_ignored,
    ]
    _rebuild_all_translation_units(result)
    reviewed_queue = []
    retained_ids = {column["id"] for column in result["columns"]}
    for column, status, item in audit:
        reviewed_queue.append({
            "column_id": column["id"],
            "status": status,
            "reason_codes": list(item.get("reason_codes") or column.get("review_reasons") or ["manual-review-v2"]),
            "current_text": original.get(column["id"], column).get("text", column["text"]),
            "polygon": column["polygon"],
            "image_bbox": column["image_bbox"],
            "layout_bbox_10pct": column["layout_bbox_10pct"],
            "resolved_text": column["text"] if column["id"] in retained_ids else None,
            "resolved_polygon": column["polygon"] if column["id"] in retained_ids else None,
            "resolved_image_bbox": column["image_bbox"] if column["id"] in retained_ids else None,
            "resolved_layout_bbox_10pct": column["layout_bbox_10pct"] if column["id"] in retained_ids else None,
            "note": str(item.get("note", "")),
        })
    result["manual_review_queue"] = reviewed_queue
    result["recovery_candidates"] = []
    result["recovery_review"] = [{
        "candidate_id": item.get("candidate_id"),
        "status": item.get("decision"),
        "note": str(item.get("note", "")),
    } for item in recoveries]
    result["manual_review"] = {
        **result.get("manual_review", {}),
        "enabled": True,
        "status": "complete" if reviewed_queue else "not_required",
        "translation_blocked": False,
        "candidate_count": len(reviewed_queue),
        "pending_count": 0,
        "bypassed_count": 0,
        "reviewed_count": len(reviewed_queue),
        "approved_count": sum(status == "approved" for _, status, _ in audit),
        "corrected_count": sum(status == "corrected" for _, status, _ in audit),
        "rejected_count": sum(status == "rejected" for _, status, _ in audit),
        "review_schema_version": 2,
    }
    _write_json(output, result)


def apply_review(source: Path, review_path: Path, output: Path) -> None:
    review = _read_json(review_path)
    if review.get("schema_version") == 2:
        _apply_review_v2(source, review, output)
        return
    if review.get("source_ocr_sha256") != _sha256(source):
        raise ValueError("review file does not match the OCR JSON hash")
    result = _read_json(source)
    columns = {item["id"]: item for item in result.get("columns", [])}
    pending_ids = {
        item["column_id"] for item in result.get("manual_review_queue", [])
        if item.get("status") in {"pending", "bypassed"}
    }
    decision_items = review.get("items", [])
    decision_ids = [item.get("column_id") for item in decision_items]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("review file contains duplicate column decisions")
    decisions = {item.get("column_id"): item for item in decision_items}
    missing = pending_ids - decisions.keys()
    if missing:
        raise ValueError(f"review decisions are missing columns: {sorted(missing)}")

    width = int(result["source"]["width"])
    height = int(result["source"]["height"])
    rejected_ids: set[str] = set()
    reviewed_queue = []
    for queue_item in result.get("manual_review_queue", []):
        column_id = queue_item["column_id"]
        decision_item = decisions.get(column_id, {})
        decision = decision_item.get("decision", "pending")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"invalid decision for {column_id}: {decision}")
        if decision == "pending":
            raise ValueError(f"manual review is still pending for {column_id}")
        column = columns[column_id]
        if decision == "rejected":
            rejected_ids.add(column_id)
        elif decision == "corrected":
            corrected_text = decision_item.get("corrected_text")
            corrected_polygon = decision_item.get("corrected_polygon")
            corrected_bbox = decision_item.get("corrected_image_bbox")
            use_paddle_geometry = bool(decision_item.get("use_paddle_geometry"))
            if use_paddle_geometry:
                if corrected_polygon is not None or corrected_bbox is not None:
                    raise ValueError(
                        f"use_paddle_geometry cannot be combined with corrected geometry for {column_id}"
                    )
                corrected_polygon = decision_item.get("paddle_polygon")
                corrected_bbox = (
                    None if corrected_polygon is not None
                    else decision_item.get("paddle_image_bbox")
                )
                if corrected_polygon is None and corrected_bbox is None:
                    raise ValueError(f"no PaddleOCR geometry is available for {column_id}")
            normalized_text = (
                str(corrected_text).strip() if corrected_text is not None else ""
            )
            if corrected_polygon is not None and corrected_bbox is not None:
                raise ValueError(
                    f"set corrected_polygon or corrected_image_bbox, not both, for {column_id}"
                )
            if not normalized_text and corrected_polygon is None and corrected_bbox is None:
                raise ValueError(f"corrected decision has no correction for {column_id}")
            if corrected_text is not None:
                if not normalized_text:
                    raise ValueError(f"corrected_text is empty for {column_id}")
                column["text"] = normalized_text
            if corrected_polygon is not None:
                column["polygon"] = _validated_polygon(corrected_polygon, width, height)
                column["image_bbox"] = _validated_bbox(
                    polygon_bbox(column["polygon"]), width, height
                )
            if corrected_bbox is not None:
                column["image_bbox"] = _validated_bbox(corrected_bbox, width, height)
                if corrected_polygon is None:
                    x1, y1, x2, y2 = column["image_bbox"]
                    column["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            column["layout_bbox_10pct"] = expand_bbox_10pct(
                column["image_bbox"], (width, height)
            )
        if decision != "rejected":
            column["review_required"] = False
            column["review_status"] = decision
            column["review_note"] = str(decision_item.get("note", ""))
        reviewed_queue.append({
            **queue_item,
            "status": decision,
            "resolved_text": None if decision == "rejected" else column.get("text"),
            "resolved_polygon": None if decision == "rejected" else column.get("polygon"),
            "resolved_image_bbox": (
                None if decision == "rejected" else column.get("image_bbox")
            ),
            "resolved_layout_bbox_10pct": (
                None if decision == "rejected" else column.get("layout_bbox_10pct")
            ),
            "note": str(decision_item.get("note", "")),
        })

    result["columns"] = [
        item for item in result.get("columns", []) if item["id"] not in rejected_ids
    ]
    _rebuild_translation_units(result, rejected_ids)
    result["manual_review_queue"] = reviewed_queue
    result["manual_review"] = {
        **result.get("manual_review", {}),
        "status": "complete",
        "translation_blocked": False,
        "pending_count": 0,
        "reviewed_count": len(reviewed_queue),
        "approved_count": sum(item["status"] == "approved" for item in reviewed_queue),
        "corrected_count": sum(item["status"] == "corrected" for item in reviewed_queue),
        "rejected_count": sum(item["status"] == "rejected" for item in reviewed_queue),
    }
    _write_json(output, result)


def apply_review_patch(
    source: Path,
    patch_path: Path,
    review_path: Path,
    output: Path,
) -> None:
    """Materialize and apply a compact, hash-bound schema 2 review patch."""
    result = _read_json(source)
    patch = _read_json(patch_path)
    source_hash = _sha256(source)
    expected_hash = str(patch.get("source_ocr_sha256", "")).lower()
    if expected_hash and expected_hash != source_hash.lower():
        raise ValueError("review patch does not match the OCR JSON hash")

    review = build_review_template_v2(result, source)
    item_by_id = {item["column_id"]: item for item in review["items"]}
    changes: dict[str, dict] = {}
    next_order = 0
    group_defaults = dict(patch.get("group_defaults", {}))
    for group_index, group in enumerate(patch.get("groups", []), 1):
        group_id = group.get("container_id") or f"review_group_{group_index:03d}"
        raw_columns = list(group.get("columns", []))
        for index, raw_column in enumerate(raw_columns):
            if isinstance(raw_column, str):
                column_id = raw_column
                overrides = {}
            else:
                overrides = dict(raw_column)
                column_id = str(overrides.pop("id"))
            if column_id in changes:
                raise ValueError(f"review patch groups contain duplicate column: {column_id}")
            change = {
                **group_defaults,
                "container_id": group_id,
                "selection_order": next_order,
                **overrides,
            }
            if index == len(raw_columns) - 1 and "break_after" not in change:
                change["break_after"] = bool(group.get("break_after", True))
            changes[column_id] = change
            next_order += 1
    for column_id, raw_change in patch.get("columns", {}).items():
        changes[column_id] = {**changes.get(column_id, {}), **dict(raw_change)}
    if patch.get("require_all_columns") and set(changes) != set(item_by_id):
        missing = sorted(set(item_by_id) - set(changes))
        extra = sorted(set(changes) - set(item_by_id))
        raise ValueError(f"review patch column set mismatch; missing={missing}, extra={extra}")
    for column_id, raw_change in changes.items():
        if column_id not in item_by_id:
            raise ValueError(f"review patch references unknown column: {column_id}")
        change = dict(raw_change)
        expected_text = change.pop("expected_text", None)
        if expected_text is not None and item_by_id[column_id]["current_text"] != expected_text:
            raise ValueError(
                f"review patch text mismatch for {column_id}: "
                f"expected {expected_text!r}, got {item_by_id[column_id]['current_text']!r}"
            )
        item_by_id[column_id].update(change)

    recovery_by_id = {
        item["candidate_id"]: item for item in review.get("recoveries", [])
    }
    recovery_changes = patch.get("recoveries", {})
    for candidate_id, change in recovery_changes.items():
        if candidate_id not in recovery_by_id:
            raise ValueError(f"review patch references unknown recovery: {candidate_id}")
        recovery_by_id[candidate_id].update(change)
    review["additions"] = list(patch.get("additions", []))
    _write_json(review_path, review)
    apply_review(source, review_path, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and apply manual OCR review decisions.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Attach a review queue and export a decision template.")
    prepare.add_argument("ocr_json", type=Path)
    prepare.add_argument("--output-ocr", type=Path)
    prepare.add_argument("--review", type=Path)
    prepare.add_argument("--schema-version", type=int, choices=[1, 2], default=2)
    apply = commands.add_parser("apply", help="Apply completed human decisions to an OCR JSON.")
    apply.add_argument("ocr_json", type=Path)
    apply.add_argument("review_json", type=Path)
    apply.add_argument("output", type=Path)
    patch = commands.add_parser(
        "apply-patch",
        help="Materialize and apply a compact hash-bound schema 2 review patch.",
    )
    patch.add_argument("ocr_json", type=Path)
    patch.add_argument("patch_json", type=Path)
    patch.add_argument("output", type=Path)
    patch.add_argument("--review", type=Path)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        output_ocr = args.output_ocr or args.ocr_json
        review_path = args.review or output_ocr.with_suffix(".review.json")
        prepare_review(args.ocr_json, output_ocr, review_path, args.schema_version)
        print(output_ocr)
        print(review_path)
        return 0
    if args.command == "apply-patch":
        review_path = args.review or args.output.with_suffix(".review.json")
        apply_review_patch(args.ocr_json, args.patch_json, review_path, args.output)
        print(review_path)
        print(args.output)
        return 0
    apply_review(args.ocr_json, args.review_json, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
