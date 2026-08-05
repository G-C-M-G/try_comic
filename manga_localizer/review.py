from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from .errors import ReviewBlockedError, ReviewValidationError
from .layout_bounds import compute_layout_bbox
from .ocr_contract import OCRContract, contract_from_data


VALID_DECISIONS = {"pending", "approved", "corrected", "rejected"}


@dataclass(frozen=True)
class ReviewApplication:
    contract: OCRContract
    mode: str
    decision_path: Optional[Path]
    decision_sha256: Optional[str]
    approved_count: int
    corrected_count: int
    rejected_count: int
    pending_count: int
    bypassed_count: int

    @property
    def translation_allowed(self) -> bool:
        return not self.contract.manual_review.translation_blocked


def unchanged_review(contract: OCRContract) -> ReviewApplication:
    statuses = [item.status for item in contract.manual_review_queue]
    return ReviewApplication(
        contract=contract,
        mode="source",
        decision_path=None,
        decision_sha256=None,
        approved_count=statuses.count("approved"),
        corrected_count=statuses.count("corrected"),
        rejected_count=statuses.count("rejected"),
        pending_count=statuses.count("pending"),
        bypassed_count=statuses.count("bypassed"),
    )


def apply_review_decisions(
    contract: OCRContract,
    review_path: Path,
    *,
    require_complete: bool = False,
) -> ReviewApplication:
    path = Path(review_path).resolve()
    if not path.is_file():
        raise ReviewValidationError(f"review file does not exist: {path}")
    try:
        raw_bytes = path.read_bytes()
        review = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewValidationError(f"cannot read review file {path}: {error}") from error
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ReviewValidationError("review file must be a schema_version 1 object")
    expected_hash = str(review.get("source_ocr_sha256", "")).lower()
    if expected_hash != contract.source_sha256.lower():
        raise ReviewValidationError(
            "review file does not match the source OCR SHA-256: "
            f"expected {contract.source_sha256}, got {expected_hash or '<missing>'}"
        )
    items = review.get("items")
    if not isinstance(items, list):
        raise ReviewValidationError("review items must be an array")
    decision_ids = [item.get("column_id") for item in items if isinstance(item, dict)]
    if len(decision_ids) != len(items) or any(not value for value in decision_ids):
        raise ReviewValidationError("every review item must have a column_id")
    if len(decision_ids) != len(set(decision_ids)):
        raise ReviewValidationError("review file contains duplicate column decisions")
    queue_ids = [item.column_id for item in contract.manual_review_queue]
    missing = sorted(set(queue_ids) - set(decision_ids))
    extra = sorted(set(decision_ids) - set(queue_ids))
    if missing or extra:
        raise ReviewValidationError(
            f"review item set differs from OCR queue; missing={missing}, extra={extra}"
        )

    data = contract.mutable_copy()
    columns = {item["id"]: item for item in data["columns"]}
    decisions = {item["column_id"]: item for item in items}
    image_size = (contract.source.width, contract.source.height)
    rejected_ids: Set[str] = set()
    reviewed_queue = []
    for queue_item in data["manual_review_queue"]:
        column_id = queue_item["column_id"]
        decision_item = decisions[column_id]
        decision = decision_item.get("decision", "pending")
        if decision not in VALID_DECISIONS:
            raise ReviewValidationError(f"invalid decision for {column_id}: {decision}")
        column = columns.get(column_id)
        if column is None:
            if queue_item.get("status") == "rejected" and decision == "rejected":
                reviewed_queue.append(queue_item)
                continue
            raise ReviewValidationError(f"review queue references unknown column {column_id}")
        updated_queue = dict(queue_item)
        if decision == "pending":
            updated_queue["status"] = "pending"
            column["review_required"] = True
            column["review_status"] = "pending"
        elif decision == "rejected":
            rejected_ids.add(column_id)
            updated_queue.update(
                {
                    "status": "rejected",
                    "resolved_text": None,
                    "resolved_polygon": None,
                    "resolved_image_bbox": None,
                    "resolved_layout_bbox_10pct": None,
                    "note": str(decision_item.get("note", "")),
                }
            )
        else:
            if decision == "corrected":
                _apply_correction(column, decision_item, image_size)
            column["review_required"] = False
            column["review_status"] = decision
            column["review_note"] = str(decision_item.get("note", ""))
            updated_queue.update(
                {
                    "status": decision,
                    "resolved_text": column["text"],
                    "resolved_polygon": column["polygon"],
                    "resolved_image_bbox": column["image_bbox"],
                    "resolved_layout_bbox_10pct": column["layout_bbox_10pct"],
                    "note": str(decision_item.get("note", "")),
                }
            )
        reviewed_queue.append(updated_queue)

    data["columns"] = [
        column for column in data["columns"] if column["id"] not in rejected_ids
    ]
    _rebuild_translation_units(data, rejected_ids)
    data["manual_review_queue"] = reviewed_queue
    _refresh_review_summary(data, enabled=True)
    if require_complete and data["manual_review"]["translation_blocked"]:
        pending = [
            item["column_id"] for item in reviewed_queue if item["status"] == "pending"
        ]
        raise ReviewValidationError(f"manual review is still pending for {pending}")

    effective = contract_from_data(
        data,
        path=contract.path,
        source_sha256=contract.source_sha256,
        schema_path=contract.schema_path,
    )
    statuses = [item["status"] for item in reviewed_queue]
    return ReviewApplication(
        contract=effective,
        mode="decisions",
        decision_path=path,
        decision_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        approved_count=statuses.count("approved"),
        corrected_count=statuses.count("corrected"),
        rejected_count=statuses.count("rejected"),
        pending_count=statuses.count("pending"),
        bypassed_count=statuses.count("bypassed"),
    )


def bypass_pending_reviews(application: ReviewApplication) -> ReviewApplication:
    data = application.contract.mutable_copy()
    columns = {item["id"]: item for item in data["columns"]}
    for queue_item in data["manual_review_queue"]:
        if queue_item["status"] == "pending":
            queue_item["status"] = "bypassed"
            column = columns.get(queue_item["column_id"])
            if column is not None:
                column["review_required"] = True
                column["review_status"] = "bypassed"
    _refresh_review_summary(data, enabled=False)
    effective = contract_from_data(
        data,
        path=application.contract.path,
        source_sha256=application.contract.source_sha256,
        schema_path=application.contract.schema_path,
    )
    statuses = [item["status"] for item in data["manual_review_queue"]]
    return ReviewApplication(
        contract=effective,
        mode="bypassed",
        decision_path=application.decision_path,
        decision_sha256=application.decision_sha256,
        approved_count=statuses.count("approved"),
        corrected_count=statuses.count("corrected"),
        rejected_count=statuses.count("rejected"),
        pending_count=statuses.count("pending"),
        bypassed_count=statuses.count("bypassed"),
    )


def ensure_translation_allowed(application: ReviewApplication) -> None:
    contract = application.contract
    if contract.manual_review.translation_blocked or application.pending_count:
        pending = [
            item.column_id
            for item in contract.manual_review_queue
            if item.status == "pending"
        ]
        raise ReviewBlockedError(
            "translation is blocked by pending manual review: " + ", ".join(pending)
        )


def _apply_correction(
    column: Dict[str, Any],
    decision: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> None:
    corrected_text = decision.get("corrected_text")
    corrected_polygon = decision.get("corrected_polygon")
    corrected_bbox = decision.get("corrected_image_bbox")
    use_paddle = bool(decision.get("use_paddle_geometry"))
    if use_paddle:
        if corrected_polygon is not None or corrected_bbox is not None:
            raise ReviewValidationError(
                f"use_paddle_geometry cannot be combined with corrected geometry for {column['id']}"
            )
        corrected_polygon = decision.get("paddle_polygon")
        corrected_bbox = None if corrected_polygon is not None else decision.get("paddle_image_bbox")
        if corrected_polygon is None and corrected_bbox is None:
            raise ReviewValidationError(f"no PaddleOCR geometry for {column['id']}")
    normalized_text = "" if corrected_text is None else str(corrected_text).strip()
    if corrected_polygon is not None and corrected_bbox is not None:
        raise ReviewValidationError(
            f"set corrected_polygon or corrected_image_bbox, not both, for {column['id']}"
        )
    if corrected_text is None and corrected_polygon is None and corrected_bbox is None:
        raise ReviewValidationError(f"corrected decision has no correction for {column['id']}")
    if corrected_text is not None:
        if not normalized_text:
            raise ReviewValidationError(f"corrected_text is empty for {column['id']}")
        column["text"] = normalized_text
    if corrected_polygon is not None:
        polygon = _validated_polygon(corrected_polygon, image_size, column["id"])
        column["polygon"] = polygon
        column["image_bbox"] = _polygon_bbox(polygon)
    if corrected_bbox is not None:
        bbox = _validated_bbox(corrected_bbox, image_size, column["id"])
        column["image_bbox"] = bbox
        x1, y1, x2, y2 = bbox
        column["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    column["layout_bbox_10pct"] = list(
        compute_layout_bbox(column["image_bbox"], 10, image_size)
    )


def _rebuild_translation_units(data: Dict[str, Any], rejected_ids: Set[str]) -> None:
    columns = {item["id"]: item for item in data["columns"]}
    data["reading_order"] = [
        column_id
        for column_id in data["reading_order"]
        if column_id in columns and column_id not in rejected_ids
    ]
    rebuilt = []
    for unit in data["translation_units"]:
        unit_columns = [
            columns[column_id]
            for column_id in unit["column_ids"]
            if column_id in columns and column_id not in rejected_ids
        ]
        if not unit_columns:
            continue
        updated = dict(unit)
        updated["column_ids"] = [column["id"] for column in unit_columns]
        updated["source_text"] = "".join(column["text"] for column in unit_columns)
        updated["translation_query"] = _translation_query(unit_columns)
        updated["layout_regions"] = [
            {
                "column_id": column["id"],
                "source_text": column["text"],
                "polygon": column["polygon"],
                "image_bbox": column["image_bbox"],
                "layout_bbox_10pct": column["layout_bbox_10pct"],
            }
            for column in unit_columns
        ]
        rebuilt.append(updated)
    data["translation_units"] = rebuilt


def _refresh_review_summary(data: Dict[str, Any], *, enabled: bool) -> None:
    statuses = [item["status"] for item in data["manual_review_queue"]]
    pending = statuses.count("pending")
    bypassed = statuses.count("bypassed")
    reviewed = sum(status in {"approved", "corrected", "rejected"} for status in statuses)
    status = (
        "not_required"
        if not statuses
        else "pending"
        if pending
        else "bypassed"
        if bypassed
        else "complete"
    )
    data["manual_review"] = {
        **data["manual_review"],
        "enabled": enabled,
        "status": status,
        "translation_blocked": bool(enabled and pending),
        "candidate_count": len(statuses),
        "pending_count": pending,
        "bypassed_count": bypassed,
        "reviewed_count": reviewed,
        "approved_count": statuses.count("approved"),
        "corrected_count": statuses.count("corrected"),
        "rejected_count": statuses.count("rejected"),
    }


def _translation_query(columns: Sequence[Mapping[str, Any]]) -> str:
    if len(columns) == 1:
        return str(columns[0]["text"])
    return "".join(
        f"<{index}>{column['text']}" for index, column in enumerate(columns, 1)
    )


def _validated_bbox(value: Any, image_size: Tuple[int, int], column_id: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ReviewValidationError(
            f"corrected_image_bbox for {column_id} must contain four coordinates"
        )
    if any(float(number) != int(number) for number in value):
        raise ReviewValidationError(f"corrected_image_bbox for {column_id} must be integer")
    x1, y1, x2, y2 = (int(number) for number in value)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ReviewValidationError(
            f"corrected_image_bbox for {column_id} is outside the image"
        )
    return [x1, y1, x2, y2]


def _validated_polygon(value: Any, image_size: Tuple[int, int], column_id: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) < 4:
        raise ReviewValidationError(
            f"corrected_polygon for {column_id} must contain at least four points"
        )
    width, height = image_size
    points = []
    for raw_point in value:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ReviewValidationError(f"invalid polygon point for {column_id}")
        if any(float(number) != int(number) for number in raw_point):
            raise ReviewValidationError(f"polygon coordinates for {column_id} must be integer")
        x, y = (int(number) for number in raw_point)
        if not (0 <= x <= width and 0 <= y <= height):
            raise ReviewValidationError(f"polygon for {column_id} is outside the image")
        points.append([x, y])
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    if abs(area) < 1e-6:
        raise ReviewValidationError(f"polygon for {column_id} has zero area")
    return points


def _polygon_bbox(polygon: Sequence[Sequence[int]]) -> list[int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))]
