from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Sequence

from background_ocr import attach_manual_review_queue, expand_bbox_10pct, polygon_bbox


VALID_DECISIONS = {"pending", "approved", "corrected", "rejected"}


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


def prepare_review(source: Path, output_ocr: Path, review_path: Path) -> None:
    result = attach_manual_review_queue(_read_json(source))
    _write_json(output_ocr, result)
    _write_json(review_path, build_review_template(result, output_ocr))


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


def apply_review(source: Path, review_path: Path, output: Path) -> None:
    review = _read_json(review_path)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and apply manual OCR review decisions.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Attach a review queue and export a decision template.")
    prepare.add_argument("ocr_json", type=Path)
    prepare.add_argument("--output-ocr", type=Path)
    prepare.add_argument("--review", type=Path)
    apply = commands.add_parser("apply", help="Apply completed human decisions to an OCR JSON.")
    apply.add_argument("ocr_json", type=Path)
    apply.add_argument("review_json", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        output_ocr = args.output_ocr or args.ocr_json
        review_path = args.review or output_ocr.with_suffix(".review.json")
        prepare_review(args.ocr_json, output_ocr, review_path)
        print(output_ocr)
        print(review_path)
        return 0
    apply_review(args.ocr_json, args.review_json, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
