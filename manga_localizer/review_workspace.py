from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .errors import ReviewValidationError
from .ocr_contract import OCRContract
from .review import VALID_DECISIONS, ReviewApplication, apply_review_decisions


REVIEW_INSTRUCTIONS = {
    "approved": "Keep the current text and geometry.",
    "corrected": "Set corrected_text and/or corrected geometry.",
    "rejected": "Remove this false-positive OCR region.",
}


@dataclass(frozen=True)
class ReviewCounts:
    approved: int
    corrected: int
    rejected: int
    pending: int

    @property
    def complete(self) -> bool:
        return self.pending == 0


class ReviewWorkspace:
    """Editable review decisions bound to one immutable OCR contract."""

    def __init__(
        self,
        contract: OCRContract,
        document: Mapping[str, Any],
        *,
        path: Optional[Path] = None,
    ) -> None:
        self.contract = contract
        self._document = deepcopy(dict(document))
        self.path = Path(path).resolve() if path is not None else None
        self.dirty = False
        self._validate_identity()

    @classmethod
    def create(cls, contract: OCRContract, *, path: Optional[Path] = None) -> "ReviewWorkspace":
        raw_queue = contract.mutable_copy()["manual_review_queue"]
        items = []
        for queue_item in raw_queue:
            item = deepcopy(queue_item)
            source_status = str(item.get("status", "pending"))
            item["decision"] = (
                source_status if source_status in {"approved", "rejected"} else "pending"
            )
            item["corrected_text"] = None
            item["corrected_polygon"] = None
            item["corrected_image_bbox"] = None
            item["note"] = ""
            items.append(item)
        return cls(
            contract,
            {
                "schema_version": 1,
                "source_ocr": str(contract.path),
                "source_ocr_sha256": contract.source_sha256,
                "instructions": REVIEW_INSTRUCTIONS,
                "items": items,
            },
            path=path,
        )

    @classmethod
    def load(cls, contract: OCRContract, path: Path) -> "ReviewWorkspace":
        resolved = Path(path).resolve()
        try:
            document = json.loads(resolved.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReviewValidationError(
                f"cannot read review workspace {resolved}: {error}"
            ) from error
        workspace = cls(contract, document, path=resolved)
        apply_review_decisions(contract, resolved)
        return workspace

    @classmethod
    def load_or_create(
        cls,
        contract: OCRContract,
        path: Optional[Path],
    ) -> "ReviewWorkspace":
        if path is not None and Path(path).is_file():
            return cls.load(contract, path)
        return cls.create(contract, path=path)

    @property
    def items(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(item) for item in self._document["items"])

    @property
    def counts(self) -> ReviewCounts:
        decisions = [str(item.get("decision", "pending")) for item in self._document["items"]]
        return ReviewCounts(
            approved=decisions.count("approved"),
            corrected=decisions.count("corrected"),
            rejected=decisions.count("rejected"),
            pending=decisions.count("pending"),
        )

    def item(self, column_id: str) -> Mapping[str, Any]:
        for item in self._document["items"]:
            if item["column_id"] == column_id:
                return deepcopy(item)
        raise ReviewValidationError(f"review workspace has no item {column_id}")

    def update(
        self,
        column_id: str,
        decision: str,
        *,
        corrected_text: Optional[str] = None,
        corrected_image_bbox: Optional[Iterable[int]] = None,
        note: str = "",
    ) -> None:
        if decision not in VALID_DECISIONS:
            raise ReviewValidationError(f"invalid review decision: {decision}")
        target = None
        for item in self._document["items"]:
            if item["column_id"] == column_id:
                target = item
                break
        if target is None:
            raise ReviewValidationError(f"review workspace has no item {column_id}")

        text = None if corrected_text is None else str(corrected_text).strip()
        bbox = None
        if corrected_image_bbox is not None:
            try:
                bbox = [int(value) for value in corrected_image_bbox]
            except (TypeError, ValueError) as error:
                raise ReviewValidationError("corrected box coordinates must be integers") from error
            if len(bbox) != 4:
                raise ReviewValidationError("corrected box must contain four coordinates")

        target["decision"] = decision
        target["note"] = str(note).strip()
        target["corrected_polygon"] = None
        if decision == "corrected":
            original_text = str(target.get("current_text", "")).strip()
            original_bbox = [int(value) for value in target.get("image_bbox", [])]
            changed_text = bool(text) and text != original_text
            changed_bbox = bbox is not None and bbox != original_bbox
            if not changed_text and not changed_bbox:
                raise ReviewValidationError(
                    "corrected decision must change the text or the complete OCR box"
                )
            target["corrected_text"] = text if changed_text else None
            target["corrected_image_bbox"] = bbox if changed_bbox else None
        else:
            target["corrected_text"] = None
            target["corrected_image_bbox"] = None
        self.dirty = True

    def save(self, path: Optional[Path] = None) -> ReviewApplication:
        destination = Path(path or self.path or "").resolve()
        if not str(path or self.path or "").strip():
            raise ReviewValidationError("review output path is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._document,
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            apply_review_decisions(self.contract, temporary_path)
            os.replace(temporary_path, destination)
            application = apply_review_decisions(self.contract, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        self.path = destination
        self.dirty = False
        return application

    def _validate_identity(self) -> None:
        if self._document.get("schema_version") != 1:
            raise ReviewValidationError("review workspace must use schema_version 1")
        source_hash = str(self._document.get("source_ocr_sha256", "")).lower()
        if source_hash != self.contract.source_sha256.lower():
            raise ReviewValidationError("review workspace belongs to a different OCR input")
        items = self._document.get("items")
        if not isinstance(items, list):
            raise ReviewValidationError("review workspace items must be an array")
        item_ids = [item.get("column_id") for item in items if isinstance(item, dict)]
        expected_ids = [item.column_id for item in self.contract.manual_review_queue]
        if len(item_ids) != len(items) or set(item_ids) != set(expected_ids):
            raise ReviewValidationError("review workspace items differ from the OCR review queue")
        if len(item_ids) != len(set(item_ids)):
            raise ReviewValidationError("review workspace contains duplicate column decisions")
