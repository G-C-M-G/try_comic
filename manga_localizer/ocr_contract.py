from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator
from PIL import Image

from .errors import (
    ContractValidationError,
    ImageDimensionError,
    ImageResolutionError,
    LayoutBoundaryError,
)
from .layout_bounds import assert_layout_bbox_10pct


Point = Tuple[float, float]
Polygon = Tuple[Point, ...]
BBox = Tuple[int, int, int, int]
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SourceInfo:
    path: str
    width: int
    height: int


@dataclass(frozen=True)
class OCRColumn:
    id: str
    text: str
    polygon: Polygon
    image_bbox: BBox
    layout_bbox_10pct: BBox
    selection_order: int
    panel_index: int
    text_source: str
    detection_sources: Tuple[str, ...]
    review_required: bool
    review_status: str
    review_reasons: Tuple[str, ...]
    direction: str = "vertical"
    role: str = "text"


@dataclass(frozen=True)
class LayoutRegion:
    column_id: str
    source_text: str
    polygon: Polygon
    image_bbox: BBox
    layout_bbox_10pct: BBox


@dataclass(frozen=True)
class TranslationUnit:
    id: str
    column_ids: Tuple[str, ...]
    source_text: str
    translation_query: str
    panel_index: int
    layout_regions: Tuple[LayoutRegion, ...]


@dataclass(frozen=True)
class ManualReviewSummary:
    enabled: bool
    status: str
    translation_blocked: bool
    candidate_count: int
    pending_count: int
    bypassed_count: int
    reviewed_count: int


@dataclass(frozen=True)
class ManualReviewItem:
    column_id: str
    status: str
    reason_codes: Tuple[str, ...]
    current_text: str
    polygon: Polygon
    image_bbox: BBox
    layout_bbox_10pct: BBox
    mangaocr_readings: Tuple[str, ...]
    mangaocr_confidence: Optional[float]
    paddleocr_text: str
    paddleocr_confidence: Optional[float]
    paddle_box_overlap: Optional[float]
    resolved_text: Optional[str]
    resolved_polygon: Optional[Polygon]
    resolved_image_bbox: Optional[BBox]
    resolved_layout_bbox_10pct: Optional[BBox]


@dataclass(frozen=True)
class OCRContract:
    path: Path
    schema_path: Path
    source_sha256: str
    effective_sha256: str
    schema_sha256: str
    schema_version: int
    backend: str
    source: SourceInfo
    capture: Mapping[str, Any]
    columns: Tuple[OCRColumn, ...]
    reading_order: Tuple[str, ...]
    translation_units: Tuple[TranslationUnit, ...]
    ignored_regions: Tuple[Mapping[str, Any], ...]
    fusion_summary: Mapping[str, Any]
    manual_review: ManualReviewSummary
    manual_review_queue: Tuple[ManualReviewItem, ...]
    benchmark: Optional[Mapping[str, Any]]
    raw: Mapping[str, Any]

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.source.width, self.source.height

    @property
    def columns_by_id(self) -> Mapping[str, OCRColumn]:
        return MappingProxyType({column.id: column for column in self.columns})

    @property
    def units_by_column_id(self) -> Mapping[str, Tuple[TranslationUnit, int]]:
        result: Dict[str, Tuple[TranslationUnit, int]] = {}
        for unit in self.translation_units:
            for index, column_id in enumerate(unit.column_ids):
                result[column_id] = (unit, index)
        return MappingProxyType(result)

    def mutable_copy(self) -> Dict[str, Any]:
        return _thaw(self.raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "schema" / "ocr_schema_v4.json"


def load_ocr_contract(
    ocr_path: Path,
    schema_path: Optional[Path] = None,
) -> OCRContract:
    path = Path(ocr_path).resolve()
    if not path.is_file():
        raise ContractValidationError(f"OCR JSON does not exist: {path}")
    try:
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"cannot read OCR JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise ContractValidationError("OCR JSON root must be an object")
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    return contract_from_data(
        data,
        path=path,
        source_sha256=source_hash,
        schema_path=schema_path,
    )


def contract_from_data(
    data: Mapping[str, Any],
    *,
    path: Path,
    source_sha256: str,
    schema_path: Optional[Path] = None,
) -> OCRContract:
    schema_file = Path(schema_path or default_schema_path()).resolve()
    if not schema_file.is_file():
        raise ContractValidationError(f"schema file does not exist: {schema_file}")
    try:
        schema_bytes = schema_file.read_bytes()
        schema = json.loads(schema_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"cannot read schema {schema_file}: {error}") from error

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path))
    if errors:
        formatted = []
        for error in errors[:20]:
            location = "$"
            for part in error.absolute_path:
                location += f"[{part}]" if isinstance(part, int) else f".{part}"
            formatted.append(f"{location}: {error.message}")
        if len(errors) > 20:
            formatted.append(f"... and {len(errors) - 20} more schema errors")
        raise ContractValidationError("schema 4 validation failed:\n" + "\n".join(formatted))

    mutable_data = _plain_copy(data)
    _validate_cross_fields(mutable_data)
    return _build_contract(
        mutable_data,
        path=Path(path).resolve(),
        source_sha256=source_sha256,
        schema_path=schema_file,
        schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
    )


def resolve_image_path(
    contract: OCRContract,
    *,
    explicit_image: Optional[Path] = None,
    sample_dirs: Sequence[Path] = (),
) -> Path:
    if explicit_image is not None:
        candidate = Path(explicit_image).expanduser().resolve()
        if not candidate.is_file():
            raise ImageResolutionError(f"explicit --image does not exist: {candidate}")
        validate_image_dimensions(candidate, contract)
        return candidate

    source_name = _portable_filename(contract.source.path)
    if not source_name:
        source_name = _ocr_stem(contract.path.name) + ".jpg"
    candidates = [contract.path.parent / source_name]
    candidates.extend(Path(directory) / source_name for directory in sample_dirs)
    candidates.extend(
        [
            contract.path.parent / "images" / source_name,
            contract.path.parent.parent / "images" / source_name,
        ]
    )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(resolved)
    for candidate in unique_candidates:
        if candidate.is_file():
            validate_image_dimensions(candidate, contract)
            return candidate
    tried = "\n".join(f"  - {path}" for path in unique_candidates)
    raise ImageResolutionError(
        "could not resolve the image without using the stale source.path; tried:\n" + tried
    )


def validate_image_dimensions(image_path: Path, contract: OCRContract) -> None:
    try:
        with Image.open(image_path) as image:
            actual = image.size
    except (OSError, ValueError) as error:
        raise ImageDimensionError(f"cannot read image {image_path}: {error}") from error
    if actual != contract.image_size:
        raise ImageDimensionError(
            f"image size {actual[0]}x{actual[1]} does not match OCR source "
            f"{contract.source.width}x{contract.source.height}: {image_path}"
        )


def _validate_cross_fields(data: Dict[str, Any]) -> None:
    source = data["source"]
    width, height = int(source["width"]), int(source["height"])
    image_size = (width, height)
    allowance = data.get("capture", {}).get("layout_overflow_allowance")
    if allowance is not None and not math.isclose(float(allowance), 0.1, abs_tol=1e-9):
        raise ContractValidationError(
            f"capture.layout_overflow_allowance must be 0.1, got {allowance}"
        )

    columns = data["columns"]
    column_ids = [column["id"] for column in columns]
    _require_unique(column_ids, "columns[].id")
    column_map = {column["id"]: column for column in columns}
    selection_orders = [int(column["selection_order"]) for column in columns]
    _require_unique(selection_orders, "columns[].selection_order")

    for column in columns:
        location = f"columns[{column['id']}]"
        _validate_polygon(column["polygon"], image_size, f"{location}.polygon")
        _validate_bbox(column["image_bbox"], image_size, f"{location}.image_bbox")
        _validate_bbox(
            column["layout_bbox_10pct"], image_size, f"{location}.layout_bbox_10pct"
        )
        try:
            assert_layout_bbox_10pct(
                column["image_bbox"], column["layout_bbox_10pct"], image_size
            )
        except LayoutBoundaryError as error:
            raise ContractValidationError(f"{location}: {error}") from error

    reading_order = data["reading_order"]
    if set(reading_order) != set(column_ids) or len(reading_order) != len(column_ids):
        raise ContractValidationError(
            "reading_order must contain every retained column exactly once"
        )
    expected_order = [
        column["id"] for column in sorted(columns, key=lambda item: item["selection_order"])
    ]
    if reading_order != expected_order:
        raise ContractValidationError(
            "reading_order must match ascending columns[].selection_order"
        )

    unit_ids = [unit["id"] for unit in data["translation_units"]]
    _require_unique(unit_ids, "translation_units[].id")
    assigned = []
    for unit in data["translation_units"]:
        location = f"translation_units[{unit['id']}]"
        ids = unit["column_ids"]
        region_ids = [region["column_id"] for region in unit["layout_regions"]]
        if ids != region_ids:
            raise ContractValidationError(
                f"{location}: column_ids must exactly match layout_regions column order"
            )
        missing = [column_id for column_id in ids if column_id not in column_map]
        if missing:
            raise ContractValidationError(f"{location}: unknown column IDs {missing}")
        unit_columns = [column_map[column_id] for column_id in ids]
        expected_source = "".join(column["text"] for column in unit_columns)
        if unit["source_text"] != expected_source:
            raise ContractValidationError(
                f"{location}.source_text does not equal the complete ordered sentence"
            )
        expected_query = _translation_query(unit_columns)
        if unit["translation_query"] != expected_query:
            raise ContractValidationError(
                f"{location}.translation_query does not preserve the column markers"
            )
        if any(column["panel_index"] != unit["panel_index"] for column in unit_columns):
            raise ContractValidationError(f"{location}: columns cross panel boundaries")
        for region, column in zip(unit["layout_regions"], unit_columns):
            if region["source_text"] != column["text"]:
                raise ContractValidationError(
                    f"{location}/{column['id']}: layout source_text differs from column text"
                )
            for field in ("polygon", "image_bbox", "layout_bbox_10pct"):
                if region[field] != column[field]:
                    raise ContractValidationError(
                        f"{location}/{column['id']}: layout {field} differs from column"
                    )
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)):
        raise ContractValidationError("a retained column belongs to multiple translation units")
    if set(assigned) != set(column_ids):
        missing = sorted(set(column_ids) - set(assigned))
        raise ContractValidationError(
            f"every retained column must belong to a translation unit; missing {missing}"
        )

    for index, ignored in enumerate(data["ignored_regions"]):
        if "polygon" in ignored:
            _validate_polygon(ignored["polygon"], image_size, f"ignored_regions[{index}].polygon")
        if "image_bbox" in ignored:
            _validate_bbox(ignored["image_bbox"], image_size, f"ignored_regions[{index}].image_bbox")
        if "layout_bbox_10pct" in ignored:
            _validate_bbox(
                ignored["layout_bbox_10pct"],
                image_size,
                f"ignored_regions[{index}].layout_bbox_10pct",
            )

    _validate_review_fields(data, column_map, image_size)


def _validate_review_fields(
    data: Dict[str, Any],
    column_map: Mapping[str, Mapping[str, Any]],
    image_size: Tuple[int, int],
) -> None:
    summary = data["manual_review"]
    queue = data["manual_review_queue"]
    queue_ids = [item["column_id"] for item in queue]
    _require_unique(queue_ids, "manual_review_queue[].column_id")
    if summary["candidate_count"] != len(queue):
        raise ContractValidationError(
            "manual_review.candidate_count must equal manual_review_queue length"
        )
    pending_count = sum(item["status"] == "pending" for item in queue)
    bypassed_count = sum(item["status"] == "bypassed" for item in queue)
    reviewed_count = sum(
        item["status"] in {"approved", "corrected", "rejected"} for item in queue
    )
    if summary["pending_count"] != pending_count:
        raise ContractValidationError("manual_review.pending_count is inconsistent with queue")
    if summary["bypassed_count"] != bypassed_count:
        raise ContractValidationError("manual_review.bypassed_count is inconsistent with queue")
    if "reviewed_count" in summary and summary["reviewed_count"] != reviewed_count:
        raise ContractValidationError("manual_review.reviewed_count is inconsistent with queue")
    expected_blocked = bool(summary["enabled"] and pending_count)
    if bool(summary["translation_blocked"]) != expected_blocked:
        raise ContractValidationError(
            "manual_review.translation_blocked must reflect enabled pending items"
        )
    expected_status = (
        "not_required"
        if not queue
        else "pending"
        if pending_count
        else "bypassed"
        if bypassed_count
        else "complete"
    )
    if summary["status"] != expected_status:
        raise ContractValidationError(
            f"manual_review.status must be {expected_status}, got {summary['status']}"
        )

    queue_map = {item["column_id"]: item for item in queue}
    for column_id, item in queue_map.items():
        location = f"manual_review_queue[{column_id}]"
        _validate_polygon(item["polygon"], image_size, f"{location}.polygon")
        _validate_bbox(item["image_bbox"], image_size, f"{location}.image_bbox")
        _validate_bbox(
            item["layout_bbox_10pct"], image_size, f"{location}.layout_bbox_10pct"
        )
        if item["status"] == "rejected":
            if column_id in column_map:
                raise ContractValidationError(f"{location}: rejected column was not removed")
            continue
        if column_id not in column_map:
            raise ContractValidationError(f"{location}: queue references an unknown column")
        column = column_map[column_id]
        resolved_text = item.get("resolved_text", item["current_text"])
        resolved_polygon = item.get("resolved_polygon", item["polygon"])
        resolved_bbox = item.get("resolved_image_bbox", item["image_bbox"])
        resolved_layout = item.get("resolved_layout_bbox_10pct", item["layout_bbox_10pct"])
        if resolved_text != column["text"]:
            raise ContractValidationError(f"{location}: resolved text differs from column")
        if resolved_polygon != column["polygon"] or resolved_bbox != column["image_bbox"]:
            raise ContractValidationError(f"{location}: resolved geometry differs from column")
        if resolved_layout != column["layout_bbox_10pct"]:
            raise ContractValidationError(f"{location}: resolved layout differs from column")
        if column["review_status"] != item["status"]:
            raise ContractValidationError(f"{location}: review status differs from column")

    for column_id, column in column_map.items():
        if column["review_required"] and column_id not in queue_map:
            raise ContractValidationError(
                f"columns[{column_id}] requires review but has no queue item"
            )


def _build_contract(
    data: Dict[str, Any],
    *,
    path: Path,
    source_sha256: str,
    schema_path: Path,
    schema_sha256: str,
) -> OCRContract:
    source = data["source"]
    columns = tuple(
        OCRColumn(
            id=item["id"],
            text=item["text"],
            polygon=_polygon(item["polygon"]),
            image_bbox=_bbox(item["image_bbox"]),
            layout_bbox_10pct=_bbox(item["layout_bbox_10pct"]),
            selection_order=int(item["selection_order"]),
            panel_index=int(item["panel_index"]),
            text_source=item["text_source"],
            detection_sources=tuple(item["detection_sources"]),
            review_required=bool(item["review_required"]),
            review_status=item["review_status"],
            review_reasons=tuple(item["review_reasons"]),
            direction=str(item.get("direction") or "vertical"),
            role=str(item.get("role") or "text"),
        )
        for item in data["columns"]
    )
    units = tuple(
        TranslationUnit(
            id=item["id"],
            column_ids=tuple(item["column_ids"]),
            source_text=item["source_text"],
            translation_query=item["translation_query"],
            panel_index=int(item["panel_index"]),
            layout_regions=tuple(
                LayoutRegion(
                    column_id=region["column_id"],
                    source_text=region["source_text"],
                    polygon=_polygon(region["polygon"]),
                    image_bbox=_bbox(region["image_bbox"]),
                    layout_bbox_10pct=_bbox(region["layout_bbox_10pct"]),
                )
                for region in item["layout_regions"]
            ),
        )
        for item in data["translation_units"]
    )
    review = data["manual_review"]
    queue = tuple(
        ManualReviewItem(
            column_id=item["column_id"],
            status=item["status"],
            reason_codes=tuple(item["reason_codes"]),
            current_text=item["current_text"],
            polygon=_polygon(item["polygon"]),
            image_bbox=_bbox(item["image_bbox"]),
            layout_bbox_10pct=_bbox(item["layout_bbox_10pct"]),
            mangaocr_readings=tuple(item.get("mangaocr_readings", ())),
            mangaocr_confidence=_optional_float(item.get("mangaocr_confidence")),
            paddleocr_text=str(item.get("paddleocr_text") or ""),
            paddleocr_confidence=_optional_float(item.get("paddleocr_confidence")),
            paddle_box_overlap=_optional_float(item.get("paddle_box_overlap")),
            resolved_text=item.get("resolved_text"),
            resolved_polygon=(
                _polygon(item["resolved_polygon"])
                if item.get("resolved_polygon") is not None
                else None
            ),
            resolved_image_bbox=(
                _bbox(item["resolved_image_bbox"])
                if item.get("resolved_image_bbox") is not None
                else None
            ),
            resolved_layout_bbox_10pct=(
                _bbox(item["resolved_layout_bbox_10pct"])
                if item.get("resolved_layout_bbox_10pct") is not None
                else None
            ),
        )
        for item in data["manual_review_queue"]
    )
    frozen = _freeze(data)
    return OCRContract(
        path=path,
        schema_path=schema_path,
        source_sha256=source_sha256,
        effective_sha256=canonical_sha256(data),
        schema_sha256=schema_sha256,
        schema_version=int(data["schema_version"]),
        backend=data["backend"],
        source=SourceInfo(source["path"], int(source["width"]), int(source["height"])),
        capture=_freeze(data["capture"]),
        columns=columns,
        reading_order=tuple(data["reading_order"]),
        translation_units=units,
        ignored_regions=tuple(_freeze(item) for item in data["ignored_regions"]),
        fusion_summary=_freeze(data["fusion_summary"]),
        manual_review=ManualReviewSummary(
            enabled=bool(review["enabled"]),
            status=review["status"],
            translation_blocked=bool(review["translation_blocked"]),
            candidate_count=int(review["candidate_count"]),
            pending_count=int(review["pending_count"]),
            bypassed_count=int(review["bypassed_count"]),
            reviewed_count=int(review.get("reviewed_count", 0)),
        ),
        manual_review_queue=queue,
        benchmark=_freeze(data["benchmark"]) if "benchmark" in data else None,
        raw=frozen,
    )


def _validate_polygon(value: Sequence[Sequence[float]], image_size: Tuple[int, int], location: str) -> None:
    width, height = image_size
    area_sum = 0.0
    points = []
    for point in value:
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ContractValidationError(f"{location}: coordinates must be finite")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ContractValidationError(
                f"{location}: point {[x, y]} is outside {width}x{height}"
            )
        points.append((x, y))
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area_sum += x1 * y2 - x2 * y1
    if abs(area_sum) < 1e-6:
        raise ContractValidationError(f"{location}: polygon has zero area")


def _validate_bbox(value: Sequence[float], image_size: Tuple[int, int], location: str) -> None:
    if any(float(number) != int(number) for number in value):
        raise ContractValidationError(f"{location}: bbox must use integer pixel coordinates")
    x1, y1, x2, y2 = (int(number) for number in value)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ContractValidationError(
            f"{location}: bbox {list(value)} is outside {width}x{height}"
        )


def _require_unique(values: Iterable[Any], location: str) -> None:
    values_list = list(values)
    if len(values_list) != len(set(values_list)):
        raise ContractValidationError(f"{location} contains duplicate values")


def _translation_query(columns: Sequence[Mapping[str, Any]]) -> str:
    if len(columns) == 1:
        return str(columns[0]["text"])
    return "".join(
        f"<{index}>{column['text']}" for index, column in enumerate(columns, 1)
    )


def _portable_filename(value: str) -> str:
    normalized = str(value).replace("\\", "/").rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    return "" if name in {"", ".", ".."} else name


def _ocr_stem(name: str) -> str:
    return name[:-9] if name.lower().endswith(".ocr.json") else Path(name).stem


def _polygon(value: Sequence[Sequence[float]]) -> Polygon:
    return tuple((float(point[0]), float(point[1])) for point in value)


def _bbox(value: Sequence[float]) -> BBox:
    return tuple(int(number) for number in value)  # type: ignore[return-value]


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _plain_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
