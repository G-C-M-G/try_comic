"""Island dataset parsing and validation (islands.json + filled_* masks).

The upstream project produces one directory per page containing
``islands.json``, ``filled_black.png`` / ``filled_blue.png`` (original
resolution, island pixels preserved on a fill-colored background),
``original.jpg`` and ``photos_ocr.png``.  These are stable, read-only inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from PIL import Image

from .errors import ContractValidationError, ImageDimensionError
from .ocr_contract import BBox, Polygon, sha256_file


Point = Tuple[float, float]
RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class IslandInfo:
    id: str
    image_bbox: BBox
    image_center: Tuple[int, int]
    image_area: float
    polygon: Optional[Polygon]
    display_bbox: Optional[BBox]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class IslandDataset:
    path: Path
    source_sha256: str
    effective_sha256: str
    page: str
    image_size: Tuple[int, int]
    fill_color: Optional[RGBColor]
    contours: bool
    islands: Tuple[IslandInfo, ...]
    raw: Mapping[str, Any]

    @property
    def islands_by_id(self) -> Mapping[str, IslandInfo]:
        return MappingProxyType({island.id: island for island in self.islands})


def load_island_dataset(islands_json: Path) -> IslandDataset:
    """Parse and validate one page's ``islands.json``."""
    path = Path(islands_json).resolve()
    if not path.is_file():
        raise ContractValidationError(f"islands.json does not exist: {path}")
    try:
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"cannot read islands.json {path}: {error}") from error
    if not isinstance(data, dict):
        raise ContractValidationError("islands.json root must be an object")
    return dataset_from_data(
        data,
        path=path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def dataset_from_data(
    data: Mapping[str, Any],
    *,
    path: Path,
    source_sha256: str,
) -> IslandDataset:
    try:
        size_value = data["image_size"]
        width, height = (int(size_value[0]), int(size_value[1]))
        if width <= 0 or height <= 0:
            raise ContractValidationError("image_size must be positive")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ContractValidationError(f"invalid image_size: {error}") from error
    image_size = (width, height)

    islands = data.get("islands")
    if not isinstance(islands, list) or not islands:
        raise ContractValidationError("islands must be a non-empty list")

    fill = data.get("fill_black") or data.get("fill_blue")
    fill_color = None
    if isinstance(fill, Sequence) and len(fill) == 3:
        try:
            fill_color = (int(fill[0]), int(fill[1]), int(fill[2]))
        except (TypeError, ValueError):
            fill_color = None

    parsed = []
    seen = set()
    for index, item in enumerate(islands):
        location = f"islands[{index}]"
        if not isinstance(item, dict):
            raise ContractValidationError(f"{location} must be an object")
        island_id = str(item.get("id") or "")
        if not island_id:
            raise ContractValidationError(f"{location}: missing id")
        if island_id in seen:
            raise ContractValidationError(f"{location}: duplicate island id {island_id}")
        seen.add(island_id)
        bbox = _validate_bbox(item.get("image_bbox"), image_size, f"{location}.image_bbox")
        center = _validate_center(item.get("image_center"), image_size, f"{location}.image_center")
        area = _validate_positive_float(item.get("image_area"), f"{location}.image_area")
        polygon = None
        if item.get("image_polygon") is not None:
            polygon = _validate_polygon(
                item["image_polygon"], image_size, f"{location}.image_polygon"
            )
        display_bbox = None
        if item.get("display_bbox") is not None:
            display_bbox = _validate_bbox(
                item["display_bbox"], (width * 10, height * 10), f"{location}.display_bbox"
            )
        parsed.append(
            IslandInfo(
                id=island_id,
                image_bbox=bbox,
                image_center=center,
                image_area=area,
                polygon=polygon,
                display_bbox=display_bbox,
                raw=MappingProxyType(dict(item)),
            )
        )

    dataset = IslandDataset(
        path=Path(path).resolve(),
        source_sha256=source_sha256,
        effective_sha256=_canonical(data),
        page=str(data.get("page") or Path(path).stem),
        image_size=image_size,
        fill_color=fill_color,
        contours=bool(data.get("contours")),
        islands=tuple(parsed),
        raw=MappingProxyType(dict(data)),
    )
    return dataset


def validate_dataset_images(
    dataset: IslandDataset,
    directory: Path,
    *,
    require: Sequence[str] = ("filled_black.png", "filled_blue.png", "original.jpg"),
) -> Mapping[str, Path]:
    """Check the page directory images exist and match the dataset size."""
    directory = Path(directory).resolve()
    found = {}
    for name in require:
        candidate = directory / name
        if not candidate.is_file():
            raise ContractValidationError(f"dataset image missing: {candidate}")
        validate_image_size(candidate, dataset.image_size)
        found[name] = candidate
    return MappingProxyType(found)


def validate_image_size(image_path: Path, image_size: Tuple[int, int]) -> None:
    try:
        with Image.open(image_path) as image:
            actual = image.size
    except (OSError, ValueError) as error:
        raise ImageDimensionError(f"cannot read image {image_path}: {error}") from error
    if actual != image_size:
        raise ImageDimensionError(
            f"image size {actual[0]}x{actual[1]} does not match islands.json "
            f"{image_size[0]}x{image_size[1]}: {image_path}"
        )


def filter_tiny_islands(
    dataset: IslandDataset,
    min_island_area: float,
) -> Tuple[IslandDataset, Tuple[IslandInfo, ...]]:
    """Drop islands whose ``image_area`` is below *min_island_area*.

    Dropped islands are excluded from erasure, translation and typesetting;
    the returned dataset carries an updated effective hash so cache signatures
    bind to the retained island set.
    """
    if min_island_area <= 0:
        return dataset, ()
    kept = [island for island in dataset.islands if island.image_area >= min_island_area]
    dropped = tuple(island for island in dataset.islands if island.image_area < min_island_area)
    if not dropped:
        return dataset, ()
    payload = {
        "schema_version": 1,
        "page": dataset.page,
        "image_size": list(dataset.image_size),
        "contours": dataset.contours,
        "islands": [dict(island.raw) for island in kept],
    }
    if dataset.fill_color is not None:
        key = "fill_black" if dataset.raw.get("fill_black") is not None else "fill_blue"
        payload[key] = list(dataset.fill_color)
    filtered = IslandDataset(
        path=dataset.path,
        source_sha256=dataset.source_sha256,
        effective_sha256=_canonical(payload),
        page=dataset.page,
        image_size=dataset.image_size,
        fill_color=dataset.fill_color,
        contours=dataset.contours,
        islands=tuple(kept),
        raw=MappingProxyType(payload),
    )
    return filtered, dropped


def expand_island_bboxes(
    dataset: IslandDataset,
    *,
    expand_ids: Sequence[str] = (),
    pad: int = 30,
    fragment_boxes: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
) -> IslandDataset:
    """Return a corrected dataset whose selected islands get an expanded bbox.

    The expansion pads each side by *pad* pixels (clamped to the image) and
    then grows further to cover any Paddle fragment boxes that stick out of
    the island (clipped text recovery).  Islands not listed in *expand_ids*
    keep their geometry; the effective hash changes so cache signatures bind
    to the corrected geometry.
    """
    if pad <= 0:
        return dataset
    targets = set(expand_ids)
    fragments = {key: list(value or ()) for key, value in (fragment_boxes or {}).items()}
    width, height = dataset.image_size
    corrected = []
    changed = False
    for island in dataset.islands:
        raw = dict(island.raw)
        if island.id in targets:
            x1, y1, x2, y2 = island.image_bbox
            new_x1, new_y1 = max(0, x1 - pad), max(0, y1 - pad)
            new_x2, new_y2 = min(width, x2 + pad), min(height, y2 + pad)
            for box in fragments.get(island.id, ()):
                if len(box) != 4:
                    continue
                fx1, fy1, fx2, fy2 = (float(value) for value in box)
                new_x1 = min(new_x1, max(0, int(math.floor(fx1)) - 2))
                new_y1 = min(new_y1, max(0, int(math.floor(fy1)) - 2))
                new_x2 = max(new_x2, min(width, int(math.ceil(fx2)) + 2))
                new_y2 = max(new_y2, min(height, int(math.ceil(fy2)) + 2))
            new_bbox = (new_x1, new_y1, new_x2, new_y2)
            if new_bbox != island.image_bbox:
                raw["image_bbox"] = list(new_bbox)
                raw["image_center"] = [
                    (new_bbox[0] + new_bbox[2]) // 2,
                    (new_bbox[1] + new_bbox[3]) // 2,
                ]
                changed = True
        corrected.append(raw)
    if not changed:
        return dataset
    payload = {
        "schema_version": 1,
        "page": dataset.page,
        "image_size": list(dataset.image_size),
        "contours": dataset.contours,
        "islands": corrected,
        "expanded_islands": sorted(targets),
        "expansion_pad": pad,
    }
    if dataset.fill_color is not None:
        key = "fill_black" if dataset.raw.get("fill_black") is not None else "fill_blue"
        payload[key] = list(dataset.fill_color)
    return IslandDataset(
        path=dataset.path,
        source_sha256=dataset.source_sha256,
        effective_sha256=_canonical(payload),
        page=dataset.page,
        image_size=dataset.image_size,
        fill_color=dataset.fill_color,
        contours=dataset.contours,
        islands=tuple(
            IslandInfo(
                id=item["id"],
                image_bbox=_validate_bbox(item["image_bbox"], dataset.image_size, f"{item['id']}.image_bbox"),
                image_center=tuple(int(value) for value in item["image_center"]),
                image_area=float(item["image_area"]),
                polygon=(
                    _validate_polygon(item["image_polygon"], dataset.image_size, f"{item['id']}.image_polygon")
                    if item.get("image_polygon") is not None
                    else None
                ),
                display_bbox=(
                    _validate_bbox(item["display_bbox"], (width * 10, height * 10), f"{item['id']}.display_bbox")
                    if item.get("display_bbox") is not None
                    else None
                ),
                raw=MappingProxyType(dict(item)),
            )
            for item in corrected
        ),
        raw=MappingProxyType(payload),
    )


def _polygon_bbox(polygon: Polygon) -> BBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def _validate_bbox(value: Any, image_size: Tuple[int, int], location: str) -> BBox:
    if not (isinstance(value, Sequence) and len(value) == 4):
        raise ContractValidationError(f"{location}: must be [x1, y1, x2, y2]")
    if any(float(number) != int(number) for number in value):
        raise ContractValidationError(f"{location}: bbox must use integer pixel coordinates")
    x1, y1, x2, y2 = (int(number) for number in value)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ContractValidationError(f"{location}: bbox {list(value)} is outside {width}x{height}")
    return (x1, y1, x2, y2)


def _validate_center(value: Any, image_size: Tuple[int, int], location: str) -> Tuple[int, int]:
    if not (isinstance(value, Sequence) and len(value) == 2):
        raise ContractValidationError(f"{location}: must be [x, y]")
    if any(float(number) != int(number) for number in value):
        raise ContractValidationError(f"{location}: center must use integer coordinates")
    x, y = (int(number) for number in value)
    width, height = image_size
    if not (0 <= x < width and 0 <= y < height):
        raise ContractValidationError(f"{location}: center {list(value)} is outside {width}x{height}")
    return (x, y)


def _validate_positive_float(value: Any, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"{location}: not a number: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ContractValidationError(f"{location}: must be a positive finite number")
    return number


def _validate_polygon(value: Any, image_size: Tuple[int, int], location: str) -> Polygon:
    if not isinstance(value, Sequence) or len(value) < 3:
        raise ContractValidationError(f"{location}: must have at least 3 points")
    width, height = image_size
    points = []
    for point in value:
        if not (isinstance(point, Sequence) and len(point) == 2):
            raise ContractValidationError(f"{location}: point must be [x, y]")
        x, y = (float(point[0]), float(point[1]))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ContractValidationError(f"{location}: coordinates must be finite")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ContractValidationError(f"{location}: point {[x, y]} is outside {width}x{height}")
        points.append((x, y))
    area_sum = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area_sum += x1 * y2 - x2 * y1
    if abs(area_sum) < 1e-6:
        raise ContractValidationError(f"{location}: polygon has zero area")
    return tuple(points)


def _canonical(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
