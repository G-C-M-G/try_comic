"""Page state and re-embedding (M4 wrap-up).

``{page}.state.json`` snapshots one full pipeline run: input hashes (original,
islands, erased image, translations), the per-island geometry derived from
Paddle lines, and the per-island typeset audit.  Re-embedding takes the state
plus an optional translation edit, validates the input hashes, and re-runs
only the typeset step (no OCR, no erasure, no translation).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .errors import StateError
from .island_contract import IslandDataset
from .island_dry_run import OcrLine
from .island_typeset import IslandTypesetResult, TypesetConfig, typeset_island_page
from .ocr_contract import sha256_file
from .sentence_splitter import ColumnGroup

STATE_FORMAT_VERSION = 1


def _canonical_sha256(data) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _island_geometry_record(dataset: IslandDataset, island_id: str, paddle_lines: Sequence[OcrLine]):
    from .island_typeset import _island_geometry
    from .island_contract import IslandInfo

    island = dataset.islands_by_id[island_id]
    area_base, area_rect, centre, has_geometry, columns = _island_geometry(island, paddle_lines)
    return {
        "area_base": area_base,
        "area_rect": area_rect,
        "centre": list(centre) if centre is not None else None,
        "has_geometry": has_geometry,
        "columns": [list(column.bbox) for column in columns],
    }


def build_page_state(
    *,
    page: str,
    original_path: Path,
    erased_path: Path,
    dataset: IslandDataset,
    translations_payload: Mapping,
    audits: Sequence,
    paddle_lines: Sequence[OcrLine],
) -> dict:
    """Build the serialisable state dict for one page."""
    translations_sha = _canonical_sha256(
        {
            item["island_id"]: item.get("translated_text", "")
            for item in translations_payload.get("islands", [])
        }
    )
    islands = []
    for audit in audits:
        if audit.status == "skipped":
            continue
        record = {
            "island_id": audit.island_id,
            "source_text": audit.text,
            "translated_text": audit.text,
            "status": audit.status,
            "reasons": list(audit.reasons),
            "font_size": audit.font_size,
            "stroke_width": audit.stroke_width,
            "tracking": audit.tracking,
            "column_gap": audit.column_gap,
            "area_base": audit.area_base,
            "area_text": audit.area_text,
            "area_ratio": audit.area_ratio,
            "center_offset": audit.center_offset,
            "center_tolerance": audit.center_tolerance,
        }
        if audit.block_bbox is not None:
            record["block_bbox"] = list(audit.block_bbox)
        if audit.slots:
            record["slots"] = [
                {
                    "index": slot.index,
                    "x_center": slot.x_center,
                    "y_center": slot.y_center,
                    "char_count": slot.char_count,
                }
                for slot in audit.slots
            ]
        record["geometry"] = _island_geometry_record(dataset, audit.island_id, paddle_lines)
        islands.append(record)
    return {
        "format_version": STATE_FORMAT_VERSION,
        "page": page,
        "inputs": {
            "original_sha256": sha256_file(Path(original_path)),
            "islands_sha256": dataset.effective_sha256,
            "erased_sha256": sha256_file(Path(erased_path)),
            "translations_sha256": translations_sha,
        },
        "islands": islands,
    }


def write_page_state(state: dict, output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_page_state(state_path: Path) -> dict:
    path = Path(state_path).resolve()
    if not path.is_file():
        raise StateError(f"state file does not exist: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"cannot read state {path}: {error}") from error
    if state.get("format_version") != STATE_FORMAT_VERSION:
        raise StateError(f"unsupported state format version: {state.get('format_version')}")
    return state


def validate_state_inputs(state: dict, dataset: IslandDataset, erased_path: Path) -> None:
    """Refuse to re-embed when the islands or the erased image changed."""
    inputs = state.get("inputs", {})
    if dataset.effective_sha256 != inputs.get("islands_sha256"):
        raise StateError(
            "islands.json hash does not match the state; refusing to re-embed stale text"
        )
    if sha256_file(Path(erased_path)) != inputs.get("erased_sha256"):
        raise StateError(
            "erased image hash does not match the state; re-run the pipeline instead"
        )


def _columns_from_bboxes(bboxes: Sequence[Sequence[float]]) -> Sequence[ColumnGroup]:
    return tuple(
        ColumnGroup(
            island_id="", lines=(), direction="vertical",
            bbox=tuple(int(round(value)) for value in bbox),
        )
        for bbox in bboxes
    )


def reembed_from_state(
    state: dict,
    dataset: IslandDataset,
    erased: "Image.Image",
    *,
    edits: Optional[Mapping[str, str]] = None,
    config: TypesetConfig = TypesetConfig(),
) -> IslandTypesetResult:
    """Re-run only the typeset step from a saved state.

    *edits* overrides individual island translations; unedited islands keep
    the state's translated text.
    """
    from PIL import Image

    if erased.size != dataset.image_size:
        raise StateError(
            f"erased image size {erased.size} differs from islands.json {dataset.image_size}"
        )
    base = {
        item["island_id"]: str(item.get("translated_text") or "")
        for item in state.get("islands", [])
    }
    base.update(edits or {})
    overrides = {}
    for item in state.get("islands", []):
        island_id = item["island_id"]
        geometry = item.get("geometry") or {}
        columns = _columns_from_bboxes(geometry.get("columns") or [])
        overrides[island_id] = (
            geometry.get("area_base"),
            geometry.get("area_rect"),
            tuple(geometry["centre"]) if geometry.get("centre") else None,
            bool(geometry.get("has_geometry", False)),
            columns,
        )
    return typeset_island_page(
        erased,
        dataset,
        base,
        (),
        config=config,
        geometry_overrides=overrides,
    )
