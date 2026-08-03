"""Headless Japanese manga OCR with per-column geometry and repeat consensus.

The runtime path is fully image based: CTD detects text pixels and MangaOCR
recognizes each retained column. Windows Photos manifests are optional benchmark
inputs only; this module never opens a window or sends keyboard/mouse input.
"""

from __future__ import annotations

import argparse
import asyncio
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence
import unicodedata

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MITS_DIR = ROOT / "manga-image-translator"
sys.path.insert(0, str(MITS_DIR))

os.environ.setdefault("HF_HOME", r"D:\tools\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from manga_translator.config import Detector
from manga_translator.detection import dispatch as dispatch_detection
from manga_translator.utils import TextBlock
from manga_translator.utils.sort import sort_regions
from manga_ocr import MangaOcr
from text_segments import group_regions_into_sentences


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
_SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PADDLE_PYTHON = ROOT / "paddle_venv" / "Scripts" / "python.exe"
PADDLE_WORKER = ROOT / "paddle_ocr_worker.py"


def normalize_ocr_text(value: str) -> str:
    return _SPACE_RE.sub("", str(value or "")).strip()


def text_similarity(left: str, right: str) -> float:
    left, right = normalize_ocr_text(left), normalize_ocr_text(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def canonical_benchmark_text(value: str) -> str:
    """Normalize visually equivalent punctuation before OCR comparison."""
    value = unicodedata.normalize("NFKC", normalize_ocr_text(value))
    return re.sub(r"(?:\.{2,}|…+)", "…", value)


def levenshtein_distance(left: str, right: str) -> int:
    left, right = canonical_benchmark_text(left), canonical_benchmark_text(right)
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def choose_consensus(readings: Sequence[str]) -> tuple[str, float]:
    values = [normalize_ocr_text(value) for value in readings]
    if not values:
        return "", 0.0
    scores = [
        sum(text_similarity(value, other) for other in values) / len(values)
        for value in values
    ]
    best = max(range(len(values)), key=lambda index: (scores[index], len(values[index])))
    return values[best], float(scores[best])


def expand_bbox_10pct(bbox: Sequence[int], image_size: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    image_w, image_h = image_size
    dx = max(1, round((x2 - x1) * 0.05))
    dy = max(1, round((y2 - y1) * 0.05))
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(image_w, x2 + dx),
        min(image_h, y2 + dy),
    ]


def _groups(active: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    indices = np.flatnonzero(active)
    if not len(indices):
        return groups
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - previous > max_gap + 1:
            groups.append((start, previous + 1))
            start = value
        previous = value
    groups.append((start, previous + 1))
    return groups


def split_polygon_columns(
    polygon: np.ndarray,
    text_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Split a CTD polygon that spans several vertical manga columns."""
    image_h, image_w = image_shape
    points = np.asarray(polygon, dtype=np.int32).reshape(4, 2)
    x1 = max(0, int(points[:, 0].min()) - 3)
    y1 = max(0, int(points[:, 1].min()) - 3)
    x2 = min(image_w, int(points[:, 0].max()) + 4)
    y2 = min(image_h, int(points[:, 1].max()) + 4)
    # A normal vertical column can have several disconnected mask fragments
    # (punctuation, furigana, or broken strokes). Only wide CTD regions are
    # candidates for multi-column splitting.
    width, height = x2 - x1, y2 - y1
    if width / max(1, height) < 0.45:
        return [points]
    crop = (text_mask[y1:y2, x1:x2] > 60).astype(np.uint8)
    if not crop.size or not np.any(crop):
        return [points]

    active_x = np.any(crop, axis=0)
    rough = _groups(active_x, max_gap=max(2, round((y2 - y1) * 0.018)))
    # Fragments narrower than a punctuation stroke belong to the nearest band.
    min_width = max(3, round((y2 - y1) * 0.025))
    merged: list[list[int]] = []
    for left, right in rough:
        if merged and (right - left < min_width or left - merged[-1][1] <= 3):
            merged[-1][1] = right
        else:
            merged.append([left, right])
    if len(merged) < 2:
        return [points]

    output = []
    for left, right in merged:
        band = crop[:, left:right]
        rows = np.flatnonzero(np.any(band, axis=1))
        if not len(rows):
            continue
        bx1, bx2 = x1 + left, x1 + right
        by1, by2 = y1 + int(rows[0]), y1 + int(rows[-1]) + 1
        if (bx2 - bx1) * (by2 - by1) < 20:
            continue
        output.append(np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]))
    if len(output) < 2:
        return [points]
    band_aspects = [
        (item[:, 1].max() - item[:, 1].min())
        / max(1, item[:, 0].max() - item[:, 0].min())
        for item in output
    ]
    return output if float(np.median(band_aspects)) >= 2.0 else [points]


def polygon_bbox(points: np.ndarray) -> list[int]:
    points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    return [
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    ]


def intersection_over_smaller(left: Sequence[int], right: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    smaller = min(max(1, (ax2 - ax1) * (ay2 - ay1)), max(1, (bx2 - bx1) * (by2 - by1)))
    return float(intersection / smaller)


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda item: (item["detection_confidence"], -np.prod([
            item["image_bbox"][2] - item["image_bbox"][0],
            item["image_bbox"][3] - item["image_bbox"][1],
        ])),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in ordered:
        duplicate = next(
            (
                prior
                for prior in kept
                if intersection_over_smaller(candidate["image_bbox"], prior["image_bbox"]) >= 0.72
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
    return kept


def fuse_paddle_detections(
    ctd_candidates: list[dict],
    paddle_predictions: Sequence[dict],
    image_shape: tuple[int, int],
    minimum_confidence: float = 0.75,
) -> tuple[list[dict], list[dict]]:
    """Cross-reference CTD boxes and add credible Paddle-only candidates."""
    image_h, image_w = image_shape
    output = []
    for candidate in ctd_candidates:
        item = dict(candidate)
        item["detection_sources"] = ["ctd"]
        item["paddle_validation"] = []
        output.append(item)

    rejected = []
    for raw in paddle_predictions or []:
        points = np.asarray(raw.get("polygon", []), dtype=np.int32).reshape(-1, 2)
        text = normalize_ocr_text(raw.get("text", ""))
        confidence = float(raw.get("confidence", 0.0))
        if len(points) < 4:
            rejected.append({"text": text, "confidence": confidence, "reason": "invalid-polygon"})
            continue
        if len(points) != 4:
            points = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, image_w)
        points[:, 1] = np.clip(points[:, 1], 0, image_h)
        bbox = polygon_bbox(points)
        validation = {
            "text": text,
            "confidence": round(confidence, 4),
            "polygon": points.tolist(),
            "image_bbox": bbox,
        }
        ranked = sorted(
            (
                (intersection_over_smaller(bbox, item["image_bbox"]), index)
                for index, item in enumerate(output)
            ),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.45:
            overlap, index = ranked[0]
            validation["box_overlap"] = round(overlap, 4)
            output[index]["paddle_validation"].append(validation)
            if "paddleocr" not in output[index]["detection_sources"]:
                output[index]["detection_sources"].append("paddleocr")
            continue
        if not _JAPANESE_RE.search(text):
            rejected.append({**validation, "reason": "no-japanese-text"})
            continue
        if confidence < minimum_confidence:
            rejected.append({**validation, "reason": "low-paddle-confidence"})
            continue
        output.append({
            "polygon": points.tolist(),
            "image_bbox": bbox,
            "detection_confidence": confidence,
            "direction": "vertical" if (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0]) else "horizontal",
            "detection_sources": ["paddleocr"],
            "paddle_validation": [{**validation, "box_overlap": 1.0}],
            "supplemental": True,
        })
    return deduplicate_candidates(output), rejected


def choose_cross_validated_text(
    readings: Sequence[str],
    paddle_validation: Sequence[dict],
) -> tuple[str, float, str, dict]:
    manga_text, manga_confidence = choose_consensus(readings)
    paddle = max(
        paddle_validation or [],
        key=lambda item: (float(item.get("confidence", 0.0)), len(item.get("text", ""))),
        default=None,
    )
    comparison = {
        "paddle_text": paddle.get("text", "") if paddle else "",
        "paddle_confidence": round(float(paddle.get("confidence", 0.0)), 4) if paddle else None,
        "text_similarity": round(text_similarity(manga_text, paddle.get("text", "")), 4) if paddle else None,
        "agrees": canonical_benchmark_text(manga_text) == canonical_benchmark_text(paddle.get("text", ""))
        if paddle
        else None,
    }
    paddle_text = normalize_ocr_text(paddle.get("text", "")) if paddle else ""
    paddle_confidence = float(paddle.get("confidence", 0.0)) if paddle else 0.0
    if _JAPANESE_RE.search(manga_text):
        if comparison["agrees"]:
            comparison["decision"] = "cross-model-agreement"
            return manga_text, manga_confidence, "cross-model-agreement", comparison
        if (
            manga_confidence < 0.67
            and len(paddle_text) > 1
            and _JAPANESE_RE.search(paddle_text)
            and paddle_confidence >= 0.90
        ):
            comparison["decision"] = "paddleocr-low-manga-consensus"
            return paddle_text, paddle_confidence, "paddleocr-rescue", comparison
        comparison["decision"] = "manga-ocr-consensus"
        return manga_text, manga_confidence, "manga_ocr", comparison

    # A single Paddle pass can rescue a missed multi-character column, but not
    # a single-character candidate; the latter still requires MangaOCR repeats.
    if len(paddle_text) > 1 and _JAPANESE_RE.search(paddle_text) and paddle_confidence >= 0.90:
        comparison["decision"] = "paddleocr-empty-manga-result"
        return paddle_text, paddle_confidence, "paddleocr-rescue", comparison
    comparison["decision"] = "no-reliable-text"
    return manga_text, manga_confidence, "manga_ocr", comparison


def run_paddle_worker(source: Path, device: str = "cpu") -> dict[str, list[dict]]:
    if not PADDLE_PYTHON.is_file():
        raise RuntimeError(
            f"PaddleOCR environment is unavailable: {PADDLE_PYTHON}. "
            "Run with --no-paddle only for diagnostic fallback."
        )
    with tempfile.TemporaryDirectory(prefix="comic_paddle_") as temporary:
        output = Path(temporary) / "paddle.json"
        environment = os.environ.copy()
        environment.update({
            "HOME": str(ROOT / "models" / "paddle"),
            "USERPROFILE": str(ROOT / "models" / "paddle"),
            "PADDLE_PDX_CACHE_HOME": str(ROOT / "models" / "paddle"),
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1",
        })
        completed = subprocess.run(
            [str(PADDLE_PYTHON), str(PADDLE_WORKER), str(source), str(output), "--device", device],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(f"PaddleOCR worker failed ({completed.returncode}): {detail}")
        payload = json.loads(output.read_text(encoding="utf-8"))
    return {
        os.path.normcase(str(Path(page["source"]).resolve())): page.get("predictions", [])
        for page in payload.get("pages", [])
    }


def _crop(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    pad = max(2, round(min(x2 - x1, y2 - y1) * 0.08))
    return image[
        max(0, y1 - pad):min(image.shape[0], y2 + pad),
        max(0, x1 - pad):min(image.shape[1], x2 + pad),
    ]


def recognition_variants(crop: np.ndarray) -> list[Image.Image]:
    rgb = np.ascontiguousarray(crop.copy())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        Image.fromarray(rgb),
        Image.fromarray(cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)),
        Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)),
    ]


def _find_manga_ocr_snapshot() -> Path:
    hub = Path(os.environ["HF_HOME"]) / "hub" / "models--kha-white--manga-ocr-base"
    ref = hub / "refs" / "main"
    if ref.is_file():
        snapshot = hub / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").is_file():
            return snapshot
    for snapshot in sorted((hub / "snapshots").glob("*"), reverse=True):
        if (snapshot / "config.json").is_file() and (snapshot / "preprocessor_config.json").is_file():
            return snapshot
    raise RuntimeError("Local MangaOCR snapshot is unavailable")


def _geometry_key(points: Sequence) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(points).reshape(-1))


def build_translation_units(columns: list[dict], image: np.ndarray) -> tuple[list[dict], list[str]]:
    """Create sentence-level units while retaining every source column box."""
    if not columns:
        return [], []

    regions = []
    by_geometry = {}
    for column in columns:
        polygon = np.asarray(column["polygon"], dtype=np.int32).reshape(1, 4, 2)
        width = column["image_bbox"][2] - column["image_bbox"][0]
        region = TextBlock(
            lines=polygon,
            texts=[column["text"]],
            font_size=max(12, round(width * 0.88)),
            direction="v" if column["direction"] == "vertical" else "h",
            source_lang="JPN",
            target_lang="CHS",
        )
        region.column_id = column["id"]
        regions.append(region)
        by_geometry[_geometry_key(polygon)] = column

    ordered = sort_regions(regions, right_to_left=True, img=image)
    reading_order = [region.column_id for region in ordered]
    for order, region in enumerate(ordered):
        column = next(item for item in columns if item["id"] == region.column_id)
        column["selection_order"] = order
        column["panel_index"] = getattr(region, "panel_index", None)

    units = []
    for index, sentence in enumerate(group_regions_into_sentences(ordered), 1):
        segments = list(getattr(sentence, "render_segments", None) or [])
        if not segments:
            segments = [{"lines": np.asarray(sentence.lines).reshape(-1, 4, 2).tolist()}]
        unit_columns = []
        for segment in segments:
            column = by_geometry.get(_geometry_key(segment["lines"]))
            if column is None:
                continue
            unit_columns.append({
                "column_id": column["id"],
                "source_text": column["text"],
                "polygon": column["polygon"],
                "image_bbox": column["image_bbox"],
                "layout_bbox_10pct": column["layout_bbox_10pct"],
            })
        units.append({
            "id": f"sentence_{index:03d}",
            "column_ids": [item["column_id"] for item in unit_columns],
            "source_text": str(sentence.text or ""),
            "translation_query": str(
                getattr(sentence, "translation_query", None) or sentence.text or ""
            ),
            "panel_index": getattr(sentence, "panel_index", None),
            "layout_regions": unit_columns,
        })
    return units, reading_order


class BackgroundMangaOCR:
    def __init__(self, device: str = "cuda"):
        self.device = device if device == "cpu" or _cuda_available() else "cpu"
        self.recognizer: MangaOcr | None = None

    def _recognizer(self) -> MangaOcr:
        if self.recognizer is None:
            self.recognizer = MangaOcr(
                str(_find_manga_ocr_snapshot()),
                force_cpu=self.device == "cpu",
            )
        return self.recognizer

    async def detect(self, image: np.ndarray, box_threshold: float = 0.30):
        lines, text_mask, _ = await dispatch_detection(
            Detector.ctd,
            image,
            1536,
            0.5,
            box_threshold,
            1.5,
            False,
            False,
            True,
            False,
            self.device,
            False,
        )
        candidates = []
        for line in lines:
            for polygon in split_polygon_columns(line.pts, text_mask, image.shape[:2]):
                bbox = polygon_bbox(polygon)
                candidates.append({
                    "polygon": np.asarray(polygon, dtype=np.int32).tolist(),
                    "image_bbox": bbox,
                    "detection_confidence": float(line.prob),
                    "direction": "vertical" if (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0]) else "horizontal",
                })
        return deduplicate_candidates(candidates), text_mask

    async def process_image(
        self,
        path: str,
        box_threshold: float = 0.30,
        paddle_predictions: Sequence[dict] | None = None,
    ) -> dict:
        source = str(Path(path).resolve())
        image = np.array(Image.open(source).convert("RGB"))
        ctd_candidates, _ = await self.detect(image, box_threshold)
        candidates, paddle_rejected = fuse_paddle_detections(
            ctd_candidates, paddle_predictions or [], image.shape[:2]
        )
        recognizer = self._recognizer()
        columns = []
        ignored = []
        for candidate in candidates:
            crop = _crop(image, candidate["image_bbox"])
            readings = [normalize_ocr_text(recognizer(variant)) for variant in recognition_variants(crop)]
            text, confidence, text_source, cross_validation = choose_cross_validated_text(
                readings, candidate.get("paddle_validation", [])
            )
            record = {
                **candidate,
                "readings": readings,
                "text": text,
                "text_source": text_source,
                "recognition_confidence": round(confidence, 4),
                "cross_validation": cross_validation,
                "layout_bbox_10pct": expand_bbox_10pct(
                    candidate["image_bbox"], (image.shape[1], image.shape[0])
                ),
            }
            if not _JAPANESE_RE.search(text):
                ignored.append({**record, "reason": "no-japanese-text"})
                continue
            if len(text) == 1 and confidence < 0.67:
                ignored.append({**record, "reason": "low-confidence-single-character"})
                continue
            columns.append(record)

        columns.sort(key=lambda item: (item["image_bbox"][1], -item["image_bbox"][2]))
        for index, column in enumerate(columns, 1):
            column["id"] = f"column_{index:03d}"
            column["role"] = "text"
        translation_units, reading_order = build_translation_units(columns, image)
        columns.sort(key=lambda item: item["selection_order"])
        paddle_enabled = paddle_predictions is not None
        fusion_summary = {
            "ctd_candidates": len(ctd_candidates),
            "paddle_candidates": len(paddle_predictions or []),
            "cross_validated_columns": sum(
                "paddleocr" in column.get("detection_sources", []) for column in columns
            ),
            "text_agreements": sum(
                column.get("cross_validation", {}).get("agrees") is True for column in columns
            ),
            "text_disagreements": sum(
                column.get("cross_validation", {}).get("agrees") is False for column in columns
            ),
            "paddle_supplemented_columns": sum(
                bool(column.get("supplemental")) for column in columns
            ),
            "paddle_rejected_candidates": len(paddle_rejected),
        }
        return {
            "schema_version": 4,
            "backend": (
                "Headless CTD + MangaOCR + PaddleOCR cross-validation"
                if paddle_enabled
                else "Headless CTD + MangaOCR consensus"
            ),
            "source": {"path": source, "width": image.shape[1], "height": image.shape[0]},
            "capture": {
                "headless": True,
                "opens_windows": False,
                "controls_mouse": False,
                "repeat_count": 3,
                "paddleocr_enabled": paddle_enabled,
                "paddleocr_model": "PP-OCRv6 medium",
                "single_character_confidence_threshold": 0.67,
                "layout_overflow_allowance": 0.10,
            },
            "columns": columns,
            "reading_order": reading_order,
            "translation_units": translation_units,
            "ignored_regions": ignored,
            "fusion_summary": fusion_summary,
            "paddle_rejected_regions": paddle_rejected,
        }


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def benchmark_against_photos(result: dict, photos: dict) -> dict:
    predictions = result.get("columns", [])
    references = photos.get("columns", [])
    unmatched = set(range(len(predictions)))
    matches = []
    for reference in references:
        ranked = sorted(
            (
                (intersection_over_smaller(reference["image_bbox"], predictions[index]["image_bbox"]), index)
                for index in unmatched
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.45:
            continue
        overlap, index = ranked[0]
        unmatched.remove(index)
        prediction = predictions[index]
        matches.append({
            "reference_id": reference["id"],
            "prediction_id": prediction["id"],
            "reference_text": reference["text"],
            "prediction_text": prediction["text"],
            "box_overlap": round(overlap, 4),
            "text_similarity": round(text_similarity(reference["text"], prediction["text"]), 4),
            "character_errors": levenshtein_distance(reference["text"], prediction["text"]),
        })
    matched_ids = {item["reference_id"] for item in matches}
    missing = [item for item in references if item["id"] not in matched_ids]
    multi_missing = [item for item in missing if len(normalize_ocr_text(item["text"])) > 1]
    total_chars = sum(
        len(canonical_benchmark_text(item["text"])) for item in references
    )
    character_errors = sum(item["character_errors"] for item in matches) + sum(
        len(canonical_benchmark_text(item["text"])) for item in missing
    )
    character_error_rate = character_errors / max(1, total_chars)
    character_accuracy = max(0.0, 1.0 - character_error_rate)
    box_recall = len(matches) / max(1, len(references))
    box_precision = len(matches) / max(1, len(predictions))
    box_rate = (
        2 * box_precision * box_recall / (box_precision + box_recall)
        if box_precision + box_recall
        else 0.0
    )
    return {
        "reference_backend": photos.get("backend"),
        "reference_columns": len(references),
        "predicted_columns": len(predictions),
        "matched_columns": len(matches),
        "text_box_precision": round(box_precision, 4),
        "text_box_recall": round(box_recall, 4),
        "text_box_match_rate": round(box_rate, 4),
        "character_errors": character_errors,
        "reference_characters": total_chars,
        "character_error_rate": round(character_error_rate, 4),
        "character_similarity": round(character_accuracy, 4),
        "missing_multi_character_columns": [item["id"] for item in multi_missing],
        "passes": {
            "no_missing_multi_character_columns": not multi_missing,
            "character_error_rate_at_most_10_percent": character_error_rate <= 0.10,
            "character_similarity_at_least_90_percent": character_accuracy >= 0.90,
            "text_box_match_rate_at_least_90_percent": box_rate >= 0.90,
        },
        "matches": matches,
        "missing_references": missing,
        "unmatched_predictions": [predictions[index]["id"] for index in sorted(unmatched)],
    }


def _images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if child.suffix.lower() in _SUPPORTED_IMAGES:
            yield child


async def _run(args) -> int:
    engine = BackgroundMangaOCR(args.device)
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    image_paths = list(_images(source))
    if not image_paths:
        raise RuntimeError(f"No supported images found in {source}")
    paddle_by_source = (
        {} if args.no_paddle else await asyncio.to_thread(run_paddle_worker, source, args.paddle_device)
    )
    output_is_file = len(image_paths) == 1 and output.suffix.lower() == ".json"
    if not output_is_file:
        output.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        paddle_predictions = (
            None
            if args.no_paddle
            else paddle_by_source.get(os.path.normcase(str(image_path.resolve())), [])
        )
        result = await engine.process_image(
            str(image_path), args.box_threshold, paddle_predictions
        )
        target = output if output_is_file else output / f"{image_path.stem}.ocr.json"
        if args.photos_baseline and len(image_paths) == 1:
            photos = json.loads(Path(args.photos_baseline).read_text(encoding="utf-8"))
            result["benchmark"] = benchmark_against_photos(result, photos)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless per-column Japanese manga OCR")
    parser.add_argument("input", help="Image or image directory")
    parser.add_argument("output", help="JSON file or output directory")
    parser.add_argument("--photos-baseline", help="Optional Windows Photos manifest for metrics")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--paddle-device", choices=["cpu", "gpu:0"], default="cpu")
    parser.add_argument("--no-paddle", action="store_true", help="Diagnostic CTD/MangaOCR-only mode")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
