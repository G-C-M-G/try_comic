"""Two-engine cross validation: manga-ocr text + Paddle geometry.

Text authority  = manga-ocr  (whole-island reading, no confidence)
Geometry source = Paddle    (island-crop lines: boxes + text + confidence)

Per island the two engines corroborate each other; any divergence, low
confidence or missing manga text pushes the island to the review queue with
the manga reading kept as the working text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from .island_contract import IslandDataset, IslandInfo
from .island_dry_run import OcrLine
from .sentence_splitter import Sentence, IslandSentenceUnit, split_island_dataset

DEFAULT_MIN_ISLAND_AREA = 800.0
DEFAULT_EDIT_DISTANCE_THRESHOLD = 0.3
DEFAULT_LOW_CONFIDENCE = 0.6
DEFAULT_STRONG_CONFIDENCE = 0.9

_DROPPED = "dropped_tiny"
_PASS = "pass"
_REVIEW = "review"


@dataclass(frozen=True)
class IslandVerdict:
    island_id: str
    status: str  # pass | review | dropped_tiny
    manga_text: str
    paddle_columns: Tuple[str, ...]  # joined column texts
    paddle_lines: Tuple[OcrLine, ...]
    edit_distance: Optional[float]
    low_confidence_lines: Tuple[str, ...]
    review_reasons: Tuple[str, ...]
    empty_paddle: bool
    dropped_reason: Optional[str] = None


@dataclass(frozen=True)
class PageCrossValidation:
    page: str
    image_size: Tuple[int, int]
    verdicts: Tuple[IslandVerdict, ...]
    dropped_tiny: Tuple[IslandInfo, ...]
    pass_count: int
    review_count: int
    dropped_count: int


def _normalized_edit_distance(first: str, second: str) -> float:
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0
    first = first.strip()
    second = second.strip()
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, 1):
        current = [i]
        for j, char_b in enumerate(second, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if char_a == char_b else 1),
                )
            )
        previous = current
    return previous[len(second)] / max(len(first), len(second))


_PUNCTUATION_FOR_MATCH = frozenset(
    " 、。！？…‥．.・,，!?\"'「」『』”’‘«»~〜ー-–—_()（）[]【】:：;；"
)


def _normalize_for_match(text: str) -> str:
    return "".join(char for char in text if char not in _PUNCTUATION_FOR_MATCH)


def _paddle_contained_in_manga(manga_text: str, paddle_fragments: Sequence[str]) -> bool:
    """True when every paddle fragment appears inside the manga reading.

    Paddle island crops emit the sentence's own fragments, usually in a
    different order; the manga reading is the authority for order, so
    containment (after punctuation normalization) counts as agreement.
    Fragments shorter than two characters are ignored as evidence; when no
    fragment qualifies, containment is not established (caller falls back to
    the edit distance).
    """
    if not manga_text:
        return False
    normalized = _normalize_for_match(manga_text)
    if not normalized:
        return False
    matched_any = False
    for fragment in paddle_fragments:
        fragment = _normalize_for_match(fragment)
        if len(fragment) < 2:
            continue
        if fragment not in normalized:
            return False
        matched_any = True
    return matched_any


def _containing_island_id(
    dataset: IslandDataset, cx: float, cy: float, tolerance: float = 2.0
) -> Optional[str]:
    """Smallest-area island whose bbox contains the point.

    Expanded bboxes can overlap (a tiny island's dots may fall inside a
    neighbour's corrected bbox); the smallest container is the tightest
    geometry, so it wins.
    """
    hits = [
        island
        for island in dataset.islands
        if _point_in_bbox(cx, cy, island.image_bbox, tolerance)
    ]
    if not hits:
        return None
    return min(hits, key=lambda island: (island.image_area, dataset.islands.index(island))).id


def _point_in_bbox(x: float, y: float, bbox, tolerance: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def _paddle_text_per_island(
    dataset: IslandDataset, paddle_lines: Sequence[OcrLine]
) -> Mapping[str, list[OcrLine]]:
    result: dict[str, list[OcrLine]] = {island.id: [] for island in dataset.islands}
    result["<unassigned>"] = []
    for line in paddle_lines:
        cx = sum(point[0] for point in line.polygon) / len(line.polygon)
        cy = sum(point[1] for point in line.polygon) / len(line.polygon)
        island_id = getattr(line, "island_id", None)
        if not island_id or island_id not in result:
            island_id = _containing_island_id(dataset, cx, cy)
        if not island_id:
            island_id = "<unassigned>"
        result[island_id].append(line)
    return result


def cross_validate_page(
    dataset: IslandDataset,
    manga_lines: Sequence[OcrLine],
    paddle_lines: Sequence[OcrLine],
    *,
    min_island_area: float = DEFAULT_MIN_ISLAND_AREA,
    edit_distance_threshold: float = DEFAULT_EDIT_DISTANCE_THRESHOLD,
    low_confidence: float = DEFAULT_LOW_CONFIDENCE,
) -> PageCrossValidation:
    """One page: manga text (authority) vs Paddle island-crop evidence."""
    manga_by_island = {island.id: "" for island in dataset.islands}
    for line in manga_lines:
        text = str(line.text or "").strip()
        if not text:
            continue
        cx = sum(point[0] for point in line.polygon) / len(line.polygon)
        cy = sum(point[1] for point in line.polygon) / len(line.polygon)
        island_id = _containing_island_id(dataset, cx, cy)
        if island_id:
            manga_by_island[island_id] = text

    paddle_by_island = _paddle_text_per_island(dataset, paddle_lines)

    verdicts = []
    dropped = []
    for island in dataset.islands:
        if island.image_area < min_island_area:
            dropped.append(island)
            verdicts.append(
                IslandVerdict(
                    island_id=island.id,
                    status=_DROPPED,
                    manga_text="",
                    paddle_columns=(),
                    paddle_lines=(),
                    edit_distance=None,
                    low_confidence_lines=(),
                    review_reasons=(),
                    empty_paddle=True,
                    dropped_reason=(
                        f"image_area={island.image_area:.1f} < min_island_area={min_island_area}"
                    ),
                )
            )
            continue

        manga_text = manga_by_island.get(island.id, "")
        paddle_items = paddle_by_island.get(island.id, [])
        paddle_fragments = tuple(
            str(item.text or "").strip()
            for item in paddle_items
            if str(item.text or "").strip()
        )
        paddle_text = "".join(paddle_fragments)
        low_lines = tuple(
            str(item.text or "")
            for item in paddle_items
            if item.confidence is not None and item.confidence < low_confidence
        )
        reasons = []
        if not manga_text:
            reasons.append("manga-ocr empty")
        if not paddle_items:
            reasons.append("paddle empty")
        distance = None
        if manga_text and paddle_text:
            if _paddle_contained_in_manga(manga_text, paddle_fragments):
                distance = 0.0
            else:
                distance = _normalized_edit_distance(
                    _normalize_for_match(manga_text),
                    _normalize_for_match(paddle_text),
                )
                if distance > edit_distance_threshold:
                    reasons.append(f"engines diverge (edit={distance:.2f})")
        if low_lines:
            reasons.append(f"low-confidence lines: {len(low_lines)}")
        status = _PASS if not reasons else _REVIEW
        verdicts.append(
            IslandVerdict(
                island_id=island.id,
                status=status,
                manga_text=manga_text,
                paddle_columns=(
                    tuple(
                        str(item.text or "") for item in paddle_items if str(item.text or "").strip()
                    )
                ),
                paddle_lines=tuple(paddle_items),
                edit_distance=distance,
                low_confidence_lines=low_lines,
                review_reasons=tuple(reasons),
                empty_paddle=not paddle_items,
            )
        )

    return PageCrossValidation(
        page=dataset.page,
        image_size=dataset.image_size,
        verdicts=tuple(verdicts),
        dropped_tiny=tuple(dropped),
        pass_count=sum(v.status == _PASS for v in verdicts),
        review_count=sum(v.status == _REVIEW for v in verdicts),
        dropped_count=sum(v.status == _DROPPED for v in verdicts),
    )


def render_verdict_markdown(result: PageCrossValidation) -> str:
    lines = []
    lines.append(f"# {result.page} 页 双引擎交叉验证（manga-ocr 文本 × Paddle 几何）")
    lines.append("")
    lines.append(
        f"- 状态: 通过 {result.pass_count} / 核对 {result.review_count} / 微岛舍弃 {result.dropped_count}"
    )
    lines.append("")
    lines.append("| 岛 | manga 文本(权威) | Paddle 列(几何) | 编辑距离 | 低置信行 | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for verdict in result.verdicts:
        paddle_text = " | ".join(verdict.paddle_columns) or "—"
        distance = "" if verdict.edit_distance is None else f"{verdict.edit_distance:.2f}"
        low = "、".join(verdict.low_confidence_lines) if verdict.low_confidence_lines else "—"
        if verdict.status == "dropped_tiny":
            lines.append(f"| {verdict.island_id} | — | — | — | — | 舍弃（{verdict.dropped_reason}） |")
        else:
            mark = "✅ 通过" if verdict.status == "pass" else "⚠️ REVIEW"
            lines.append(
                f"| {verdict.island_id} | {verdict.manga_text or '—'} | {paddle_text} | "
                f"{distance} | {low} | {mark} |"
            )
    lines.append("")
    review = [v for v in result.verdicts if v.status == "review"]
    if review:
        lines.append("## REVIEW 原因明细")
        lines.append("")
        for verdict in review:
            lines.append(f"- **{verdict.island_id}**: {verdict.review_reasons}")
            if verdict.manga_text:
                lines.append(f"  - manga: {verdict.manga_text}")
            if verdict.paddle_columns:
                lines.append(f"  - paddle: {verdict.paddle_columns}")
    lines.append("")
    return "\n".join(lines)


def verdicts_to_json(result: PageCrossValidation) -> dict:
    return {
        "page": result.page,
        "image_size": list(result.image_size),
        "pass_count": result.pass_count,
        "review_count": result.review_count,
        "dropped_count": result.dropped_count,
        "verdicts": [
            {
                "island_id": v.island_id,
                "status": v.status,
                "manga_text": v.manga_text,
                "paddle_columns": list(v.paddle_columns),
                "edit_distance": v.edit_distance,
                "low_confidence_lines": list(v.low_confidence_lines),
                "review_reasons": list(v.review_reasons),
                "dropped_reason": v.dropped_reason,
            }
            for v in result.verdicts
        ],
    }
