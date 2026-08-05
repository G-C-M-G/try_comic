"""Sentence splitting for the island pipeline (design section 2).

Layer 2.1 assigns OCR lines to islands (container authority), 2.2 clusters
lines into columns, 2.3 splits the island text at sentence boundaries, and
2.4 orders islands/columns/lines for reading.  Engine agnostic: works with
Windows OCR lines (no confidence) and Paddle lines (with confidence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .island_contract import IslandDataset, IslandInfo
from .island_dry_run import OcrLine
from .ocr_contract import BBox, Polygon


SENTENCE_END_CHARS = frozenset("。！？…」』」』！？!?")
PARTICLE_ENDINGS = frozenset(
    "かねよなぞわだですますだろうだろのもんからんだしなよな"
)
UNASSIGNED_ISLAND = "<unassigned>"


@dataclass(frozen=True)
class AssignedLine:
    line: OcrLine
    island_id: str
    center: Tuple[float, float]


@dataclass(frozen=True)
class ColumnGroup:
    island_id: str
    lines: Tuple[OcrLine, ...]
    direction: str
    bbox: BBox

    @property
    def text(self) -> str:
        return "".join(line.text for line in self.lines)


@dataclass(frozen=True)
class Sentence:
    island_id: str
    text: str
    column_ids: Tuple[str, ...]
    boundary_signal: str
    needs_review: bool


@dataclass(frozen=True)
class IslandSentenceUnit:
    island: IslandInfo
    columns: Tuple[ColumnGroup, ...]
    full_text: str
    sentences: Tuple[Sentence, ...]
    reading_order: Tuple[str, ...]


@dataclass(frozen=True)
class SentenceSplitResult:
    units: Tuple[IslandSentenceUnit, ...]
    unassigned: Tuple[OcrLine, ...]
    empty_islands: Tuple[IslandInfo, ...]
    conflicts: Tuple[Tuple[OcrLine, Tuple[str, ...]], ...]
    polygon_violations: Tuple[Tuple[OcrLine, str], ...] = ()


def line_center(line: OcrLine) -> Tuple[float, float]:
    xs = [point[0] for point in line.polygon]
    ys = [point[1] for point in line.polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _inside_bbox(x: float, y: float, bbox: BBox, tolerance: float = 2.0) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def assign_lines_to_islands(
    dataset: IslandDataset, lines: Sequence[OcrLine]
) -> Tuple[
    Tuple[AssignedLine, ...],
    Tuple[OcrLine, ...],
    Tuple[Tuple[OcrLine, Tuple[str, ...]], ...],
    Tuple[Tuple[OcrLine, str], ...],
]:
    """Layer 2.1: every line's center must fall inside exactly one island.

    The geometry authority is ``image_bbox``; ``image_polygon`` is an
    approximate display contour (per island contract it is not authoritative),
    so a center inside the bbox but outside the ragged polygon is still
    assigned and recorded as a soft ``polygon_violation`` instead of dropped.
    """
    assigned = []
    unassigned = []
    conflicts = []
    polygon_violations = []
    for line in lines:
        cx, cy = line_center(line)
        hits = []
        violation = None
        for island in dataset.islands:
            if not _inside_bbox(cx, cy, island.image_bbox):
                continue
            if island.polygon is not None and not _point_in_polygon(cx, cy, island.polygon):
                if violation is None:
                    violation = island.id
                continue
            hits.append(island.id)
        if not hits and violation is not None:
            hits = [violation]
            polygon_violations.append((line, violation))
        if not hits:
            unassigned.append(line)
        elif len(hits) > 1:
            conflicts.append((line, tuple(hits)))
        else:
            assigned.append(AssignedLine(line=line, island_id=hits[0], center=(cx, cy)))
    return tuple(assigned), tuple(unassigned), tuple(conflicts), tuple(polygon_violations)


def _bbox_of(lines: Sequence[OcrLine]) -> BBox:
    xs = [point[0] for line in lines for point in line.polygon]
    ys = [point[1] for line in lines for point in line.polygon]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def cluster_lines_into_columns(
    lines: Sequence[OcrLine], *, vertical: bool
) -> Tuple[ColumnGroup, ...]:
    """Layer 2.2: cluster lines sharing a band into one column.

    Thresholds mirror the validated page-level clustering: band overlap >= 0.45,
    gap <= max(30, 0.45 * min height), height ratio <= 1.6.
    """
    if not lines:
        return ()
    ordered = sorted(lines, key=lambda line: _order_key(line, vertical))
    clusters: list[list[OcrLine]] = []
    for line in ordered:
        if clusters:
            last = clusters[-1][-1]
            if _can_join(last, line, vertical):
                clusters[-1].append(line)
                continue
        clusters.append([line])
    groups = []
    for cluster in clusters:
        groups.append(
            ColumnGroup(
                island_id="",
                lines=tuple(cluster),
                direction="vertical" if vertical else "horizontal",
                bbox=_bbox_of(cluster),
            )
        )
    return tuple(groups)


def _order_key(line: OcrLine, vertical: bool) -> Tuple[float, float]:
    xs = [point[0] for point in line.polygon]
    ys = [point[1] for point in line.polygon]
    if vertical:
        return (min(xs), min(ys))
    return (min(ys), min(xs))


def _can_join(first: OcrLine, second: OcrLine, vertical: bool) -> bool:
    fx = [p[0] for p in first.polygon]
    fy = [p[1] for p in first.polygon]
    sx = [p[0] for p in second.polygon]
    sy = [p[1] for p in second.polygon]
    if vertical:
        band_a1, band_a2 = min(fx), max(fx)
        band_b1, band_b2 = min(sx), max(sx)
        pos_a1, pos_a2 = min(fy), max(fy)
        pos_b1, pos_b2 = min(sy), max(sy)
        size_a, size_b = pos_a2 - pos_a1, pos_b2 - pos_b1
    else:
        band_a1, band_a2 = min(fy), max(fy)
        band_b1, band_b2 = min(sy), max(sy)
        pos_a1, pos_a2 = min(fx), max(fx)
        pos_b1, pos_b2 = min(sx), max(sx)
        size_a, size_b = pos_a2 - pos_a1, pos_b2 - pos_b1
    overlap = max(0, min(band_a2, band_b2) - max(band_a1, band_b1))
    gap = max(0, pos_b1 - pos_a2) if pos_b1 > pos_a2 else max(0, pos_a1 - pos_b2)
    return (
        overlap / max(1.0, min(band_a2 - band_a1, band_b2 - band_b1)) >= 0.45
        and gap <= max(30, 0.45 * min(size_a, size_b))
        and max(size_a, size_b) / max(1.0, min(size_a, size_b)) <= 1.6
    )


def _line_end(text: str) -> str:
    return text.strip()[-1] if text.strip() else ""


def _split_on_strong_punctuation(text: str) -> Tuple[Tuple[str, ...], bool]:
    """Split *text* at sentence-end punctuation (including mid-line).

    A run of consecutive sentence-end chars (``！！``, ``……``) counts as ONE
    boundary at the last char of the run.  Returns ``(parts, ended)``: each
    part is a fragment of a sentence; a trailing fragment after the last
    strong char continues the next sentence.
    """
    parts = []
    current = []
    ended = False
    index = 0
    while index < len(text):
        char = text[index]
        current.append(char)
        if char in SENTENCE_END_CHARS:
            run_end = index
            while run_end + 1 < len(text) and text[run_end + 1] in SENTENCE_END_CHARS:
                run_end += 1
                current.append(text[run_end])
            if run_end == len(text) - 1:
                ended = True
                parts.append("".join(current))
                current = []
            elif text[run_end + 1] not in SENTENCE_END_CHARS:
                parts.append("".join(current))
                current = []
            index = run_end
        index += 1
    if current:
        parts.append("".join(current))
    return tuple(parts), ended


def _column_mean_confidence(column: ColumnGroup) -> Optional[float]:
    scores = [
        line.confidence for line in getattr(column, "lines", ()) if line.confidence is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def split_sentences(
    island: IslandInfo,
    columns: Sequence[ColumnGroup],
    *,
    full_text: str,
    strong_confidence: float = 0.9,
) -> Tuple[Sentence, ...]:
    """Layer 2.3: sentence boundaries from punctuation (strong), then a
    particle-ending fallback signal for the whole island.

    v1 rules: a strong sentence-end char (including mid-line) closes a
    sentence; particles never split mid-island (Japanese particles appear
    mid-sentence too often) but set the island signal to ``particle`` when
    the last fragment ends with one; no signal at all means ``low`` +
    review.  The island text itself is always translated as one unit.

    Strong splits carry a confidence gate: manga-ocr lines have no
    confidence (``None``) and split directly (text authority); Paddle
    fallback lines only split when the column's mean confidence reaches
    ``strong_confidence`` (default 0.90), otherwise the punctuation is
    treated as untrustworthy and the island is flagged for review.
    """
    if not columns:
        return ()
    ordered = sorted(columns, key=_column_key)
    sentences: list[Sentence] = []
    current_text: list[str] = []
    current_columns: list[str] = []
    last_signal = "low"
    for index, column in enumerate(ordered):
        column_id = f"{island.id}_c{index + 1}"
        confidence = _column_mean_confidence(column)
        split_allowed = confidence is None or confidence >= strong_confidence
        parts, ended = _split_on_strong_punctuation(column.text)
        if not parts:
            continue
        for part_index, part in enumerate(parts):
            current_text.append(part)
            current_columns.append(column_id)
            is_last_fragment = part_index == len(parts) - 1
            if not split_allowed:
                last_signal = "low"
                continue
            if part_index < len(parts) - 1:
                sentences.append(
                    Sentence(
                        island_id=island.id,
                        text="".join(current_text),
                        column_ids=tuple(current_columns),
                        boundary_signal="strong",
                        needs_review=False,
                    )
                )
                current_text = []
                current_columns = []
            elif ended and is_last_fragment:
                sentences.append(
                    Sentence(
                        island_id=island.id,
                        text="".join(current_text),
                        column_ids=tuple(current_columns),
                        boundary_signal="strong",
                        needs_review=False,
                    )
                )
                current_text = []
                current_columns = []
            else:
                if _line_end(part) in PARTICLE_ENDINGS:
                    last_signal = "particle"
                else:
                    last_signal = "low"
    if current_text:
        sentences.append(
            Sentence(
                island_id=island.id,
                text="".join(current_text),
                column_ids=tuple(current_columns),
                boundary_signal=last_signal,
                needs_review=last_signal == "low" or not split_allowed,
            )
        )
    if not sentences:
        sentences = [
            Sentence(
                island_id=island.id,
                text=full_text,
                column_ids=tuple(f"{island.id}_c{i + 1}" for i in range(len(ordered))),
                boundary_signal="low",
                needs_review=True,
            )
        ]
    return tuple(sentences)


def _column_key(column: ColumnGroup) -> Tuple[float, float]:
    x1, y1, x2, y2 = column.bbox
    if column.direction == "vertical":
        return (-x2, y1)
    return (y1, x1)


def _detect_direction(lines: Sequence[OcrLine]) -> bool:
    """True when the island text is horizontal.

    A vertical manga column has y-spread clearly dominating x-spread (line
    boxes stack down a column); when the spreads are comparable, majority
    y-band sharing between line pairs decides (true horizontal text shares
    its y band, slanted vertical columns do not).
    """
    if len(lines) < 2:
        box = lines[0].polygon
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (max(xs) - min(xs)) > (max(ys) - min(ys))
    boxes = []
    for line in lines:
        xs = [p[0] for p in line.polygon]
        ys = [p[1] for p in line.polygon]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    spread_x = max(box[2] for box in boxes) - min(box[0] for box in boxes)
    spread_y = max(box[3] for box in boxes) - min(box[1] for box in boxes)
    shared_y_pairs = 0
    total = 0
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            total += 1
            overlap = max(
                0, min(boxes[a][3], boxes[b][3]) - max(boxes[a][1], boxes[b][1])
            )
            height = min(boxes[a][3] - boxes[a][1], boxes[b][3] - boxes[b][1])
            if height > 0 and overlap / height >= 0.45:
                shared_y_pairs += 1
    if spread_y > spread_x * 1.5:
        return False
    return shared_y_pairs / max(1, total) >= 0.5


def split_island_dataset(
    dataset: IslandDataset,
    lines: Sequence[OcrLine],
) -> SentenceSplitResult:
    """Run layers 2.1-2.4 over one page."""
    assigned, unassigned, conflicts, polygon_violations = assign_lines_to_islands(dataset, lines)
    by_island: dict[str, list[AssignedLine]] = {}
    for item in assigned:
        by_island.setdefault(item.island_id, []).append(item)

    units = []
    empty = []
    for island in sorted(dataset.islands, key=lambda item: -item.image_bbox[2]):
        island_lines = by_island.get(island.id, [])
        if not island_lines:
            empty.append(island)
            continue
        vertical = not _detect_direction([item.line for item in island_lines])
        columns = cluster_lines_into_columns(
            [item.line for item in island_lines], vertical=vertical
        )
        ordered = sorted(columns, key=_column_key)
        full_text = "".join(column.text for column in ordered)
        sentences = split_sentences(island, columns, full_text=full_text)
        units.append(
            IslandSentenceUnit(
                island=island,
                columns=tuple(ordered),
                full_text=full_text,
                sentences=sentences,
                reading_order=tuple(
                    column_id for sentence in sentences for column_id in sentence.column_ids
                ),
            )
        )
    return SentenceSplitResult(
        units=tuple(units),
        unassigned=tuple(unassigned),
        empty_islands=tuple(empty),
        conflicts=tuple(conflicts),
        polygon_violations=tuple(polygon_violations),
    )
