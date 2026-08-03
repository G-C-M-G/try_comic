"""Build sentence-level translation units without losing source geometry."""
import copy
import re

import numpy as np


_SENTENCE_RE = re.compile(r".+?(?:[。！？!?]+|$)")
_TERMINAL_RE = re.compile(r"[。！？!?]+\s*$")
_JAPANESE_TERMINAL_RE = re.compile(r"(?:[。！？!?]+|…+|[.．]{2,})[」』）)]*\s*$")
_CHINESE_CHUNK_RE = re.compile(r".+?(?:[，、；：。！？!?]+|…{2,}|[.．]{2,}|$)")
_SEGMENT_MARKER_RE = re.compile(r"<\s*(\d+)\s*>")
_GEOMETRY_CACHE_KEYS = (
    "xyxy", "xywh", "center", "unrotated_polygons", "unrotated_min_rect",
    "min_rect", "polygon_aspect_ratio", "unrotated_size", "aspect_ratio",
    "polygon_object", "area", "real_area",
)


def has_sentence_terminal(text):
    """Return whether an OCR block ends a Japanese sentence."""
    return bool(_JAPANESE_TERMINAL_RE.search(str(text or "").strip()))


def _clear_geometry_cache(region):
    for key in _GEOMETRY_CACHE_KEYS:
        region.__dict__.pop(key, None)


def _render_segment(region):
    return {
        "text": str(region.text or ""),
        "lines": np.asarray(region.lines, dtype=np.int32).reshape(-1, 4, 2).tolist(),
        "font_size": int(region.font_size),
        "fg": [int(c) for c in region.fg_colors],
        "bg": [int(c) for c in region.bg_colors],
        "direction": region.direction,
        "alignment": region.alignment,
    }


def _merge_sentence_regions(regions):
    merged = copy.copy(regions[0])
    merged.lines = np.concatenate(
        [np.asarray(region.lines, dtype=np.int32).reshape(-1, 4, 2) for region in regions]
    )
    merged.texts = [
        str(text or "")
        for region in regions
        for text in (list(getattr(region, "texts", None) or []) or [region.text])
    ]
    merged.text = "".join(str(region.text or "") for region in regions)
    merged.translation_query = "".join(
        f"<{index}>{region.text}" for index, region in enumerate(regions, 1)
    )
    merged.translation = ""
    merged.render_segments = [_render_segment(region) for region in regions]
    merged.translation_segments = None
    merged.sentence_grouped = True
    merged.font_size = min(int(region.font_size) for region in regions)
    merged.prob = min(float(region.prob) for region in regions)
    _clear_geometry_cache(merged)
    return merged


def _geometry_can_continue(left, right):
    """Conservative fallback when panel detection did not assign containers."""
    lx1, ly1, lx2, ly2 = (float(v) for v in left.xyxy)
    rx1, ry1, rx2, ry2 = (float(v) for v in right.xyxy)
    overlap = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    min_height = max(1.0, min(ly2 - ly1, ry2 - ry1))
    horizontal_gap = max(0.0, max(lx1, rx1) - min(lx2, rx2))
    max_width = max(lx2 - lx1, rx2 - rx1)
    return overlap / min_height >= 0.25 and horizontal_gap <= max(40.0, max_width * 1.5)


def group_regions_into_sentences(regions):
    """Join consecutive OCR regions in a panel until terminal punctuation.

    MITS has already sorted regions in manga reading order and assigned a
    ``panel_index``.  A panel change is always a hard boundary; within one
    panel, ``。！？`` and ellipses close the current sentence.
    """
    output = []
    current = []
    current_panel = None

    def flush():
        nonlocal current
        if len(current) == 1:
            output.append(current[0])
        elif current:
            output.append(_merge_sentence_regions(current))
        current = []

    for region in regions or []:
        panel = getattr(region, "panel_index", None)
        if current and panel != current_panel:
            flush()
        elif current and panel is None and current_panel is None \
                and not _geometry_can_continue(current[-1], region):
            flush()
        if not current:
            current_panel = panel
        current.append(region)
        if has_sentence_terminal(region.text):
            flush()
            current_panel = None
    flush()
    return output


def make_sentence_grouping_hook(base_hook=None, enable=True):
    """Compose existing OCR filtering with sentence-level grouping."""
    def hook(ctx):
        regions = ctx.text_regions
        if base_hook is not None:
            filtered = base_hook(ctx)
            if filtered is not None:
                regions = filtered
        if not enable:
            return regions
        return group_regions_into_sentences(regions)

    return hook


def split_translation_for_segments(translation, source_texts):
    """Split one Chinese translation back over its original render boxes."""
    sources = [str(text or "").strip() for text in source_texts]
    count = len(sources)
    raw_text = str(translation or "").strip()
    marked = list(_SEGMENT_MARKER_RE.finditer(raw_text))
    if len(marked) == count and [int(match.group(1)) for match in marked] == list(range(1, count + 1)):
        parts = []
        for index, match in enumerate(marked):
            end = marked[index + 1].start() if index + 1 < len(marked) else len(raw_text)
            parts.append(raw_text[match.end():end].strip())
        if all(parts):
            return parts

    text = _SEGMENT_MARKER_RE.sub("", raw_text).strip()
    if count < 2 or not text:
        return [text] if count == 1 and text else None

    chunks = [match.group(0).strip() for match in _CHINESE_CHUNK_RE.finditer(text)]
    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) == count:
        return chunks

    weights = [max(1, len(source)) for source in sources]
    targets = np.cumsum(weights)[:-1] / sum(weights) * len(text)
    punctuation = [
        match.end() for match in re.finditer(r"[，、；：。！？!?…]+", text)
        if match.end() < len(text)
    ]
    boundaries = []
    previous = 0
    for target in targets:
        candidates = [position for position in punctuation if previous < position < len(text)]
        if candidates:
            boundary = min(candidates, key=lambda position: abs(position - target))
        else:
            boundary = max(previous + 1, min(len(text) - 1, round(target)))
        boundaries.append(boundary)
        previous = boundary

    parts = []
    start = 0
    for boundary in boundaries + [len(text)]:
        parts.append(text[start:boundary].strip())
        start = boundary
    return parts if len(parts) == count and all(parts) else None


def align_translation_segments(source_texts, translation):
    """Return one translated sentence per source segment when alignment is unambiguous."""
    sources = [str(text or "").strip() for text in source_texts]
    if len(sources) < 2 or any(not text for text in sources):
        return None
    if not all(_TERMINAL_RE.search(text) for text in sources):
        return None

    parts = [match.group(0).strip() for match in _SENTENCE_RE.finditer((translation or "").strip())]
    if len(parts) != len(sources) or any(not part for part in parts):
        return None
    return parts


def attach_translation_segments(region):
    """Attach conservative source-to-translation alignment metadata to a TextBlock."""
    render_segments = list(getattr(region, "render_segments", None) or [])
    if render_segments:
        segments = split_translation_for_segments(
            getattr(region, "translation", ""),
            [segment.get("text", "") for segment in render_segments],
        )
        region.translation_segments = segments
        if segments:
            region.translation = "".join(segments)
        else:
            region.translation = _SEGMENT_MARKER_RE.sub("", getattr(region, "translation", "")).strip()
        return segments

    source_texts = list(getattr(region, "texts", None) or [])
    lines = getattr(region, "lines", None)
    if lines is None or len(lines) != len(source_texts):
        region.translation_segments = None
        return None
    segments = align_translation_segments(source_texts, getattr(region, "translation", ""))
    region.translation_segments = segments
    return segments
