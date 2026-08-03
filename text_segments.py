"""Align translated sentences with the source OCR sub-regions."""
import re


_SENTENCE_RE = re.compile(r".+?(?:[。！？!?]+|$)")
_TERMINAL_RE = re.compile(r"[。！？!?]+\s*$")


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
    source_texts = list(getattr(region, "texts", None) or [])
    lines = getattr(region, "lines", None)
    if lines is None or len(lines) != len(source_texts):
        region.translation_segments = None
        return None
    segments = align_translation_segments(source_texts, getattr(region, "translation", ""))
    region.translation_segments = segments
    return segments
