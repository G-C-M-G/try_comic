"""Headless Japanese text reading via Windows.Media.Ocr (winrt).

The ja-JP OCR language pack must be installed (DISM /Online /Add-Capability
/CapabilityName:Language.OCR~~~ja-JP~0.0.1.0).  Returns line boxes and text in
full image coordinates without any window or UI automation.
"""

from __future__ import annotations

import asyncio
from typing import Sequence


async def _recognize(path: str):
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.globalization import Language
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage import FileAccessMode, StorageFile

    file = await StorageFile.get_file_from_path_async(path)
    stream = await file.open_async(FileAccessMode.READ)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = OcrEngine.try_create_from_language(Language("ja"))
    if engine is None:
        return None
    return await engine.recognize_async(bitmap)


def windows_ocr_predictions(path: str) -> list[dict] | None:
    """OCR one image file; returns ``[{polygon, text, confidence}]``.

    ``confidence`` is always ``None`` (Windows.Media.Ocr exposes no score) and
    the polygon is the bounding box of each recognized line.  Returns ``None``
    when the ja recognizer is unavailable and ``[]`` when nothing is found.
    """
    result = asyncio.run(_recognize(path))
    if result is None:
        return None
    predictions: list[dict] = []
    for line in result.lines:
        words = list(line.words)
        if not words:
            continue
        xs = [word.bounding_rect.x for word in words]
        ys = [word.bounding_rect.y for word in words]
        x2s = [word.bounding_rect.x + word.bounding_rect.width for word in words]
        y2s = [word.bounding_rect.y + word.bounding_rect.height for word in words]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(x2s)), int(max(y2s))
        predictions.append({
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "text": "".join(word.text for word in words),
            "confidence": None,
        })
    return predictions


def windows_ocr_available() -> bool:
    try:
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.globalization import Language

        return OcrEngine.try_create_from_language(Language("ja")) is not None
    except Exception:
        return False
