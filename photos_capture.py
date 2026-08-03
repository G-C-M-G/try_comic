"""Fully automatic column capture from the Windows Photos OCR overlay.

The Photos app owns OCR.  This module only drives its public UI, reads copied
text, and maps the visual selection back to the source image.  MangaOCR is not
used anywhere in this capture path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class OverlayBlock:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


def _clipboard_text() -> str:
    import win32clipboard

    for _ in range(10):
        try:
            win32clipboard.OpenClipboard()
            try:
                value = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return str(value or "").strip()
        except Exception:
            time.sleep(0.05)
    return ""


def _set_clipboard_marker(marker: str) -> None:
    import win32clipboard

    for _ in range(10):
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(marker, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("cannot initialize the clipboard")


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    text = text.replace("!!", "！！").replace("!?", "！？")
    return text


def _wait_for_window(title: str, timeout: float):
    from pywinauto import Desktop

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values = [w for w in Desktop(backend="uia").windows() if w.window_text() == title]
        if len(values) == 1:
            return values[0]
        time.sleep(0.25)
    raise RuntimeError(f"Photos window {title!r} did not become available")


def _open_photos(path: str, timeout: float):
    title = os.path.basename(path)
    existing = None
    try:
        existing = _wait_for_window(title, 0.2)
    except RuntimeError:
        pass
    if existing is None:
        os.startfile(os.path.abspath(path))
    window = _wait_for_window(title, timeout)
    window.set_focus()
    time.sleep(0.5)
    return window


def _capture(control) -> np.ndarray:
    return np.array(control.capture_as_image().convert("RGB"))


def _get_control(window, automation_id: str, timeout: float = 10):
    from pywinauto import Desktop

    handle = window.handle
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fresh = Desktop(backend="uia").window(handle=handle)
            return fresh.child_window(auto_id=automation_id).wrapper_object()
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Photos did not expose {automation_id}")


def _try_get_control(window, automation_id: str):
    try:
        return _get_control(window, automation_id, 0.6)
    except RuntimeError:
        return None


def _inventory_text(active_control, block: OverlayBlock) -> list[str]:
    """Copy Photos' complete OCR inventory for capture completeness checks."""
    from pywinauto import keyboard, mouse

    rect = active_control.rectangle()
    mouse.click(coords=(rect.left + block.center[0], rect.top + block.center[1]))
    time.sleep(0.1)
    marker = f"__PHOTOS_ALL_{time.time_ns()}__"
    _set_clipboard_marker(marker)
    keyboard.send_keys("^a")
    time.sleep(0.08)
    keyboard.send_keys("^c")
    time.sleep(0.12)
    copied = _clipboard_text()
    if not copied or copied == marker:
        raise RuntimeError("Photos did not copy its complete OCR inventory")
    return [
        _normalize_text(item)
        for item in re.split(r"\r?\n+", copied)
        if _JAPANESE_RE.search(item)
    ]


def detect_overlay_blocks(base: np.ndarray, active: np.ndarray) -> list[OverlayBlock]:
    """Find the bright white-outline OCR islands retained by Photos."""
    if base.shape != active.shape:
        raise ValueError("Photos OCR before/after frames have different sizes")
    delta = np.mean(np.abs(active.astype(np.int16) - base.astype(np.int16)), axis=2)
    brightness = np.mean(active, axis=2)
    mask = ((delta < 13) & (brightness > 145)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    values = []
    for x, y, width, height, area in stats[1:]:
        if area < 40 or width < 5 or height < 9:
            continue
        values.append(OverlayBlock(int(x), int(y), int(width), int(height)))
    return sorted(values, key=lambda item: (item.y, -item.x))


def _projection_clicks(active: np.ndarray, block: OverlayBlock) -> list[tuple[int, int]]:
    """Split a white OCR island into vertical columns using its black glyphs."""
    gray = cv2.cvtColor(active, cv2.COLOR_RGB2GRAY)
    x, y, width, height = block.x, block.y, block.width, block.height
    crop = gray[y:y + height, x:x + width]
    ink = crop < 90
    projection = ink.sum(axis=0)
    threshold = max(2, int(height * 0.018))
    active_x = projection >= threshold
    groups = []
    start = None
    for index, present in enumerate(active_x.tolist() + [False]):
        if present and start is None:
            start = index
        elif not present and start is not None:
            if index - start >= 3:
                groups.append((start, index - 1))
            start = None
    if not groups:
        return [block.center]

    merged = []
    for left, right in groups:
        if merged and left - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    clicks = []
    for left, right in sorted(merged, reverse=True):
        band = ink[:, left:right + 1]
        rows = np.where(band.any(axis=1))[0]
        cy = int(np.median(rows)) if len(rows) else height // 2
        clicks.append((x + (left + right) // 2, y + cy))
        # Photos often splits a visual island into several selectable ranges.
        # Probe both sides of a glyph band as well as its centre; clipboard and
        # geometry de-duplication remove the repeated hits.
        clicks.append((x + max(left, (2 * left + right) // 3), y + cy))
        clicks.append((x + min(right, (left + 2 * right) // 3), y + cy))
    clicks.append(block.center)
    return clicks


def _selection_bbox(active: np.ndarray, selected: np.ndarray, click: tuple[int, int]) -> list[int]:
    """Extract Photos' blue selected-range geometry around the clicked column."""
    delta = np.max(np.abs(selected.astype(np.int16) - active.astype(np.int16)), axis=2)
    mask = (delta > 20).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    cx, cy = click
    candidates = []
    for x, y, width, height, area in stats[1:]:
        if area < 20:
            continue
        contains = x - 8 <= cx <= x + width + 8 and y - 8 <= cy <= y + height + 8
        distance = (cx - (x + width / 2)) ** 2 + (cy - (y + height / 2)) ** 2
        candidates.append((not contains, distance, int(x), int(y), int(width), int(height)))
    if not candidates:
        return [max(0, cx - 10), max(0, cy - 20), cx + 10, cy + 20]
    _, _, x, y, width, height = min(candidates)
    return [x, y, x + width, y + height]


def _display_bbox_to_image(bbox: Sequence[int], viewport_size, image_size) -> list[int]:
    view_w, view_h = viewport_size
    image_w, image_h = image_size
    scale = min(view_w / image_w, view_h / image_h)
    shown_w, shown_h = image_w * scale, image_h * scale
    offset_x, offset_y = (view_w - shown_w) / 2, (view_h - shown_h) / 2
    x1, y1, x2, y2 = bbox
    mapped = [
        round((x1 - offset_x) / scale),
        round((y1 - offset_y) / scale),
        round((x2 - offset_x) / scale),
        round((y2 - offset_y) / scale),
    ]
    mapped[0] = max(0, min(image_w - 1, mapped[0]))
    mapped[2] = max(mapped[0] + 1, min(image_w, mapped[2]))
    mapped[1] = max(0, min(image_h - 1, mapped[1]))
    mapped[3] = max(mapped[1] + 1, min(image_h, mapped[3]))
    return mapped


def capture_photos_columns(path: str, timeout: float = 20, debug_dir: str | None = None) -> dict:
    from pywinauto import keyboard, mouse

    source = os.path.abspath(path)
    image = Image.open(source).convert("RGB")
    window = _open_photos(source, timeout)
    # OCR mode replaces D2DImageControl with OcrControl.  Reset it first so a
    # previous run or manual Photos session cannot break automatic capture.
    if _try_get_control(window, "OcrControl") is not None:
        _get_control(window, "OcrButton").invoke()
        time.sleep(0.5)
    base_control = _get_control(window, "D2DImageControl")
    base = _capture(base_control)
    button = _get_control(window, "OcrButton")
    button.invoke()

    active_control = _get_control(window, "OcrControl", timeout)
    deadline = time.monotonic() + timeout
    active = None
    blocks = []
    while time.monotonic() < deadline:
        window.set_focus()
        active_control = _get_control(window, "OcrControl", timeout)
        active = _capture(active_control)
        blocks = detect_overlay_blocks(base, active)
        if blocks:
            break
        time.sleep(0.25)
    if not blocks or active is None:
        raise RuntimeError("Photos OCR overlay did not produce detectable text")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(base).save(os.path.join(debug_dir, "photos_base.png"))
        Image.fromarray(active).save(os.path.join(debug_dir, "photos_ocr.png"))

    rect = active_control.rectangle()
    seen: list[dict] = []
    click_index = 0
    for block_index, block in enumerate(blocks):
        for click in _projection_clicks(active, block):
            click_index += 1
            marker = f"__PHOTOS_OCR_{click_index}_{time.time_ns()}__"
            _set_clipboard_marker(marker)
            mouse.click(coords=(rect.left + click[0], rect.top + click[1]))
            time.sleep(0.12)
            keyboard.send_keys("^c")
            time.sleep(0.08)
            copied = _clipboard_text()
            if not copied or copied == marker or not _JAPANESE_RE.search(copied):
                continue
            text = _normalize_text(copied)
            mouse.move(coords=(rect.right + 18, rect.bottom + 18))
            time.sleep(0.05)
            selected = _capture(active_control)
            display_bbox = _selection_bbox(active, selected, click)
            image_bbox = _display_bbox_to_image(
                display_bbox, (active.shape[1], active.shape[0]), image.size
            )
            duplicate = False
            for prior in seen:
                if prior["text"] != text:
                    continue
                ax1, ay1, ax2, ay2 = prior["image_bbox"]
                bx1, by1, bx2, by2 = image_bbox
                intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                if smaller and intersection / smaller >= 0.55:
                    duplicate = True
                    # Keep the tighter selection when the mouse previously
                    # inflated one of the visual difference components.
                    if (bx2 - bx1) * (by2 - by1) < (ax2 - ax1) * (ay2 - ay1):
                        prior["display_bbox"] = display_bbox
                        prior["image_bbox"] = image_bbox
                    break
            if duplicate:
                continue
            seen.append({
                "id": "",
                "text": text,
                "display_bbox": display_bbox,
                "image_bbox": image_bbox,
                "selection_order": len(seen),
                "direction": "vertical" if (image_bbox[3] - image_bbox[1]) >= (image_bbox[2] - image_bbox[0]) else "horizontal",
                "role": "text",
                "overlay_block": block_index,
            })

    columns = list(seen)
    columns.sort(key=lambda item: (item["image_bbox"][1], -item["image_bbox"][2]))
    for index, column in enumerate(columns, 1):
        column["id"] = f"column_{index:03d}"
        column["selection_order"] = index - 1
    inventory = _inventory_text(active_control, blocks[0])
    captured_counts = Counter(column["text"] for column in columns)
    inventory_counts = Counter(inventory)
    missing_counts = inventory_counts - captured_counts
    extra_counts = captured_counts - inventory_counts
    if missing_counts or extra_counts:
        missing = ", ".join(
            text if count == 1 else f"{text} x{count}"
            for text, count in missing_counts.items()
        ) or "none"
        extra = ", ".join(
            text if count == 1 else f"{text} x{count}"
            for text, count in extra_counts.items()
        ) or "none"
        raise RuntimeError(
            f"Photos OCR capture does not match its inventory; missing: {missing}; extra: {extra}"
        )
    return {
        "schema_version": 2,
        "backend": "Windows Photos Scan Text",
        "source": {"path": source, "width": image.width, "height": image.height},
        "capture": {
            "app": "Windows Photos",
            "automation_control": "OcrControl",
            "viewport_size": [active.shape[1], active.shape[0]],
            "inventory_count": len(inventory),
            "captured_count": len(columns),
            "complete": True,
        },
        "columns": columns,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture column OCR from Windows Photos")
    parser.add_argument("image")
    parser.add_argument("-o", "--output")
    parser.add_argument("--debug-dir")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    result = capture_photos_columns(args.image, args.timeout, args.debug_dir)
    output = args.output or str(Path(args.image).with_suffix(".photos-ocr.json"))
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
