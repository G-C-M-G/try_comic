"""Fully automatic column capture from the Windows Photos OCR overlay.

The Photos app owns OCR.  This module only drives its public UI, reads copied
text, and maps the visual selection back to the source image.  MangaOCR is not
used anywhere in this capture path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

# Force line-buffered stdout/stderr: when ``Popen(..., stdout=PIPE)`` the
# subprocess defaults to block buffering, so any ``print()`` stays in the
# buffer until the process exits or flushes.  A signal-terminated run (GUI
# cancel / Task Manager) would lose every layout line and look like "the
# capture never started" -- this is what produced the 23:16 / 23:19 gaps in
# the run.log.  Line buffering flushes each line immediately so the parent
# forwards progress in real time and a killed child leaves a meaningful
# trace in the log instead of an empty buffer.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from windows_ocr import windows_ocr_available, windows_ocr_predictions


ROOT = os.path.dirname(os.path.abspath(__file__))
PADDLE_PYTHON = os.path.join(ROOT, "paddle_venv", "Scripts", "python.exe")
PADDLE_WORKER = os.path.join(ROOT, "paddle_ocr_worker.py")
PADDLE_ENV_OVERRIDES = {
    "HOME": os.path.join(ROOT, "models", "paddle"),
    "USERPROFILE": os.path.join(ROOT, "models", "paddle"),
    "PADDLE_PDX_CACHE_HOME": os.path.join(ROOT, "models", "paddle"),
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1",
}

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Right boundary for the display-scale window layout: the rightmost window
# never crosses this x (e.g. 2400 keeps windows clear of an obstruction on
# a 2560px-wide screen).  0 = screen edge.
DISPLAY_RIGHT_LIMIT = 2400

# Run in physical pixels: a DPI-unaware process gets display-scaled system
# metrics (e.g. 1707x1067 for a 2560x1600 screen at 150%), which shrinks
# the layout budget and makes 6 pinned windows impossible.  DPI awareness
# makes GetSystemMetrics / SetWindowPos / ImageGrab all agree in real
# pixels and keeps the captured frames sharp.
try:
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# Serializes focus + keyboard sequences across parallel Photos windows:
# SendInput delivers to the focused window, so only one worker may hold
# focus while typing at a time.  Frame grabs (ImageGrab) stay parallel.
_INPUT_LOCK = threading.Lock()


class PhotosNoTextError(RuntimeError):
    """Photos scan-text found no text (e.g. a pure-illustration page).

    The page itself is normal -- it simply has nothing to translate.  The
    capture flow treats it like a black page: no retries, no fallback
    dialog, ``no_text`` is recorded in the manifest and downstream skips it.
    """


@dataclass(frozen=True)
class OverlayBlock:
    x: int
    y: int
    width: int
    height: int
    page_scale: bool = False
    split_region: bool = False

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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def _parse_pages(value: str | None) -> set[str] | None:
    if not value:
        return None
    # Tolerate full-width punctuation (users type "4，6" / "4、6" in an
    # IME): normalize to the half-width separator before splitting, so a
    # full-width comma cannot become part of a page name and silently
    # match nothing.
    value = value.replace("，", ",").replace("、", ",").replace(" ", "")
    pages: set[str] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            pages.update(str(number) for number in range(int(left), int(right) + 1))
        else:
            pages.add(str(int(item)) if item.isdigit() else item)
    return pages


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _wait_for_window(title: str, timeout: float):
    from pywinauto import Desktop

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values = [w for w in Desktop(backend="uia").windows() if w.window_text() == title]
        if values:
            return values[0]
        time.sleep(0.25)
    raise RuntimeError(f"Photos window {title!r} did not become available")


def _launch_photos(path: str) -> None:
    absolute_path = os.path.abspath(path)
    try:
        os.startfile(absolute_path)
        return
    except OSError as error:
        # WinError 1155 means the image extension has no default application.
        if getattr(error, "winerror", None) != 1155:
            raise

    subprocess.Popen(["explorer.exe", absolute_path], close_fds=True)


def _ensure_review_image(page_dir: Path, image_path) -> None:
    """Guarantee a reviewable image for failed pages.

    The base/ocr frames are only saved after ``_prepare_overlay`` succeeds,
    so a failed page (no-text / exception) would leave the review gate with
    NOTHING to look at.  Copy the ORIGINAL page as ``photos_base.png`` when
    the frame files are missing.
    """
    base_path = Path(page_dir) / "photos_base.png"
    if base_path.is_file():
        return
    source = Path(image_path)
    if source.is_file():
        try:
            import shutil

            shutil.copyfile(source, base_path)
        except OSError:
            pass


def _close_window_wait(window, timeout: float = 1.5) -> None:
    """Close a Photos window and WAIT until it is really gone.

    A closed-but-not-yet-destroyed window can be picked up by the next
    page's ``_open_photos`` title scan, which then waits on a dying window
    and fails -- making parallel capture appear to stop opening windows.
    Falls back to WM_CLOSE (twice) so a Photos dialog cannot pin the
    window open.
    """
    if window is None:
        return
    try:
        window.close()
    except Exception:
        pass
    try:
        import ctypes

        handle = int(window.handle)
        user32 = ctypes.windll.user32
        deadline = time.monotonic() + timeout
        sent = 0
        while time.monotonic() < deadline:
            try:
                if not user32.IsWindow(handle):
                    return
            except Exception:
                return
            if sent == 0 and time.monotonic() > deadline - 1.2:
                user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE
                sent = 1
            elif sent == 1 and time.monotonic() > deadline - 0.5:
                user32.PostMessageW(handle, 0x0010, 0, 0)
                sent = 2
            time.sleep(0.2)
    except Exception:
        pass


def _open_photos(path: str, timeout: float):
    title = os.path.basename(path)
    existing = None
    try:
        existing = _wait_for_window(title, 0.2)
    except RuntimeError:
        pass
    if existing is not None:
        # A window with the same title is a leftover from THIS page's
        # previous attempt (each page gets a fresh window).  Reusing it
        # risks showing a stale/mismatched image (parallel drift) and a
        # half-closed window.  Close it for real, then open fresh.
        try:
            _close_window_wait(existing)
        except Exception:
            pass
        existing = None
    if existing is None:
        _launch_photos(path)
    window = _wait_for_window(title, timeout)
    window.set_focus()
    time.sleep(0.5)
    return window


def _close_residual_photos_windows(target_names: set[str]) -> int:
    """Close every Photos window whose title matches one of ``target_names``.

    The GUI/CLI can be killed at any time (closeEvent cancel, Task Manager,
    power loss) and the Photos windows it opened are regular Explorer
    children -- they survive the parent and stack up on the desktop.  When
    a new batch starts we sweep the desktop for any window whose title
    equals a page name we are about to capture and close it, so the new
    workers do not race against orphan windows that may still be
    displaying a different image.  Other Photos windows (the user
    manually viewing unrelated photos) are left alone.

    Returns the number of windows closed.
    """
    if not target_names:
        return 0
    try:
        from pywinauto import Desktop
    except Exception:
        return 0
    closed = 0
    try:
        windows = Desktop(backend="uia").windows()
    except Exception:
        return 0
    for window in windows:
        try:
            title = (window.window_text() or "").strip()
        except Exception:
            continue
        if not title or title not in target_names:
            continue
        try:
            _close_window_wait(window)
            closed += 1
        except Exception:
            pass
    return closed


def _capture(control) -> np.ndarray:
    return np.array(control.capture_as_image().convert("RGB"))


def _capture_control_region(window, control) -> np.ndarray:
    """Grab the rendered screen area of one Photos control.

    Photos 2026 renders the scan-text gray mask and highlighted text islands
    as a composition layer that ``capture_as_image`` does not include.  A
    screen grab sees exactly what the user sees.  The window and control
    rectangles are re-fetched on every call because Photos may move or resize
    its window when scan-text mode is enabled.

    NOTE: this is the original f984790 behaviour -- the window is grabbed
    exactly where it currently is, with no overlap guard.  A parallel
    window that could not be pinned exactly still captures its frames;
    frame pollution from a tiny edge overlap is preferable to losing the
    page entirely.
    """
    from PIL import ImageGrab

    window_rect = _window_rect_tuple(window)
    control_rect = control.rectangle()
    full = ImageGrab.grab(bbox=window_rect)
    x = control_rect.left - window_rect[0]
    y = control_rect.top - window_rect[1]
    width = control_rect.right - control_rect.left
    height = control_rect.bottom - control_rect.top
    crop = full.crop((x, y, x + width, y + height))
    return np.array(crop.convert("RGB"))


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


def _align_frames(first: np.ndarray, second: np.ndarray):
    """Tolerate tiny layout drift between two screen grabs.

    Photos resizes its window by 1-2px between the base grab and the
    scan-text switch (window-edge jitter).  Frames within 2px are aligned
    to their common top-left region; larger mismatches raise.
    """
    if first.shape == second.shape:
        return first, second
    if (
        abs(first.shape[0] - second.shape[0]) <= 2
        and abs(first.shape[1] - second.shape[1]) <= 2
    ):
        height = min(first.shape[0], second.shape[0])
        width = min(first.shape[1], second.shape[1])
        return first[:height, :width], second[:height, :width]
    raise ValueError("Photos OCR before/after frames have different sizes")


def detect_overlay_blocks(base: np.ndarray, active: np.ndarray) -> list[OverlayBlock]:
    """Find the highlight islands Photos renders in scan-text mode.

    Photos dims the page with a gray overlay and keeps the recognized text
    regions at original brightness, so islands are the *undimmed* bright
    areas.  Some Photos builds draw white outline strokes instead; those are
    detected as closed contours of brightened pixels.  Returns ``[]`` while
    the overlay is not yet present so the caller keeps polling instead of
    mistaking plain page regions for OCR islands.
    """
    base, active = _align_frames(base, active)
    base_f = base.astype(np.int16)
    active_f = active.astype(np.int16)
    delta = np.mean(np.abs(active_f - base_f), axis=2)
    signed = np.mean(active_f - base_f, axis=2)
    active_bright = np.mean(active_f, axis=2)
    base_bright = np.mean(base, axis=2)
    dimmed = (delta > 10) & (signed < -4)
    brightened = (delta > 10) & (signed > 4) & (active_bright > 200)
    dim_fraction = float(np.mean(dimmed))
    bright_fraction = float(np.mean(brightened))
    # A real scan-text gray mask dims most of the page (~80% measured); the
    # Photos UI panel only changes a few percent.  Below the gate the overlay
    # is still loading and the caller keeps polling instead of mistaking the
    # panel or plain page regions for OCR islands.
    if dim_fraction >= 0.15:
        mask = ((~dimmed) & (base_bright > 145)).astype(np.uint8) * 255
    elif bright_fraction >= 0.01:
        mask = cv2.morphologyEx(
            (brightened.astype(np.uint8) * 255),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
        )
    else:
        return []
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


def _text_aware_split(
    frame: np.ndarray,
    x: int, y: int, width: int, height: int,
) -> list[tuple[int, int, int, int]]:
    """Split a large selection box into text blocks by x projection gaps.

    Photos sometimes draws ONE selection box around several text blocks
    (e.g. a footer bar with a few columns).  The box is split where the
    local-contrast text mask shows a real x gap (>= 6px, a column pitch),
    so every returned rectangle hugs one block of text.
    """
    if frame is None or width <= 0 or height <= 0:
        return []
    gray = frame[y:y + height, x:x + width]
    if gray.size == 0:
        return []
    gray = gray.astype(np.float32)
    median = cv2.medianBlur(gray.astype(np.uint8), 15).astype(np.float32)
    text = np.abs(gray - median) > 45
    proj = text.sum(axis=0)
    segments = []
    start = None
    for index in range(proj.shape[0] + 1):
        present = index < proj.shape[0] and proj[index] > 2
        if present and start is None:
            start = index
        elif not present and start is not None:
            segments.append((start, index - 1))
            start = None
    if len(segments) < 2:
        return []
    merged = []
    for left, right in segments:
        if merged and left - merged[-1][1] - 1 < 6:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    if len(merged) < 2:
        return []
    boxes = []
    for left, right in merged:
        band = text[:, left:right + 1]
        rows = np.where(band.any(axis=1))[0]
        if rows.size == 0:
            continue
        top, bottom = int(rows.min()), int(rows.max()) + 1
        if bottom - top < 8 or right - left + 1 < 6:
            continue
        boxes.append((x + left, y + top, right - left + 1, bottom - top))
    return boxes


# Photos scan-text selection overlay: detected by comparing the two
# Photos' blue selection overlay (measured): a normal blend of
# RGB(0, 120, 211) at alpha 81/255 over the ORIGINAL pixel.  A covered
# pixel c becomes c' = c * 174/255 + (0, 38, 67), so the exact overlay is
# reconstructed from the base frame and the boxes are found by comparing
# the all-selected frame against that reconstruction.
BLUE_OVERLAY_KEEP = 174.0 / 255.0
BLUE_OVERLAY_OFFSET_G = 38.0
BLUE_OVERLAY_OFFSET_B = 67.0
# Match slack for the overlay formula.  Measured residuals at real box
# pixels are <= 3 per channel, so 5 is safe and tighter than the old 12;
# tightening below ~2 fragments boxes and drops the stored-box hit rate.
BLUE_OVERLAY_TOLERANCE = 5.0
# A pixel must have been repainted by more than this to count as part of
# the overlay (untouched pure-blue artwork satisfies the formula ~by
# identity and must not become a box).  Tolerance 2 is permissive; the
# formula gate above is the real discriminator.
BLUE_OVERLAY_PAINT_DELTA = 2.0

# Whole-page selection threshold: a connected blue region covering more
# than this fraction of the frame is a whole-page misselection and is
# dropped (not an atomic text box).
WHOLE_PAGE_AREA_RATIO = 0.20


def blue_overlay_image(base: np.ndarray) -> np.ndarray:
    """Expected all-selected frame under Photos' blue selection overlay.

    A pixel painted by the overlay is c' = c * 174/255 + (0, 38, 67) with
    the offsets applied to the original G/B channels.
    """
    f = base.astype(np.float64)
    out = f * BLUE_OVERLAY_KEEP
    out[..., 1] += BLUE_OVERLAY_OFFSET_G
    out[..., 2] += BLUE_OVERLAY_OFFSET_B
    return out


def detect_selection_boxes(
    active: np.ndarray,
    selected: np.ndarray,
    *,
    base: np.ndarray | None = None,
    tolerance: float = BLUE_OVERLAY_TOLERANCE,
    whole_page_ratio: float = WHOLE_PAGE_AREA_RATIO,
) -> list[OverlayBlock]:
    """Find the blue selection rectangles Photos draws over recognized text.

    After Ctrl+A every scan-text range gets a blue overlay box (Photos draws
    them as a composition layer, so only a screen grab sees them).  Each blue
    box is one atomic OCR unit -- it must never be split.

    Single detection path: the selection overlay is Photos' blue RGB(0, 120, 211)
    blended at 31.76% over the ORIGINAL pixel, i.e.
    ``selected == blue_overlay_image(base)``.  Pixels that match the
    formula within ``tolerance`` but were NOT repainted (blue artwork whose
    overlay is ~the identity) are excluded by the paint-delta guard, so
    only genuine selection boxes survive.  ``whole_page_ratio`` drops
    connected regions covering more than that fraction of the frame.

    When ``base`` is missing the page is treated as a capture failure: the
    formula needs the ORIGINAL-pixel baseline to reconstruct the expected
    overlay, and guessing from the two frame differences alone would only
    fabricate boxes on rendering noise.  An empty list is returned and the
    review gate picks the page up.
    """
    if active.shape != selected.shape:
        active, selected = _align_frames(active, selected)
    if base is None:
        return []
    if base.shape != selected.shape:
        base, selected = _align_frames(base, selected)
    gray_once = np.mean(active, axis=2).astype(np.float32)
    expected = blue_overlay_image(base)
    formula = np.abs(selected.astype(np.float64) - expected).max(axis=2) <= tolerance
    painted = np.abs(selected.astype(np.float64) - base.astype(np.float64)).max(axis=2) > BLUE_OVERLAY_PAINT_DELTA
    mask = cv2.morphologyEx(
        (formula & painted).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    values = _selection_boxes_from_mask(
        mask, gray_once, active.shape, whole_page_ratio=whole_page_ratio
    )
    return sorted(values, key=lambda item: (item.y, -item.x))


def _selection_boxes_from_mask(
    mask: np.ndarray, gray: np.ndarray, shape,
    whole_page_ratio: float = WHOLE_PAGE_AREA_RATIO,
) -> list[OverlayBlock]:
    """Connected-component box extraction from a blue-selection mask.

    Despeckled components become atomic boxes; components covering more
    than ``whole_page_ratio`` of the frame (Photos treating a whole page as
    one OCR block) are dropped -- a whole-page selection is not an atomic
    text box.  Big components are split by x projection and text-aware gaps.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    values = []
    for x, y, width, height, area in stats[1:]:
        if area < 60 or width < 6 or height < 8:
            continue
        # Whole-page selections are not atomic text boxes: drop them.
        if (width * height) > shape[0] * shape[1] * whole_page_ratio:
            continue
        # Split a component at empty x columns: Photos may fuse a few boxes
        # that touch, but columns leave measurable gaps between the boxes.
        sub = mask[y:y + height, x:x + width]
        proj = sub.sum(axis=0)
        segments = []
        start = None
        for index in range(proj.shape[0] + 1):
            present = index < proj.shape[0] and proj[index] > 0
            if present and start is None:
                start = index
            elif not present and start is not None:
                segments.append((start, index - 1))
                start = None
        for left, right in segments:
            box_width = right - left + 1
            if box_width < 6:
                continue
            band = sub[:, left:right + 1]
            rows = np.where(band.any(axis=1))[0]
            if rows.size == 0:
                continue
            box_y1 = y + int(rows.min())
            box_y2 = y + int(rows.max()) + 1
            if box_y2 - box_y1 < 8:
                continue
            # A large non-page-scale box often wraps several text blocks
            # (e.g. a footer bar): split it by text x gaps so each returned
            # box hugs one block of text.
            if (
                (box_width > shape[1] * 0.35
                 or (box_y2 - box_y1) > shape[0] * 0.35)
            ):
                pieces = _text_aware_split(
                    gray, x + left, box_y1, box_width, box_y2 - box_y1
                )
                if len(pieces) >= 2:
                    for px, py, pw, ph in pieces:
                        if pw < 6 or ph < 8:
                            continue
                        values.append(
                            OverlayBlock(
                                int(px), int(py), int(pw), int(ph),
                                page_scale=False,
                            )
                        )
                    continue
            values.append(
                OverlayBlock(
                    int(x + left), int(box_y1), int(box_width), int(box_y2 - box_y1),
                    page_scale=False,
                )
            )
    return values


def aggregate_selection_boxes(
    blocks: Sequence[OverlayBlock], x_gap: int = 24, y_pad: int = 8
) -> list[tuple[tuple[int, int, int, int], list[OverlayBlock]]]:
    """Group atomic selection boxes into sentence-level blocks.

    Two boxes belong to one group when their x gap is within *x_gap* and
    their y ranges overlap (within *y_pad*): horizontally adjacent columns
    of the same line merge, while vertically split halves of different text
    (e.g. ``同様に`` above ``純然たる``) stay separate.  The group bbox is
    the union of its member boxes -- downstream OCR then does not need to
    decide which boxes form a line.
    """
    ordered = sorted(blocks, key=lambda item: (item.y, item.x))
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, item in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            other = ordered[j]
            y_overlap = min(
                item.y + item.height, other.y + other.height
            ) - max(item.y, other.y)
            if y_overlap < -y_pad:
                continue
            gap = max(item.x, other.x) - min(
                item.x + item.width, other.x + other.width
            )
            if gap <= x_gap:
                union(i, j)

    groups: dict[int, list[OverlayBlock]] = {}
    for index, block in enumerate(ordered):
        groups.setdefault(find(index), []).append(block)
    result = []
    for members in groups.values():
        x1 = min(item.x for item in members)
        y1 = min(item.y for item in members)
        x2 = max(item.x + item.width for item in members)
        y2 = max(item.y + item.height for item in members)
        result.append(((x1, y1, x2, y2), members))
    return sorted(result, key=lambda item: (item[0][1], -item[0][2]))


def _capture_selection_boxes(
    window,
    active_control,
    active: np.ndarray,
    debug_dir: str | None = None,
    base: np.ndarray | None = None,
    fixed_window_rect: tuple[int, int, int, int] | None = None,
    box_delta: int = 8,
    box_tolerance: float = 5.0,
    box_area_ratio: float = 0.20,
) -> tuple[list[OverlayBlock], np.ndarray | None, str | None]:
    """Ctrl+A all scan-text ranges and grab the blue selection overlay.

    The focus + Ctrl+A sequence is serialized, then the blue frame is
    grabbed and verified: when no blue boxes appear (the overlay is still
    rendering or focus was stolen), Ctrl+A is re-sent with an increasing
    wait (0.25s / 0.65s / 1.05s) up to three attempts.  ``fixed_window_rect``
    (the pinned tile) is re-applied right before every grab: parallel
    windows may drift/resize while unfocused, and a drifted grab extends
    into the neighbour's tile.  Fails soft only after all attempts: the
    frame is still saved and the caller marks the page
    ``selection_missing`` so the batch is never blocked.

    ``box_delta`` is accepted for CLI compatibility but currently has no
    effect: detection is a single strict-formula path and there is no
    per-channel-delta fallback left.
    """
    from pywinauto import keyboard

    last_selected = None
    error = None
    for attempt in range(3):
        wait = 0.25 + 0.4 * attempt
        try:
            with _INPUT_LOCK:
                window.set_focus()
                time.sleep(0.2)
                if fixed_window_rect is not None:
                    _ensure_window_rect(window, fixed_window_rect)
                keyboard.send_keys("^a")
                time.sleep(wait)
                if fixed_window_rect is not None:
                    _ensure_window_rect(window, fixed_window_rect)
                selected = _capture_control_region(window, active_control)
                blocks = detect_selection_boxes(
                    active, selected, base=base,
                    tolerance=box_tolerance,
                    whole_page_ratio=box_area_ratio,
                )
            if blocks:
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    Image.fromarray(selected).save(
                        os.path.join(debug_dir, "photos_selected.png")
                    )
                return blocks, selected, None
            last_selected = selected
            error = "no blue boxes after Ctrl+A (retrying)"
        except Exception as exc:  # noqa: BLE001 - soft failure by design
            error = str(exc)
            last_selected = None
        time.sleep(0.3)
    if debug_dir and last_selected is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(last_selected).save(
                os.path.join(debug_dir, "photos_selected.png")
            )
        except Exception:
            pass
    return [], last_selected, error or "selection boxes unavailable"


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


def _glyph_clicks(base: np.ndarray) -> list[tuple[int, int]]:
    """Find likely character centres for OCR islands merged into white artwork."""
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    ink = (gray < 100).astype(np.uint8) * 255
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    clicks = []
    for x, y, width, height, area in stats[1:]:
        if not (5 <= width <= 35 and 7 <= height <= 38 and 18 <= area <= 800):
            continue
        if area / (width * height) < 0.16:
            continue
        padding = 7
        neighborhood = gray[
            max(0, y - padding):min(gray.shape[0], y + height + padding),
            max(0, x - padding):min(gray.shape[1], x + width + padding),
        ]
        if neighborhood.size and float(np.mean(neighborhood > 210)) >= 0.52:
            clicks.append((int(x + width // 2), int(y + height // 2)))
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


def calibrate_viewport_transform(
    source_image: np.ndarray,
    display_image: np.ndarray,
) -> dict:
    """Estimate and validate the source-to-Photos display transform."""
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
    display_gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    source_points, source_descriptors = detector.detectAndCompute(source_gray, None)
    display_points, display_descriptors = detector.detectAndCompute(display_gray, None)
    if source_descriptors is None or display_descriptors is None:
        raise RuntimeError("cannot calibrate Photos viewport: not enough image features")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        source_descriptors, display_descriptors
    )
    matches = sorted(matches, key=lambda item: item.distance)[:800]
    if len(matches) < 30:
        raise RuntimeError(
            f"cannot calibrate Photos viewport: only {len(matches)} feature matches"
        )
    source_xy = np.float32([source_points[item.queryIdx].pt for item in matches])
    display_xy = np.float32([display_points[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_xy,
        display_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.995,
    )
    if matrix is None or inliers is None:
        raise RuntimeError("cannot calibrate Photos viewport: affine estimation failed")
    accepted = inliers.reshape(-1).astype(bool)
    projected = cv2.transform(source_xy.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - display_xy, axis=1)
    inlier_count = int(accepted.sum())
    inlier_ratio = float(inlier_count / len(matches))
    median_error = float(np.median(errors[accepted])) if inlier_count else float("inf")
    scale_x = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    scale_y = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    valid = (
        inlier_count >= 30
        and inlier_ratio >= 0.35
        and median_error <= 2.5
        and scale_x > 0
        and abs(scale_x - scale_y) / scale_x <= 0.02
    )
    result = {
        "method": "orb-ransac-affine",
        "source_to_display": [[round(float(value), 8) for value in row] for row in matrix],
        "feature_matches": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "median_reprojection_error_px": round(median_error, 4),
        "valid": valid,
    }
    if not valid:
        raise RuntimeError(f"Photos viewport calibration did not pass: {result}")
    return result


def map_display_bbox_affine(bbox: Sequence[int], transform: dict, image_size) -> list[int]:
    matrix = np.asarray(transform["source_to_display"], dtype=np.float64)
    inverse = cv2.invertAffineTransform(matrix)
    x1, y1, x2, y2 = (float(value) for value in bbox)
    display_points = np.asarray(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float64
    )
    source_points = cv2.transform(display_points, inverse).reshape(-1, 2)
    image_w, image_h = image_size
    mapped = [
        max(0, min(image_w - 1, round(float(source_points[:, 0].min())))),
        max(0, min(image_h - 1, round(float(source_points[:, 1].min())))),
        max(1, min(image_w, round(float(source_points[:, 0].max())))),
        max(1, min(image_h, round(float(source_points[:, 1].max())))),
    ]
    mapped[2] = max(mapped[0] + 1, mapped[2])
    mapped[3] = max(mapped[1] + 1, mapped[3])
    return mapped


def _image_bbox_to_display(bbox: Sequence[int], viewport_size, image_size) -> list[int]:
    view_w, view_h = viewport_size
    image_w, image_h = image_size
    scale = min(view_w / image_w, view_h / image_h)
    offset_x, offset_y = (view_w - image_w * scale) / 2, (view_h - image_h * scale) / 2
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale + offset_x),
        round(y1 * scale + offset_y),
        round(x2 * scale + offset_x),
        round(y2 * scale + offset_y),
    ]


def _split_column_bands(active: np.ndarray, block: OverlayBlock) -> list[list[int]]:
    """Split one OCR island into display-space column bands using glyph ink."""
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
        return [[x, y, x + width, y + height]]
    merged = []
    for left, right in groups:
        if merged and left - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    bands = []
    for left, right in merged:
        band_ink = ink[:, left:right + 1]
        rows = np.where(band_ink.any(axis=1))[0]
        top = int(rows.min()) if len(rows) else 0
        bottom = int(rows.max()) + 1 if len(rows) else height
        pad_x = max(2, (right - left + 1) // 3)
        pad_y = max(2, (bottom - top) // 10)
        bands.append([
            max(0, x + left - pad_x),
            max(0, y + top - pad_y),
            min(active.shape[1], x + right + 1 + pad_x),
            min(active.shape[0], y + bottom + pad_y),
        ])
    return bands


def _build_island_mask(image: np.ndarray, islands: Sequence[Sequence[int]]) -> np.ndarray:
    """Black out everything outside the Photos OCR islands."""
    masked = np.zeros_like(image)
    for x1, y1, x2, y2 in islands:
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return masked


def _run_paddle_worker(arguments: Sequence[str], output_path: str, timeout: float = 300) -> dict:
    if not os.path.isfile(PADDLE_PYTHON):
        raise RuntimeError(f"PaddleOCR environment is unavailable: {PADDLE_PYTHON}")
    environment = os.environ.copy()
    environment.update(PADDLE_ENV_OVERRIDES)
    completed = subprocess.run(
        [PADDLE_PYTHON, PADDLE_WORKER, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"PaddleOCR worker failed ({completed.returncode}): {detail}")
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_paddle_whole_page(masked_path: str, device: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="photos_fast_") as temporary:
        output = os.path.join(temporary, "paddle.json")
        payload = _run_paddle_worker([masked_path, output, "--device", device], output)
    pages = payload.get("pages") or []
    return pages[0].get("predictions", []) if pages else []


def _run_masked_ocr(masked_path: str, engine: str, paddle_device: str = "cpu") -> tuple[list[dict], str]:
    """Run masked-page OCR with the selected engine.

    ``auto`` = current behaviour: Windows OCR first, PaddleOCR fallback.
    A forced engine that is unavailable or reads nothing raises
    RuntimeError (no silent fallback).
    Returns ``(predictions, engine_used)``.
    """
    if engine == "auto":
        predictions = windows_ocr_predictions(masked_path)
        if predictions:
            return predictions, "windows-ocr"
        return _run_paddle_whole_page(masked_path, paddle_device), "paddle"
    if engine == "windows-ocr":
        if not windows_ocr_available():
            raise RuntimeError("OCR 引擎 windows-ocr 不可用（Windows 日语 OCR 语言包未安装）")
        predictions = windows_ocr_predictions(masked_path)
        if not predictions:
            raise RuntimeError("OCR 引擎 windows-ocr 未识别到任何文字")
        return predictions, "windows-ocr"
    if engine == "paddle":
        predictions = _run_paddle_whole_page(masked_path, paddle_device)
        if not predictions:
            raise RuntimeError("OCR 引擎 paddle 未识别到任何文字")
        return predictions, "paddle"
    raise ValueError(f"unsupported OCR engine: {engine!r}")


def _run_paddle_crops(manifest: dict, device: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="photos_fast_") as temporary:
        manifest_path = os.path.join(temporary, "crops.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False)
        output = os.path.join(temporary, "crops-out.json")
        payload = _run_paddle_worker(
            ["--crops", manifest_path, output, "--device", device], output
        )
    return payload


class PaddleSession:
    """Persistent PP-OCRv6 worker: one engine initialization for many pages."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._process = None
        self._request_counter = 0
        self._spawn()

    def _spawn(self) -> None:
        if not os.path.isfile(PADDLE_PYTHON):
            raise RuntimeError(f"PaddleOCR environment is unavailable: {PADDLE_PYTHON}")
        environment = os.environ.copy()
        environment.update(PADDLE_ENV_OVERRIDES)
        environment["PYTHONIOENCODING"] = "utf-8"
        self._process = subprocess.Popen(
            [PADDLE_PYTHON, PADDLE_WORKER, "--serve", "--device", self._device],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _request(self, request: dict, timeout: float = 300) -> dict:
        if self._process is None or self._process.poll() is not None:
            self._spawn()
        self._request_counter += 1
        request = dict(request)
        request["id"] = f"req_{self._request_counter}"
        line = json.dumps(request, ensure_ascii=False) + "\n"
        try:
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(line)
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            self.close()
            self._spawn()
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(line)
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        if not response:
            raise RuntimeError("PaddleOCR serve worker closed without a response")
        payload = json.loads(response)
        if not payload.get("ok"):
            raise RuntimeError(f"PaddleOCR serve worker error: {payload.get('error')}")
        return payload

    def whole_page(self, masked_path: str) -> list[dict]:
        payload = self._request({"kind": "page", "input": masked_path})
        return payload.get("predictions", [])

    def crops(self, manifest: dict) -> dict:
        payload = self._request({"kind": "crops", "manifest": manifest})
        return {"crops": payload.get("crops", [])}

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.poll() is None and self._process.stdin is not None:
                self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _assign_paddle_predictions(column: dict, predictions: Sequence[dict]) -> int:
    """Attach Paddle polygons whose centre falls inside the column box."""
    x1, y1, x2, y2 = column["image_bbox"]
    hits = []
    for prediction in predictions:
        points = np.asarray(prediction.get("polygon") or [], dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            continue
        cx, cy = points.mean(axis=0)
        if x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2:
            hits.append(prediction)
    column["paddle_text"] = ""
    column["paddle_confidence"] = None
    if not hits:
        return 0
    if column["direction"] == "vertical":
        hits.sort(key=lambda item: (
            min(point[1] for point in item["polygon"]),
            min(point[0] for point in item["polygon"]),
        ))
    else:
        hits.sort(key=lambda item: (
            min(point[0] for point in item["polygon"]),
            min(point[1] for point in item["polygon"]),
        ))
    column["paddle_text"] = _normalize_text("".join(item["text"] for item in hits))
    confidences = [
        item["confidence"] for item in hits if item.get("confidence") is not None
    ]
    column["paddle_confidence"] = (
        round(float(np.mean(confidences)), 4) if confidences else None
    )
    return len(hits)


def _text_acceptable(text: str, confidence: float | None) -> bool:
    if not text or not _JAPANESE_RE.search(text):
        return False
    if confidence is None:
        # Windows.Media.Ocr exposes no confidence score; treat its text as
        # trusted when it contains Japanese.
        return True
    if len(text) == 1:
        return confidence >= 0.90
    return confidence >= 0.55


def _finalize_paddle_column(column: dict, text_source: str) -> bool:
    text = column.get("paddle_text") or ""
    confidence = column.get("paddle_confidence")
    if _text_acceptable(text, confidence):
        column["text"] = text
        column["text_source"] = text_source
        column["needs_review"] = False
        return True
    column["text"] = ""
    return False


def _retry_columns_with_crops(
    source: str,
    columns: Sequence[dict],
    device: str,
    paddle: PaddleSession | None = None,
    ocr_engine: str = "auto",
) -> None:
    """OCR unrecognized columns again, Windows OCR crops first."""
    failing = [column for column in columns if not column["text"]]
    if not failing:
        return
    if ocr_engine != "paddle" and paddle is None:
        # Windows.Media.Ocr is in-process and cheap: retry each failing column
        # on its own crop before paying for a PaddleOCR subprocess.
        image = np.array(Image.open(source).convert("RGB"))
        remaining: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="photos_fast_") as temporary:
            for column in failing:
                x1, y1, x2, y2 = column["image_bbox"]
                pad_x = max(6, (x2 - x1) // 4)
                pad_y = max(6, (y2 - y1) // 8)
                crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                crop = image[crop_y1:y2 + pad_y, crop_x1:x2 + pad_x]
                crop_path = os.path.join(temporary, "crop.png")
                Image.fromarray(crop).save(crop_path)
                predictions = windows_ocr_predictions(crop_path)
                if predictions:
                    for prediction in predictions:
                        prediction["polygon"] = [
                            [px + crop_x1, py + crop_y1]
                            for px, py in prediction["polygon"]
                        ]
                    _assign_paddle_predictions(column, predictions)
                    if _finalize_paddle_column(column, "windows-ocr-retry"):
                        continue
                remaining.append(column)
        failing = remaining
        if not failing:
            return
    if ocr_engine == "windows-ocr":
        return
    crops = []
    for column in failing:
        x1, y1, x2, y2 = column["image_bbox"]
        pad_x = max(6, (x2 - x1) // 4)
        pad_y = max(6, (y2 - y1) // 8)
        crops.append({
            "id": f"column_{failing.index(column):03d}",
            "bbox": [max(0, x1 - pad_x), max(0, y1 - pad_y), x2 + pad_x, y2 + pad_y],
            "variants": ["raw", "raw2x", "clahe", "otsu"],
        })
    if paddle is not None:
        payload = paddle.crops({"source": source, "crops": crops})
    else:
        payload = _run_paddle_crops({"source": source, "crops": crops}, device)
    for crop_result in payload.get("crops") or []:
        index = int(str(crop_result["id"]).rsplit("_", 1)[-1])
        if index >= len(failing):
            continue
        column = failing[index]
        for variant in ("raw", "raw2x", "clahe", "otsu"):
            predictions = (crop_result.get("variants") or {}).get(variant)
            if predictions is None:
                continue
            _assign_paddle_predictions(column, predictions)
            if _finalize_paddle_column(column, "paddle-retry"):
                break


def _group_paddle_columns(
    predictions: Sequence[dict],
    excluded_columns: Sequence[dict],
    image_size,
    text_source: str = "paddle-masked",
) -> list[dict]:
    """Group Paddle polygons that fall inside a page-scale island into columns.

    Photos sometimes preserves most of a white manga page as one OCR island.
    The island itself is then not split by glyph projection; the Paddle
    polygons give the fine column geometry instead.
    """
    def _inside_any(prediction: dict) -> bool:
        points = np.asarray(prediction.get("polygon") or [], dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            return True
        cx, cy = points.mean(axis=0)
        for column in excluded_columns:
            x1, y1, x2, y2 = column["image_bbox"]
            if x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2:
                return True
        return False

    remaining = [item for item in predictions if not _inside_any(item)]
    if not remaining:
        return []
    remaining.sort(key=lambda item: (min(p[0] for p in item["polygon"]), min(p[1] for p in item["polygon"])))
    clusters: list[list[dict]] = []

    for prediction in remaining:
        if clusters:
            last = clusters[-1][-1]
            ax1, ax2 = min(p[0] for p in last["polygon"]), max(p[0] for p in last["polygon"])
            bx1, bx2 = min(p[0] for p in prediction["polygon"]), max(p[0] for p in prediction["polygon"])
            overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            last_w = ax2 - ax1
            this_w = bx2 - bx1
            ay1, ay2 = min(p[1] for p in last["polygon"]), max(p[1] for p in last["polygon"])
            by1, by2 = min(p[1] for p in prediction["polygon"]), max(p[1] for p in prediction["polygon"])
            gap = max(0, by1 - ay2) if by1 > ay2 else max(0, ay1 - by2)
            last_h = ay2 - ay1
            this_h = by2 - by1
            # Adjacent manga columns often touch by a few pixels, and different
            # rows of panels share x bands.  Only polys that share most of
            # their horizontal band, sit close vertically and have comparable
            # sizes belong to one vertical column.
            if (
                overlap / max(1.0, min(last_w, this_w)) >= 0.45
                and gap <= max(30, 0.45 * min(last_h, this_h))
                and max(last_h, this_h) / max(1.0, min(last_h, this_h)) <= 1.6
            ):
                clusters[-1].append(prediction)
                continue
        clusters.append([prediction])

    image_w, image_h = image_size
    columns = []
    for cluster in clusters:
        points = np.asarray([p for item in cluster for p in item["polygon"]], dtype=np.float64).reshape(-1, 2)
        x1 = int(points[:, 0].min()) - 3
        y1 = int(points[:, 1].min()) - 3
        x2 = int(points[:, 0].max()) + 3
        y2 = int(points[:, 1].max()) + 3
        bbox = [max(0, x1), max(0, y1), min(image_w, x2), min(image_h, y2)]
        direction = "vertical" if (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0]) else "horizontal"
        hits = sorted(
            cluster,
            key=lambda item: (
                min(p[1] for p in item["polygon"]),
                min(p[0] for p in item["polygon"]),
            )
            if direction == "vertical"
            else (
                min(p[0] for p in item["polygon"]),
                min(p[1] for p in item["polygon"]),
            ),
        )
        text = _normalize_text("".join(item["text"] for item in hits))
        confidences = [
            item["confidence"] for item in hits if item.get("confidence") is not None
        ]
        confidence = round(float(np.mean(confidences)), 4) if confidences else None
        columns.append({
            "id": "",
            "text": text if _text_acceptable(text, confidence) else "",
            "paddle_text": text,
            "paddle_confidence": confidence,
            "text_source": text_source if _text_acceptable(text, confidence) else "",
            "needs_review": False,
            "display_bbox": None,
            "image_bbox": bbox,
            "selection_order": 0,
            "direction": direction,
            "role": "text",
            "overlay_block": -1,
        })
    return columns


def _split_columns_by_vertical_gaps(
    image: np.ndarray, columns: Sequence[dict]
) -> list[dict]:
    """Split merged columns whose ink projection contains a tall empty gap.

    Paddle's vertical-line polygons can carry so much padding that two
    different text blocks sharing one x band merge into a single column
    (e.g. ``戻ってきた`` stacked above ``打ち消して…``).  A real vertical
    column has small inter-character gaps (3-15 px); different text blocks
    leave a gap of several character heights.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    min_gap = max(30, round(image.shape[1] / 30))
    result: list[dict] = []
    for column in columns:
        x1, y1, x2, y2 = column["image_bbox"]
        if y2 - y1 < min_gap * 2 or column.get("paddle_confidence") is None:
            result.append(column)
            continue
        crop = gray[y1:y2, x1:x2]
        empty = crop < 110
        row_ink = empty.sum(axis=1)
        blank = row_ink < 2
        runs: list[tuple[int, int]] = []
        start = None
        for index, value in enumerate(blank.tolist() + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        boundaries = [
            (left + right) // 2
            for left, right in runs
            if right - left >= min_gap
        ]
        if not boundaries:
            result.append(column)
            continue
        spans = []
        previous = 0
        for boundary in boundaries:
            spans.append((previous, boundary))
            previous = boundary
        spans.append((previous, y2 - y1))
        for top, bottom in spans:
            top += y1
            bottom += y1
            if bottom - top < max(12, min_gap // 2):
                continue
            sub = dict(column)
            sub["image_bbox"] = [x1, top, x2, bottom]
            sub["paddle_text"] = ""
            sub["paddle_confidence"] = None
            sub["text"] = ""
            sub["text_source"] = ""
            sub["needs_review"] = False
            result.append(sub)
    return result


def _photos_fallback_copy(active_control, column: dict, viewport_size, image_size) -> None:
    """Click recorded pixel points in Photos and copy its OCR text."""
    from pywinauto import keyboard, mouse

    rect = active_control.rectangle()
    display = _image_bbox_to_display(column["image_bbox"], viewport_size, image_size)
    d1, d2, d3, d4 = display
    probes = []
    for fraction_x in (0.25, 0.5, 0.75):
        for fraction_y in (0.25, 0.5, 0.75):
            probes.append((
                round(d1 + (d3 - d1) * fraction_x),
                round(d2 + (d4 - d2) * fraction_y),
            ))
    seen_texts: set[str] = set()
    for probe in probes:
        px, py = probe
        if not (d1 <= px <= d3 and d2 <= py <= d4):
            continue
        marker = f"__PHOTOS_FAST_{time.time_ns()}__"
        _set_clipboard_marker(marker)
        mouse.click(coords=(rect.left + px, rect.top + py))
        time.sleep(0.12)
        keyboard.send_keys("^c")
        time.sleep(0.08)
        copied = _clipboard_text()
        if not copied or copied == marker:
            continue
        text = _normalize_text(copied)
        if not _JAPANESE_RE.search(text) or text in seen_texts:
            continue
        seen_texts.add(text)
        column["text"] = text
        column["text_source"] = "photos-fallback"
        column["needs_review"] = True
        break
    mouse.move(coords=(rect.right + 18, rect.bottom + 18))


def capture_photos_fast(
    path: str,
    timeout: float = 20,
    debug_dir: str | None = None,
    screenshot: str | None = None,
    skip_fallback: bool = False,
    paddle_device: str = "cpu",
    ocr_engine: str = "auto",
    window=None,
    paddle: PaddleSession | None = None,
    keep_sfx: bool = True,
) -> dict:
    """Fast capture: Photos island geometry + masked-page PaddleOCR.

    Photos supplies the highlighted OCR boxes (geometry only); everything
    outside them is blackened, PaddleOCR reads the islands in the background,
    unrecognized islands are retried with image variants, and only the columns
    that still have no text fall back to Photos click-copy.

    ``window`` reuses an already-open Photos window (batch mode) and
    ``paddle`` reuses a persistent PaddleOCR serve session.  ``keep_sfx``
    retains furigana/onomatopoeia islands as SFX columns instead of dropping
    them.
    """
    manifest, _ = _capture_photos_fast_page(
        window,
        path,
        timeout,
        debug_dir,
        screenshot,
        skip_fallback,
        paddle_device,
        ocr_engine,
        paddle,
        keep_sfx,
    )
    return manifest


def _capture_photos_fast_page(
    window,
    path: str,
    timeout: float = 20,
    debug_dir: str | None = None,
    screenshot: str | None = None,
    skip_fallback: bool = False,
    paddle_device: str = "cpu",
    ocr_engine: str = "auto",
    paddle: PaddleSession | None = None,
    keep_sfx: bool = True,
    frames_only: bool = False,
    fixed_window_rect: tuple[int, int, int, int] | None = None,
    display_scale: int = 100,
    box_delta: int = 8,
    box_tolerance: float = 5.0,
    box_area_ratio: float = 0.20,
) -> tuple[dict, object]:
    try:
        """One fast-mask page capture; returns ``(manifest, window)`` for reuse.

        ``frames_only`` captures only the three frames (base / overlay /
        all-selected) plus the blue selection boxes and returns immediately --
        no masked OCR, no column pass, no clipboard work.  All recognition
        happens later in a background batch, so the Photos window is held open
        for the shortest possible time.
        """
        timings: dict[str, float] = {}

        def _tick(name: str) -> None:
            timings[name] = round(time.monotonic() - start, 2)

        start = time.monotonic()
        foreground_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        source = os.path.abspath(path)
        image = Image.open(source).convert("RGB")
        image_arr = np.array(image)
        target_small = _thumbnail(image_arr)
        opened_window = window is None
        open_start = time.monotonic()
        if window is None:
            window = _open_photos(source, timeout)
            _tick("open_photos")
            photos_wait_open = timings["open_photos"]
        else:
            _launch_photos(source)
            deadline = time.monotonic() + timeout
            base_ready = False
            while time.monotonic() < deadline:
                try:
                    fresh = _get_control(window, "D2DImageControl", 0.6)
                    probe = _capture(fresh)
                    if probe is not None and _display_matches(target_small, probe):
                        base_ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.15)
            if not base_ready:
                raise RuntimeError(f"Photos window did not become available: {source}")
            photos_wait_open = round(time.monotonic() - open_start, 2)
        if fixed_window_rect is not None:
            _fix_window(window, fixed_window_rect)
        # The overlay frame must match the requested page.  Parallel
        # Photos windows race for the single Photos UWP instance:
        # ``os.startfile`` can land a different page in a freshly opened
        # window, so a mismatch is retried by closing the window and
        # opening a fresh one (up to 3 times) instead of failing the
        # page outright.  Only a persistent mismatch fails.
        overlay = None
        for attempt in range(3):
            if attempt > 0:
                if window is not None:
                    _close_window_wait(window)
                    window = None
                window = _open_photos(source, timeout)
                opened_window = True
            base, active, blocks, active_control = _prepare_overlay(
                window, timeout, screen_capture=True,
                fixed_window_rect=fixed_window_rect,
            )
            if _display_matches(target_small, base):
                overlay = (base, active, blocks, active_control)
                break
            print(
                f"warning: page {Path(source).name} window showed a "
                f"different image (attempt {attempt + 1}); reopening",
                file=sys.stderr,
            )
        if overlay is None:
            raise RuntimeError(f"Photos displayed a different image than requested: {source}")
        base, active, blocks, active_control = overlay
        _tick("overlay")
        timings["photos_wait_seconds"] = round(photos_wait_open + timings["overlay"], 2)
        viewport = (active.shape[1], active.shape[0])

        # Persist the base + scan-text frames right after the overlay
        # matches the requested page (before Ctrl+A / blue-box work), so
        # a page that later fails still leaves its scan frames on disk
        # for the review gate -- previously the frames were only saved
        # after selection_boxes succeeded, and a failing page kept only
        # the fallback original copy.
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(base).save(os.path.join(debug_dir, "photos_base.png"))
            Image.fromarray(active).save(os.path.join(debug_dir, "photos_ocr.png"))

        if fixed_window_rect is not None:
            _ensure_window_rect(window, fixed_window_rect)
        selection_blocks, selected_frame, selection_error = _capture_selection_boxes(
            window, active_control, active, debug_dir, base=base,
            fixed_window_rect=fixed_window_rect,
            box_delta=box_delta, box_tolerance=box_tolerance,
            box_area_ratio=box_area_ratio,
        )
        _tick("selection_boxes")
        selection_boxes_status = (
            "ok" if not selection_error else f"failed: {selection_error}"
        )
        selection_missing = bool(selection_error)
        # Hard rule: three Ctrl+A attempts produced no blue overlay pixels --
        # the page must be reviewed by the user before it can be closed.
        # ``selection_review_required`` gates the page out of downstream
        # processing until the user confirms the page genuinely has no boxes.
        selection_review_required = bool(selection_error)
        if selection_error:
            selection_blocks = []

        screenshot_path = None
        if screenshot:
            screenshot_path = os.path.abspath(screenshot)
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            Image.fromarray(active).save(screenshot_path)

        if frames_only:
            # Publish the frames + blue boxes and stop: no masked OCR, no
            # column pass.  Recognition runs later in a background batch.
            timings["total"] = round(time.monotonic() - start, 2)
            timings["foreground_started_at"] = foreground_started_at
            timings["foreground_ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            timings["foreground_active_seconds"] = timings["total"]
            manifest = {
                "schema_version": 2,
                "backend": "Windows Photos Scan Text",
                "source": {"path": source, "width": image.width, "height": image.height},
                "capture": {
                    "app": "Windows Photos",
                    "automation_control": "OcrControl",
                    "viewport_size": viewport,
                    "method": "frames-only",
                    "frames_only": True,
                    "display_scale": display_scale,
                    "screenshot": screenshot_path,
                    "timing": timings,
                    "island_count": len(blocks),
                    "selection_boxes_status": selection_boxes_status,
                    "selection_missing": selection_missing,
                    "selection_review_required": selection_review_required,
                    "selection_blocks": [
                        {
                            "id": f"sel_{index:03d}",
                            "display_bbox": [
                                block.x, block.y,
                                block.x + block.width, block.y + block.height,
                            ],
                            "image_bbox": _display_bbox_to_image(
                                [block.x, block.y, block.x + block.width, block.y + block.height],
                                viewport,
                                image.size,
                            ),
                            "page_scale": block.page_scale,
                        }
                        for index, block in enumerate(selection_blocks, 1)
                    ],
                    "selection_groups": [
                        {
                            "id": f"selg_{index:03d}",
                            "display_bbox": list(bbox),
                            "image_bbox": _display_bbox_to_image(
                                bbox, viewport, image.size
                            ),
                            "members": [
                                f"sel_{selection_blocks.index(member) + 1:03d}"
                                for member in members
                            ],
                        }
                        for index, (bbox, members) in enumerate(
                            aggregate_selection_boxes(selection_blocks), 1
                        )
                    ],
                    "complete": True,
                },
                "columns": [],
            }
            if opened_window:
                _close_window_wait(window)
                window = None
            return manifest, window

        viewport_area = active.shape[0] * active.shape[1]
        page_scale_blocks = [
            block
            for block in blocks
            if block.width * block.height >= 0.35 * viewport_area
        ]
        tiny_blocks = [
            block
            for block in blocks
            if block not in page_scale_blocks
            and (block.width < 10 or block.height < 12)
        ]
        normal_blocks = [
            block
            for block in blocks
            if block not in page_scale_blocks and block not in tiny_blocks
        ]
        if keep_sfx:
            # Tiny islands are usually furigana or SFX.  Keep them as SFX columns
            # so onomatopoeia is translated instead of being silently dropped.
            normal_blocks = normal_blocks + tiny_blocks

        islands = [
            _display_bbox_to_image(
                [block.x, block.y, block.x + block.width, block.y + block.height],
                viewport,
                image.size,
            )
            for block in blocks
        ]
        tiny_islands = [
            _display_bbox_to_image(
                [block.x, block.y, block.x + block.width, block.y + block.height],
                viewport,
                image.size,
            )
            for block in tiny_blocks
        ]

        masked = _build_island_mask(image_arr, islands)
        if debug_dir:
            Image.fromarray(masked).save(os.path.join(debug_dir, "photos_masked_ocr.png"))
            # Publish the blackened preview as soon as the islands are known so it
            # does not wait for PaddleOCR and the column pass.
            display_islands = [
                _display_bbox_to_image(
                    [block.x, block.y, block.x + block.width, block.y + block.height],
                    viewport,
                    image.size,
                )
                for block in blocks
                if block not in page_scale_blocks
            ]
            Image.fromarray(_build_island_mask(image_arr, display_islands)).save(
                os.path.join(debug_dir, "photos_masked.png")
            )
        # OCR runs in the background: keep Photos out of the foreground.
        try:
            window.minimize()
        except Exception:
            pass
        ocr_engine = ocr_engine
        with tempfile.TemporaryDirectory(prefix="photos_fast_") as temporary:
            masked_path = os.path.join(temporary, "masked.png")
            Image.fromarray(masked).save(masked_path)
            if paddle is not None:
                # PaddleSession 持久会话优先（生产未启用，保留此分支）
                predictions = paddle.whole_page(masked_path)
                ocr_engine = "paddle"
            else:
                predictions, ocr_engine = _run_masked_ocr(masked_path, ocr_engine, paddle_device)
        _tick("paddle_masked")

        def _island_polys(prediction_list, island_bbox) -> list[dict]:
            x1, y1, x2, y2 = island_bbox
            hits = []
            for prediction in prediction_list:
                points = np.asarray(
                    prediction.get("polygon") or [], dtype=np.float64
                ).reshape(-1, 2)
                if points.size == 0:
                    continue
                cx, cy = points.mean(axis=0)
                if x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2:
                    hits.append(prediction)
            return hits

        columns = []
        for block in normal_blocks:
            block_index = blocks.index(block)
            island_bbox = _display_bbox_to_image(
                [block.x, block.y, block.x + block.width, block.y + block.height],
                viewport,
                image.size,
            )
            sfx = keep_sfx and block in tiny_blocks
            # Photos box granularity varies between runs (one line per box or a
            # whole region per box), so OCR polygons define the columns inside an
            # island.  Bands not covered by any OCR-derived column are still added
            # so text the OCR engine missed (e.g. one line of a multi-line box)
            # can be recovered by the crop-retry / Photos copy chain.
            derived = _group_paddle_columns(
                _island_polys(predictions, island_bbox), [], image.size,
                f"{ocr_engine}-masked",
            )
            covered: list[list[int]] = []
            for column in derived:
                column["overlay_block"] = block_index
                if sfx:
                    column["role"] = "sfx"
                covered.append(
                    _image_bbox_to_display(column["image_bbox"], viewport, image.size)
                )
                columns.append(column)
            for band in _split_column_bands(active, block):
                bx1, by1, bx2, by2 = band
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                if any(
                    dx1 - 3 <= cx <= dx2 + 3 and dy1 - 3 <= cy <= dy2 + 3
                    for dx1, dy1, dx2, dy2 in covered
                ):
                    continue
                image_bbox = _display_bbox_to_image(band, viewport, image.size)
                columns.append({
                    "id": "",
                    "text": "",
                    "paddle_text": "",
                    "paddle_confidence": None,
                    "text_source": "",
                    "needs_review": False,
                    "display_bbox": band,
                    "image_bbox": image_bbox,
                    "selection_order": len(columns),
                    "direction": (
                        "vertical"
                        if (image_bbox[3] - image_bbox[1]) >= (image_bbox[2] - image_bbox[0])
                        else "horizontal"
                    ),
                    "role": "sfx" if sfx else "text",
                    "overlay_block": block_index,
                })

        if page_scale_blocks:
            # The page-scale island's interior is the whole page: Photos gave no
            # column geometry, so Paddle polygons supply it. Polygons already
            # claimed by normal/SFX islands are excluded.
            block_index = blocks.index(page_scale_blocks[0])
            excluded = [
                {
                    "image_bbox": _display_bbox_to_image(
                        [block.x, block.y, block.x + block.width, block.y + block.height],
                        viewport,
                        image.size,
                    )
                }
                for block in normal_blocks
            ]
            for column in _group_paddle_columns(
                predictions, excluded, image.size, f"{ocr_engine}-masked"
            ):
                column["overlay_block"] = block_index
                columns.append(column)

        for column in columns:
            if column.get("paddle_confidence") is None:
                _assign_paddle_predictions(column, predictions)
                _finalize_paddle_column(column, f"{ocr_engine}-masked")

        columns = _split_columns_by_vertical_gaps(image_arr, columns)
        for column in columns:
            if column.get("paddle_confidence") is None:
                _assign_paddle_predictions(column, predictions)
                _finalize_paddle_column(column, f"{ocr_engine}-masked")

        _retry_columns_with_crops(source, columns, paddle_device, paddle, ocr_engine)
        _tick("retry_crops")

        input_control_start = time.monotonic()
        needs_fallback = (
            [column for column in columns if not column["text"]] if not skip_fallback else []
        )
        if needs_fallback:
            # Re-fetch only when clicks are actually needed: Photos UIA can block
            # for many seconds while the window is busy re-rendering, and without
            # mouse activity the prepared control wrapper stays valid.
            try:
                window.restore()
                window.set_focus()
            except Exception:
                pass
            active_control = _get_control(window, "OcrControl", min(timeout, 5))
            for column in needs_fallback:
                _photos_fallback_copy(active_control, column, viewport, image.size)
            try:
                window.minimize()
            except Exception:
                pass
        _tick("fallback")

        inventory: list[str] = []
        inventory_error = None
        try:
            try:
                window.restore()
                window.set_focus()
            except Exception:
                pass
            if needs_fallback:
                active_control = _get_control(window, "OcrControl", min(timeout, 5))
            inventory = _inventory_text(active_control, blocks[0])
        except Exception as error:
            inventory_error = str(error)
        _tick("inventory")
        timings["input_control_seconds"] = round(time.monotonic() - input_control_start, 2)
        columns.sort(key=lambda item: (item["image_bbox"][1], -item["image_bbox"][2]))
        for index, column in enumerate(columns, 1):
            column["id"] = f"column_{index:03d}"
            column["selection_order"] = index - 1

        empty = [column for column in columns if not column["text"]]
        dropped_noise = []
        if keep_sfx:
            # Onomatopoeia and furigana boxes that neither engine could read are
            # retained as SFX columns instead of being dropped; empty text renders
            # nothing downstream but the geometry stays auditable.
            for column in empty:
                column["role"] = "sfx"
                column["text"] = column.get("paddle_text") or ""
                column["text_source"] = column.get("text_source") or "sfx-kept"
                column["needs_review"] = True
        else:
            for column in empty:
                # A confident, acceptable OCR reading that Photos refused to copy
                # is a genuine capture problem.  An empty or unacceptable box (noise
                # such as bubble tails, digits or dashed art) that Photos also cannot
                # select is dropped with an audit record instead of failing the page.
                if _text_acceptable(
                    column.get("paddle_text") or "",
                    column.get("paddle_confidence"),
                ):
                    raise RuntimeError(
                        f"Photos fast capture could not copy text for {column['id']} "
                        f"({column['image_bbox']}) even though the OCR engine read "
                        f"{column['paddle_text']!r}"
                    )
                dropped_noise.append({
                    "id": column["id"],
                    "image_bbox": column["image_bbox"],
                    "paddle_text": column.get("paddle_text") or "",
                    "paddle_confidence": column.get("paddle_confidence"),
                })
            if dropped_noise:
                columns = [column for column in columns if column["text"]]
                for index, column in enumerate(columns, 1):
                    column["id"] = f"column_{index:03d}"
                    column["selection_order"] = index - 1
        timings["total"] = round(time.monotonic() - start, 2)
        timings["foreground_started_at"] = foreground_started_at
        timings["foreground_ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        timings["foreground_active_seconds"] = timings["total"]
        if debug_dir:
            # The blackened preview was already published right after island
            # detection.  Only a page-scale island needs the column-based update
            # because Photos highlighted the whole page there.
            if page_scale_blocks:
                display_islands = [
                    _display_bbox_to_image(
                        [block.x, block.y, block.x + block.width, block.y + block.height],
                        viewport,
                        image.size,
                    )
                    for block in blocks
                    if block not in page_scale_blocks
                ]
                for column in columns:
                    display_islands.append(
                        _tight_ink_bbox(image_arr, column["image_bbox"], pad=6)
                    )
                Image.fromarray(_build_island_mask(image_arr, display_islands)).save(
                    os.path.join(debug_dir, "photos_masked.png")
                )
            with open(os.path.join(debug_dir, "timing.json"), "w", encoding="utf-8") as handle:
                json.dump(timings, handle, ensure_ascii=False, indent=2)
        manifest = {
            "schema_version": 2,
            "backend": "Windows Photos Scan Text",
            "source": {"path": source, "width": image.width, "height": image.height},
            "capture": {
                "app": "Windows Photos",
                "automation_control": "OcrControl",
                "viewport_size": viewport,
                "method": "fast-mask",
                "display_scale": display_scale,
                "screenshot": screenshot_path,
                "timing": timings,
                "inventory": inventory,
                "inventory_error": inventory_error,
                "inventory_count": len(inventory),
                "island_count": len(blocks),
                "page_scale_islands": len(page_scale_blocks),
                "selection_boxes_status": selection_boxes_status,
                    "selection_missing": selection_missing,
                "selection_blocks": [
                    {
                        "id": f"sel_{index:03d}",
                        "display_bbox": [block.x, block.y, block.x + block.width, block.y + block.height],
                        "image_bbox": _display_bbox_to_image(
                            [block.x, block.y, block.x + block.width, block.y + block.height],
                            viewport,
                            image.size,
                        ),
                        "page_scale": block.page_scale,
                    }
                    for index, block in enumerate(selection_blocks, 1)
                ],
                "selection_groups": [
                    {
                        "id": f"selg_{index:03d}",
                        "display_bbox": list(bbox),
                        "image_bbox": _display_bbox_to_image(bbox, viewport, image.size),
                        "members": [
                            f"sel_{selection_blocks.index(member) + 1:03d}"
                            for member in members
                        ],
                    }
                    for index, (bbox, members) in enumerate(
                        aggregate_selection_boxes(selection_blocks), 1
                    )
                ],
                "ignored_tiny_islands": tiny_islands if not keep_sfx else [],
                "dropped_noise_columns": dropped_noise,
                "sfx_columns": sum(column["role"] == "sfx" for column in columns),
                "column_count": len(columns),
                "ocr_engine": ocr_engine,
                "masked_columns": sum(
                    column["text_source"] in {"paddle-masked", "windows-ocr-masked"}
                    for column in columns
                ),
                "retry_columns": sum(
                    column["text_source"] in {"paddle-retry", "windows-ocr-retry"}
                    for column in columns
                ),
                "fallback_columns": sum(
                    column["text_source"] == "photos-fallback" for column in columns
                ),
                "complete": True,
            },
            "columns": columns,
        }
        # Do not leave Photos hanging in the foreground: a window opened by this
        # call is closed; a reused window (batch) stays minimized.
        if opened_window:
            _close_window_wait(window)
            window = None
        return manifest, window

    except Exception:
        # Never leak the Photos window this function opened: close it
        # before propagating so the next page starts from a clean slate.
        if opened_window and window is not None:
            _close_window_wait(window)
        raise
def capture_photos_quick(
    path: str,
    output_dir: str | Path,
    timeout: float = 20,
    *,
    close_window: bool = True,
) -> dict:
    """Capture one unselected OCR overlay and one full Photos text inventory."""
    source = str(Path(path).resolve())
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pil_image = Image.open(source).convert("RGB")
    image = np.asarray(pil_image)
    window = None
    try:
        window = _open_photos(source, timeout)
        if _try_get_control(window, "OcrControl") is not None:
            _get_control(window, "OcrButton").invoke()
            time.sleep(0.5)
        base_control = _get_control(window, "D2DImageControl")
        base = _capture(base_control)
        _get_control(window, "OcrButton").invoke()
        active_control = _get_control(window, "OcrControl", timeout)
        deadline = time.monotonic() + timeout
        active = None
        blocks: list[OverlayBlock] = []
        while time.monotonic() < deadline:
            window.set_focus()
            active_control = _get_control(window, "OcrControl", timeout)
            active = _capture(active_control)
            blocks = detect_overlay_blocks(base, active)
            if blocks:
                break
            time.sleep(0.25)
        if active is None or not blocks:
            raise RuntimeError("Photos OCR overlay did not produce detectable text")

        Image.fromarray(base).save(destination / "base.png")
        Image.fromarray(active).save(destination / "ocr-overlay.png")
        transform = calibrate_viewport_transform(image, base)
        inventory = _inventory_text(active_control, blocks[0])
        overlay_blocks = []
        for index, block in enumerate(blocks, 1):
            display_bbox = [block.x, block.y, block.x + block.width, block.y + block.height]
            overlay_blocks.append({
                "id": f"overlay_{index:03d}",
                "display_bbox": display_bbox,
                "image_bbox": map_display_bbox_affine(display_bbox, transform, pil_image.size),
            })
        return {
            "schema_version": 3,
            "backend": "Windows Photos Scan Text quick capture",
            "source": {
                "path": source,
                "width": pil_image.width,
                "height": pil_image.height,
                "sha256": _sha256(source),
            },
            "capture": {
                "app": "Windows Photos",
                "automation_control": "OcrControl",
                "quick": True,
                "foreground_input_control": True,
                "viewport_size": [active.shape[1], active.shape[0]],
                "base_image": "base.png",
                "ocr_overlay_image": "ocr-overlay.png",
                "coordinate_transform": transform,
                "complete": True,
            },
            "inventory": inventory,
            "inventory_count": len(inventory),
            "overlay_blocks": overlay_blocks,
        }
    finally:
        if close_window and window is not None:
            try:
                window.close()
            except Exception:
                pass


def ingest_quick_capture_artifacts(path: str | Path, output_dir: str | Path) -> dict:
    """Build a capture manifest from screenshots collected by an approved UI driver."""
    source_path = Path(path).resolve()
    destination = Path(output_dir).resolve()
    base_path = destination / "base.png"
    overlay_path = destination / "ocr-overlay.png"
    inventory_path = destination / "inventory.txt"
    for required in (base_path, overlay_path, inventory_path):
        if not required.is_file():
            raise RuntimeError(f"missing quick capture artifact: {required}")

    pil_image = Image.open(source_path).convert("RGB")
    source_image = np.asarray(pil_image)
    base = np.asarray(Image.open(base_path).convert("RGB"))
    active = np.asarray(Image.open(overlay_path).convert("RGB"))
    if base.shape != active.shape:
        raise RuntimeError(
            f"base/overlay screenshot dimensions differ: {base.shape} != {active.shape}"
        )
    transform = calibrate_viewport_transform(source_image, base)
    matrix = np.asarray(transform["source_to_display"], dtype=np.float64)
    source_corners = np.asarray([[[
        0.0,
        0.0,
    ], [
        float(pil_image.width),
        0.0,
    ], [
        float(pil_image.width),
        float(pil_image.height),
    ], [
        0.0,
        float(pil_image.height),
    ]]], dtype=np.float64)
    display_polygon = cv2.transform(source_corners, matrix).reshape(-1, 2).astype(np.float32)

    overlay_blocks = []
    for block in detect_overlay_blocks(base, active):
        center = (float(block.x + block.width / 2), float(block.y + block.height / 2))
        if cv2.pointPolygonTest(display_polygon, center, False) < 0:
            continue
        display_bbox = [block.x, block.y, block.x + block.width, block.y + block.height]
        overlay_blocks.append({
            "id": f"overlay_{len(overlay_blocks) + 1:03d}",
            "display_bbox": display_bbox,
            "image_bbox": map_display_bbox_affine(display_bbox, transform, pil_image.size),
        })
    if not overlay_blocks:
        raise RuntimeError("ingested Photos overlay did not contain any calibrated text blocks")

    inventory = [
        normalized
        for line in inventory_path.read_text(encoding="utf-8-sig").splitlines()
        if (normalized := _normalize_text(line))
    ]
    if not inventory:
        raise RuntimeError("ingested Photos text inventory is empty")
    return {
        "schema_version": 3,
        "backend": "Windows Photos Scan Text quick capture",
        "source": {
            "path": str(source_path),
            "width": pil_image.width,
            "height": pil_image.height,
            "sha256": _sha256(source_path),
        },
        "capture": {
            "app": "Windows Photos",
            "automation_control": "Computer Use",
            "quick": True,
            "foreground_input_control": True,
            "method": "approved-ui-artifact-ingest",
            "viewport_size": [active.shape[1], active.shape[0]],
            "base_image": "base.png",
            "ocr_overlay_image": "ocr-overlay.png",
            "inventory_file": "inventory.txt",
            "coordinate_transform": transform,
            "complete": True,
        },
        "inventory": inventory,
        "inventory_count": len(inventory),
        "overlay_blocks": overlay_blocks,
    }


def ingest_photos_batch(
    images: str | Path,
    output_dir: str | Path,
    *,
    pages: set[str] | None = None,
    resume: bool = False,
) -> dict:
    source_dir = Path(images).resolve()
    destination = Path(output_dir).resolve()
    image_paths = sorted(
        (
            path for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in _SUPPORTED_IMAGES
            and (pages is None or path.stem in pages)
        ),
        key=_natural_key,
    )
    results = []
    for image_path in image_paths:
        page_dir = destination / image_path.stem
        manifest_path = page_dir / "capture.json"
        if resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if (
                existing.get("capture", {}).get("complete")
                and existing.get("source", {}).get("sha256") == _sha256(image_path)
            ):
                results.append({"page": image_path.stem, "status": "reused", "manifest": str(manifest_path)})
                continue
        try:
            manifest = ingest_quick_capture_artifacts(image_path, page_dir)
            _write_json_atomic(manifest_path, manifest)
            results.append({"page": image_path.stem, "status": "ingested", "manifest": str(manifest_path)})
        except Exception as error:
            results.append({"page": image_path.stem, "status": "failed", "error": str(error), "manifest": str(manifest_path)})
    report = {
        "schema_version": 1,
        "artifact": "windows-photos-quick-capture-ingest",
        "images": str(source_dir),
        "output_dir": str(destination),
        "pages": results,
        "complete": bool(results)
        and all(
            item["status"] in {"captured", "reused", "black-page", "no-text"}
            for item in results
        ),
    }
    _write_json_atomic(destination / "batch.capture.json", report)
    return report


def capture_photos_batch(
    images: str | Path,
    output_dir: str | Path,
    *,
    pages: set[str] | None = None,
    timeout: float = 20,
    resume: bool = False,
) -> dict:
    source_dir = Path(images).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        (
            path for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in _SUPPORTED_IMAGES
            and (pages is None or path.stem in pages)
        ),
        key=_natural_key,
    )
    results = []
    for image_path in image_paths:
        page_dir = destination / image_path.stem
        manifest_path = page_dir / "capture.json"
        if resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if (
                existing.get("capture", {}).get("complete")
                and existing.get("source", {}).get("sha256") == _sha256(image_path)
            ):
                results.append({"page": image_path.stem, "status": "reused", "manifest": str(manifest_path)})
                continue
        try:
            manifest = capture_photos_quick(image_path, page_dir, timeout)
            _write_json_atomic(manifest_path, manifest)
            results.append({"page": image_path.stem, "status": "captured", "manifest": str(manifest_path)})
        except Exception as error:
            failure = {
                "schema_version": 3,
                "backend": "Windows Photos Scan Text quick capture",
                "source": {"path": str(image_path), "sha256": _sha256(image_path)},
                "capture": {"complete": False, "error": str(error)},
            }
            _write_json_atomic(manifest_path, failure)
            results.append({"page": image_path.stem, "status": "failed", "error": str(error), "manifest": str(manifest_path)})
    report = {
        "schema_version": 1,
        "artifact": "windows-photos-quick-capture-batch",
        "images": str(source_dir),
        "output_dir": str(destination),
        "pages": results,
        "complete": bool(results) and all(item["status"] in {"captured", "reused"} for item in results),
    }
    _write_json_atomic(destination / "batch.capture.json", report)
    return report


def _fix_window(
    window, rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Pin the Photos window to a fixed screen rect.

    A pinned window eliminates the 1-2px edge jitter between the base and
    overlay grabs (the root cause of the occasional "frames have different
    sizes" failures) and keeps rendering consistent.  Returns the actual
    window rect after SetWindowPos, or None when the move failed -- callers
    must verify the returned rect against the planned one.
    """
    try:
        import ctypes

        x1, y1, x2, y2 = (int(value) for value in rect)
        ctypes.windll.user32.SetWindowPos(
            window.handle,
            0,
            x1,
            y1,
            max(100, x2 - x1),
            max(100, y2 - y1),
            0x0004,  # SWP_NOZORDER
        )
        time.sleep(0.3)
        return _window_rect_tuple(window)
    except Exception:
        return None


def _prompt_fallback(image_path: str | Path) -> bool:
    """Ask the user whether to route a failed capture to the manga fallback.

    Returns True when the user clicks "Yes" (fallback-manga), False for
    "No" (page stays failed).  Falls back to False when no interactive
    desktop is available (e.g. a service session).
    """
    try:
        import ctypes

        message = (
            f"第 {Path(image_path).stem} 页（{Path(image_path).name}）采集失败。\n\n"
            "是否用 manga 兜底？\n"
            "（winrt/Paddle 文本定位 + manga 识别 + 翻译嵌字）"
        )
        answer = ctypes.windll.user32.MessageBoxW(
            None, message, "采集失败 - manga 兜底", 0x00000004 | 0x00000020
        )
        return answer == 6  # IDYES
    except Exception:
        return False


def _virtual_screen_bounds() -> tuple[int, int, int, int]:
    """(x, y, width, height) of the full virtual screen."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        vx = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        vy = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
        vw = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        vh = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        if vw <= 0 or vh <= 0:
            vx = vy = 0
            vw = user32.GetSystemMetrics(0)
            vh = user32.GetSystemMetrics(1)
    except Exception:
        vx, vy, vw, vh = 0, 0, 2560, 1660
    return vx, vy, vw, vh


def _work_area_bounds() -> tuple[int, int, int, int]:
    """(x, y, width, height) of the primary monitor work area (taskbar out)."""
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        # SPI_GETWORKAREA
        if ctypes.windll.user32.SystemParametersInfoW(
            0x0030, 0, ctypes.byref(rect), 0
        ):
            return (
                int(rect.left),
                int(rect.top),
                int(rect.right - rect.left),
                int(rect.bottom - rect.top),
            )
    except Exception:
        pass
    return _virtual_screen_bounds()


def _rect_close(
    actual: tuple[int, int, int, int],
    planned: tuple[int, int, int, int],
    tolerance: int = 2,
) -> bool:
    return all(
        abs(int(actual[index]) - int(planned[index])) <= tolerance
        for index in range(4)
    )


def _window_rect_tuple(window) -> tuple[int, int, int, int]:
    current = window.rectangle()
    return (
        int(current.left),
        int(current.top),
        int(current.right),
        int(current.bottom),
    )


def _page_display_rects(
    image_paths: Sequence[Path],
    scale: int,
    *,
    parallel: int = 3,
    margin: int = 24,
    gap: int = 32,
    min_width: int = 320,
    min_height: int = 240,
    max_per_row: int = 3,
    screen_bounds: tuple[int, int, int, int] | None = None,
    right_limit: int | None = None,
    row_gap_px: int | None = None,
    bottom_shift_up: int = 0,
) -> tuple[dict[str, tuple[int, int, int, int]], int, float]:
    """Window rects at up to ``scale`` percent of each page's size,
    edge-aligned on the usable area.

    Every rect keeps its page's aspect ratio, so Photos fit-in-view renders
    the whole page at exactly ``scale`` percent (the image file itself is
    never resized -- resolution is unchanged; coordinates map back to the
    original image through the existing fit transform).  The requested
    ``scale`` is an UPPER BOUND: when the rows cannot fit the usable area
    (``screen_bounds`` region, clipped by ``right_limit``), the zoom shrinks
    uniformly -- every window stays the same fraction of its page, so the
    alignment stays intact -- down to a 12% floor.  Only when even the
    floor cannot fit are rows dropped (fewer concurrent windows).

    The grid hugs the usable area's edges: the first row aligns to the top,
    a second row aligns to the bottom, the leftmost window hugs the left
    edge and the rightmost window hugs the right edge (``right_limit`` when
    given); columns keep at least ``gap`` px between windows
    (``row_gap_px`` fixes the column spacing instead of spreading).
    ``bottom_shift_up`` lifts the last row that many pixels off the bottom
    edge.  Rects never overlap and never cross the right limit or the
    bottom edge.

    NOTE: the window floor is a small absolute 320x240 -- NOT the 462x660
    hard minimum.  The 462x660 floor plus the measured Photos minimum
    (774x774) collapsed the parallel count from 6 to 2; edge-aligned
    sizing from display_scale keeps all 6 windows on the usable area.

    Returns ``({page_stem: rect}, effective_count, effective_fraction)``.
    """
    vx, vy, vw, vh = (
        screen_bounds if screen_bounds is not None else _virtual_screen_bounds()
    )
    right = min(vx + vw, right_limit) if right_limit else vx + vw
    base_fraction = max(1, int(scale)) / 100.0
    floor_fraction = 0.12
    requested = max(1, int(parallel))
    cols = max(1, int(max_per_row))
    usable_h = vh - 2 * margin
    usable_w = (right - vx) - 2 * margin
    min_width = max(1, int(min_width))
    min_height = max(1, int(min_height))
    if min_width > usable_w or min_height > usable_h:
        raise ValueError(
            f"cannot place even one window without overlap: minimum window "
            f"size {min_width}x{min_height} exceeds the layout area "
            f"{usable_w}x{usable_h}"
        )

    page_sizes: dict[str, tuple[int, int]] = {}
    for path in image_paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError):
            width = height = 0
        if width <= 0 or height <= 0:
            continue
        page_sizes[path.stem] = (width, height)
    if not page_sizes:
        return {}, 0, base_fraction
    keys = list(page_sizes.keys())

    def _slot_of(count: int) -> dict[str, int]:
        return {key: index % count for index, key in enumerate(keys)}

    def _win_dims(key: str, fraction: float) -> tuple[int, int]:
        width, height = page_sizes[key]
        return (
            max(min_width, min(usable_w, int(round(width * fraction)))),
            max(min_height, min(usable_h, int(round(height * fraction)))),
        )

    def _rows_fit(fraction: float, count: int, row_count: int) -> bool:
        """True when every row fits vertically and horizontally in the
        usable area (with the real minimum column gap)."""
        slot_of = _slot_of(count)
        row_w = [min_width] * row_count
        row_h = [min_height] * row_count
        for key in keys:
            row = slot_of[key] // cols
            if row >= row_count:
                continue
            w, h = _win_dims(key, fraction)
            row_w[row] = max(row_w[row], w)
            row_h[row] = max(row_h[row], h)
        if sum(row_h) + gap * (row_count - 1) > usable_h:
            return False
        col_gap = max(1, int(row_gap_px)) if row_gap_px is not None else max(1, gap)
        for row in range(row_count):
            n = min(cols, count - row * cols)
            if n <= 0:
                continue
            if row_w[row] * n + col_gap * (n - 1) > usable_w:
                return False
        return True

    row_count_full = (requested + cols - 1) // cols
    count = min(requested, row_count_full * cols)
    row_count = row_count_full
    fraction = base_fraction
    if not _rows_fit(base_fraction, count, row_count):
        # Auto-fit: keep as many concurrent windows as possible.  For each
        # candidate count (requested, requested-1, ... 1) find the largest
        # zoom that fits -- windows never shrink below the hard minimum
        # size and rects never overlap.  Raise when even a single window
        # cannot be placed.
        best = None
        for candidate in range(requested, 0, -1):
            rc = (candidate + cols - 1) // cols
            if not _rows_fit(floor_fraction, candidate, rc):
                continue
            lo, hi = floor_fraction, base_fraction
            if _rows_fit(hi, candidate, rc):
                best = (hi, candidate, rc)
            else:
                for _ in range(24):
                    mid = (lo + hi) / 2.0
                    if _rows_fit(mid, candidate, rc):
                        lo = mid
                    else:
                        hi = mid
                best = (lo, candidate, rc)
            break
        if best is None:
            raise ValueError(
                f"cannot place any of the {requested} windows without overlap: "
                f"layout area {usable_w}x{usable_h} is smaller than the "
                f"minimum window size {min_width}x{min_height}"
            )
        fraction, count, row_count = best

    slot_of = _slot_of(count)
    row_heights = [
        max(
            (_win_dims(key, fraction)[1] for key in keys if slot_of[key] // cols == row),
            default=min_height,
        )
        for row in range(row_count)
    ]

    # Vertical distribution: the first row hugs the top, a second row hugs
    # the bottom (lifted by ``bottom_shift_up`` when given), further rows
    # spread evenly.
    row_tops: list[int] = []
    if row_count == 1:
        row_tops.append(vy + margin)
    elif row_count == 2:
        top = vy + margin
        bottom = vy + vh - margin - row_heights[1] - bottom_shift_up
        row_tops = [top, max(top + gap, bottom)]
    else:
        total = sum(row_heights)
        pitch = max(gap, (usable_h - total) // (row_count - 1)) if row_count > 1 else gap
        cursor = vy + margin
        for height in row_heights:
            row_tops.append(cursor)
            cursor += height + pitch

    rects: dict[str, tuple[int, int, int, int]] = {}
    for row in range(row_count):
        in_row = [key for key in keys if slot_of[key] // cols == row]
        if not in_row:
            continue
        # Uniform spacing from the row's widest window so the row spans the
        # full usable width: the left window hugs the left edge, the right
        # window hugs the right edge.  The row holds at most ``cols``
        # slots -- the page count in the row may be larger.  ``row_gap_px``
        # fixes the column spacing instead of spreading the row (keeps the
        # rightmost window inset).
        row_w = max((_win_dims(key, fraction)[0] for key in in_row), default=min_width)
        n = min(cols, count - row * cols)
        total_w = row_w * n
        spread = usable_w - total_w
        if row_gap_px is not None:
            row_gap = max(1, int(row_gap_px)) if n > 1 else 0
        else:
            row_gap = max(gap, spread // (n - 1)) if n > 1 else 0
        xs: list[int] = []
        x = vx + margin
        for _ in range(n):
            xs.append(x)
            x += row_w + row_gap
        y = row_tops[row]
        for key in in_row:
            win_w, win_h = _win_dims(key, fraction)
            slot_x = xs[slot_of[key] % cols]
            rects[key] = (slot_x, y, slot_x + win_w, y + win_h)
    return rects, count, fraction


def _tile_rects(
    parallel: int,
    *,
    columns: int | None = None,
    rows: int | None = None,
    margin: int = 24,
    shrink: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Split the virtual screen into ``parallel`` vertical (portrait) tiles.

    Windows are portrait (wider than tall is avoided): ``parallel=3`` gives
    one row of 3 portrait windows (~800x1560 on a 2560x1660 screen),
    ``parallel=6`` gives 3 columns x 2 rows (~800x750).  The column count
    auto-picks a portrait-biased grid when not given.  Every window rect
    is at least the hard minimum window size and rows/columns never
    overlap; when the screen cannot hold the requested rows at the minimum
    height the layout collapses to a single row (or raises when even one
    window cannot fit).
    """
    vx, vy, vw, vh = _virtual_screen_bounds()
    count = max(1, int(parallel))
    natural_w, min_h = 462, 660
    usable_w = vw - 2 * margin
    usable_h = vh - 2 * margin
    if natural_w > usable_w or min_h > usable_h:
        raise ValueError(
            f"cannot place even one Photos window: minimum window size "
            f"{natural_w}x{min_h} exceeds the screen area {usable_w}x{usable_h}"
        )
    col_gap = 136
    cols = columns or 3
    while cols > 1 and natural_w * cols + col_gap * (cols - 1) > usable_w:
        cols -= 1
    row_count = int(math.ceil(count / cols))
    natural_h = min_h
    row_gap = 40
    if row_count > 1:
        # rows must never overlap: reserve a gap of 150% of the window
        # height between rows, solving h*rows + 1.5h*(rows-1) = total.
        fit_h = int(usable_h / (row_count + 1.5 * (row_count - 1)))
        if fit_h >= min_h:
            natural_h = fit_h
            row_gap = max(40, int(1.5 * natural_h))
        else:
            # Not enough vertical room for multiple rows at the minimum
            # height: collapse to a single row with enough columns.
            cols = count
            if natural_w * cols + col_gap * (cols - 1) > usable_w:
                raise ValueError(
                    f"cannot place {count} Photos windows in one non-overlapping "
                    f"row on a {usable_w}x{usable_h} screen area"
                )
            row_count = 1
    col_x = [
        vx + margin + index * (natural_w + col_gap)
        for index in range(cols)
    ]
    rects = []
    for index in range(count):
        col = index % cols
        row = index // cols
        x1 = col_x[col]
        y1 = vy + margin + row * (natural_h + row_gap)
        rects.append((x1, y1, x1 + natural_w, y1 + natural_h))
    return rects


def _ensure_window_rect(window, rect: tuple[int, int, int, int]) -> None:
    """Re-pin the Photos window to its tile before every screen grab.

    Parallel workers set focus on their own windows; Windows may move or
    restore a window between grabs, letting it drift over a neighbour's
    tile and pollute that neighbour's screen grab.  Locking the rectangle
    right before each grab makes overlap impossible.
    """
    try:
        x1, y1, x2, y2 = (int(value) for value in rect)
        current = window.rectangle()
        drift = (
            abs(current.left - x1) > 2
            or abs(current.top - y1) > 2
            or abs(current.right - x2) > 2
            or abs(current.bottom - y2) > 2
        )
        if drift:
            _fix_window(window, rect)
    except Exception:
        pass


_BLACK_PAGE_DARK_FRACTION = 0.90


def _is_black_page(path: str | Path) -> bool:
    """A solid black / no-content page (e.g. a back cover) is normal.

    Photos renders no image control for it, so capture would waste retries
    and dialogs on something that has nothing to translate.
    """
    try:
        image = Image.open(path).convert("L")
        small = image.resize((64, 64))
        arr = np.asarray(small, dtype=np.float32)
        return float((arr < 60).mean()) > _BLACK_PAGE_DARK_FRACTION
    except Exception:
        return False


def _capture_one_page(
    image_path: str | Path,
    destination: Path,
    timeout: float,
    resume: bool,
    paddle_device: str,
    max_attempts: int,
    ask_fallback: bool,
    frames_only: bool,
    fixed_rect: tuple[int, int, int, int] | None,
    display_scale: int = 100,
    box_delta: int = 8,
    box_tolerance: float = 5.0,
    box_area_ratio: float = 0.20,
    ocr_engine: str = "auto",
) -> tuple[dict, float | None]:
    """Capture one page with per-page retries.

    Returns ``(result_dict, foreground_seconds_or_None)``.
    """
    page_dir = destination / image_path.stem
    manifest_path = page_dir / "capture.json"
    if _is_black_page(image_path):
        # A solid black page is normal: no retries, no fallback dialog.
        failure = {
            "schema_version": 2,
            "backend": "Windows Photos Scan Text",
            "source": {"path": str(image_path), "sha256": _sha256(image_path)},
            "capture": {
                "complete": False,
                "error": "black page (no content)",
                "black_page": True,
                "attempts": 0,
                "display_scale": display_scale,
            },
        }
        _write_json_atomic(manifest_path, failure)
        return (
            {
                "page": image_path.stem,
                "status": "black-page",
                "manifest": str(manifest_path),
            },
            None,
        )
    if resume and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if (
                existing.get("capture", {}).get("complete")
                and existing.get("source", {}).get("sha256") == _sha256(image_path)
                and existing.get("capture", {}).get("display_scale", 100) == display_scale
            ):
                return (
                    {
                        "page": image_path.stem,
                        "status": "reused",
                        "selection_boxes": len(
                            existing.get("capture", {}).get("selection_blocks") or []
                        ),
                        "selection_missing": bool(
                            existing.get("capture", {}).get("selection_missing")
                        ),
                        "selection_review_required": bool(
                            existing.get("capture", {}).get("selection_review_required")
                        ),
                        "manifest": str(manifest_path),
                    },
                    None,
                )
        except (OSError, ValueError):
            pass
    attempts = 0
    last_error = ""
    window = None
    # The window's lifetime is the whole attempt loop: even the success
    # path (which used to leak the Photos window -- causing a "ghost"
    # window to sit on the desktop after capture) is now closed by the
    # finally block.  Each Photos page opens one window; we always close
    # it before returning, success or failure.
    try:
        while attempts < max(1, max_attempts):
            attempts += 1
            try:
                page_dir.mkdir(parents=True, exist_ok=True)
                # A fresh Photos window per page: startfile does not guarantee
                # same-window reuse, and stale OCR sessions make title/control
                # matching unreliable.  Content matching still guards against
                # capturing the wrong page.  Frames-only mode closes the window
                # as soon as the frames + blue boxes are captured.
                manifest, window = _capture_photos_fast_page(
                    None,
                    str(image_path),
                    timeout,
                    debug_dir=str(page_dir),
                    paddle_device=paddle_device,
                    ocr_engine=ocr_engine,
                    screenshot=str(page_dir / "photos_ocr.png"),
                    paddle=None,
                    frames_only=frames_only,
                    fixed_window_rect=fixed_rect,
                    display_scale=display_scale,
                    box_delta=box_delta,
                    box_tolerance=box_tolerance,
                    box_area_ratio=box_area_ratio,
                )
                manifest["source"]["sha256"] = _sha256(image_path)
                capture = manifest.setdefault("capture", {})
                capture["attempts"] = attempts
                _write_json_atomic(manifest_path, manifest)
                if not frames_only:
                    inventory = manifest.get("capture", {}).get("inventory") or []
                    (page_dir / "inventory.txt").write_text(
                        "\n".join(inventory), encoding="utf-8"
                    )
                timing = manifest.get("capture", {}).get("timing") or {}
                active = timing.get("foreground_active_seconds")
                seconds = float(active) if isinstance(active, (int, float)) else None
                return (
                    {
                        "page": image_path.stem,
                        "status": "captured",
                        "attempts": attempts,
                        "selection_boxes": len(
                            manifest.get("capture", {}).get("selection_blocks") or []
                        ),
                        "selection_missing": bool(
                            manifest.get("capture", {}).get("selection_missing")
                        ),
                        "selection_review_required": bool(
                            manifest.get("capture", {}).get("selection_review_required")
                        ),
                        "manifest": str(manifest_path),
                    },
                    seconds,
                )
            except PhotosNoTextError as error:
                # A page Photos cannot find text on is usually a pure
                # illustration/cover, but it still goes to the review gate so
                # the user can confirm before it is skipped.
                failure = {
                    "schema_version": 2,
                    "backend": "Windows Photos Scan Text",
                    "source": {"path": str(image_path), "sha256": _sha256(image_path)},
                    "capture": {
                        "complete": False,
                        "error": str(error),
                        "attempts": attempts,
                        "no_text": True,
                        "selection_review_required": True,
                        "display_scale": display_scale,
                    },
                }
                _write_json_atomic(manifest_path, failure)
                _ensure_review_image(page_dir, image_path)
                return (
                    {
                        "page": image_path.stem,
                        "status": "no-text",
                        "selection_review_required": True,
                        "manifest": str(manifest_path),
                    },
                    None,
                )
            except Exception as error:
                last_error = str(error)
                # A failed attempt: drop the dead window before the next try
                # (it can be hung on a different image) and back off.
                _close_window_wait(window)
                window = None
                if attempts < max(1, max_attempts):
                    time.sleep(2)
        # Hard rule: a failed capture is NEVER silently closed -- it goes to the
        # review gate (``selection_review_required``).  The manga fallback is a
        # review-page decision (one click there), not a blocking dialog here.
        failure = {
            "schema_version": 2,
            "backend": "Windows Photos Scan Text",
            "source": {"path": str(image_path), "sha256": _sha256(image_path)},
            "capture": {
                "complete": False,
                "error": last_error,
                "attempts": attempts,
                "selection_review_required": True,
                "display_scale": display_scale,
            },
        }
        _write_json_atomic(manifest_path, failure)
        _ensure_review_image(page_dir, image_path)
        return (
            {
                "page": image_path.stem,
                "status": "failed",
                "error": last_error,
                "attempts": attempts,
                "selection_review_required": True,
                "manifest": str(manifest_path),
            },
            None,
        )
    finally:
        # Final safety net: the success path returned early above, so the
        # outer loop and any unhandled exception path land here.  Closing
        # the window is idempotent -- _close_window_wait no-ops on a
        # window that has already gone.
        if window is not None:
            _close_window_wait(window)


def capture_photos_fast_batch(
    images: str | Path,
    output_dir: str | Path,
    *,
    pages: set[str] | None = None,
    timeout: float = 20,
    resume: bool = False,
    paddle_device: str = "cpu",
    close_window: bool = True,
    max_attempts: int = 3,
    ask_fallback: bool = True,
    parallel: int = 1,
    frames_only: bool = False,
    ocr_engine: str = "auto",
    fixed_window: bool = True,
    display_scale: int = 100,
    row_gap_px: int | None = None,
    bottom_shift_up: int = 0,
    layout_region: str = "",
    box_delta: int = 8,
    box_tolerance: float = 5.0,
    box_area_ratio: float = 0.20,
) -> dict:
    """Batch fast capture: one Photos window per page, retried up to
    ``max_attempts`` times (a fresh window each attempt; first success
    exits the retry loop).  When a page fails all attempts and
    ``ask_fallback`` is set, a dialog asks whether to route the page to
    the manga fallback path (winrt/Paddle text location + manga reading).

    ``frames_only`` captures only the three frames + blue selection boxes
    (no masked OCR / column pass) so the window is held open as briefly as
    possible; recognition runs later in a background batch.  ``parallel``
    tiles several Photos windows side by side and captures pages
    concurrently (keyboard/focus sequences are serialized internally).
    ``fixed_window`` pins every window to a tile rect (removes the 1-2px
    edge jitter and keeps rendering consistent).

    ``display_scale`` (percent) pins every window to ``display_scale``
    percent of its page's size instead of the fixed portrait tiles: the
    whole page is always fully visible at a fixed zoom, the image file is
    never resized, and the existing fit transform maps coordinates back to
    the original resolution.
    """
    source_dir = Path(images).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        (
            path for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in _SUPPORTED_IMAGES
            and (pages is None or path.stem in pages)
        ),
        key=_natural_key,
    )
    results: list[dict] = []
    foreground_totals: list[float] = []
    lock = threading.Lock()
    requested = max(1, int(parallel))
    count = requested
    effective_scale = display_scale
    scale_rects = None
    if display_scale != 100:
        region_bounds = None
        if layout_region:
            try:
                region_w, region_h = (
                    int(part.strip()) for part in layout_region.lower().split("x")
                )
                region_bounds = (0, 0, region_w, region_h)
            except (ValueError, TypeError):
                raise ValueError(
                    f"invalid --layout-region {layout_region!r} (expected WxH, e.g. 2400x1400)"
                ) from None
        else:
            # No explicit region: use the primary monitor work area (taskbar
            # excluded) so the bottom row is never hidden or overlapped.
            region_bounds = _work_area_bounds()
        # Edge-aligned display-scale layout (window = page x display_scale%,
        # 320x240 floor), exactly the original f984790 sizing that made
        # six parallel windows fit.  No measured-minimum probing and no
        # overlap guard: windows grab exactly where they are.
        scale_rects, count, scale_fraction = _page_display_rects(
            image_paths,
            display_scale,
            parallel=requested,
            right_limit=DISPLAY_RIGHT_LIMIT or None,
            row_gap_px=row_gap_px,
            bottom_shift_up=bottom_shift_up,
            screen_bounds=region_bounds,
        )
        effective_scale = round(scale_fraction * 100)
    tiles = (
        _tile_rects(parallel)
        if (fixed_window or parallel > 1) and scale_rects is None
        else None
    )
    if scale_rects is not None:
        print(
            f"layout: {count} concurrent windows at {effective_scale}% "
            f"(requested {display_scale}%)"
        )

    # Sweep orphan Photos windows from a previous (killed) batch.  Anything
    # whose title is a page name we are about to capture would race against
    # the new workers and confuse the blue-box detector; close them once
    # before the threads start.
    residual = _close_residual_photos_windows({path.name for path in image_paths})
    if residual:
        print(
            f"cleared {residual} residual Photos window(s) from a previous run",
            file=sys.stderr,
        )
        time.sleep(0.5)  # let Windows settle before we open new windows

    def _work(paths: list[Path], slot: int) -> None:
        for image_path in paths:
            rect = scale_rects.get(image_path.stem) if scale_rects is not None else (
                tiles[slot] if tiles else None
            )
            result, seconds = _capture_one_page(
                image_path,
                destination,
                timeout,
                resume,
                paddle_device,
                max_attempts,
                ask_fallback,
                frames_only,
                rect,
                display_scale,
                box_delta=box_delta,
                box_tolerance=box_tolerance,
                box_area_ratio=box_area_ratio,
                ocr_engine=ocr_engine,
            )
            with lock:
                results.append(result)
                if seconds is not None:
                    foreground_totals.append(seconds)

    count = max(1, int(count))
    if count > 1:
        chunks = [image_paths[index::count] for index in range(count)]
        threads = [
            threading.Thread(target=_work, args=(chunk, index))
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _work(image_paths, 0)

    def _percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(float(ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]), 2)

    foreground = {
        "pages_measured": len(foreground_totals),
        "total_foreground_seconds": round(sum(foreground_totals), 2),
        "median_foreground_seconds": _percentile(foreground_totals, 0.5),
        "p95_foreground_seconds": _percentile(foreground_totals, 0.95),
    }
    no_blue_boxes = [
        item["page"]
        for item in results
        if item["status"] in {"captured", "reused"}
        and item.get("selection_boxes", 0) == 0
    ]
    review_required = [
        item["page"]
        for item in results
        if item.get("selection_review_required")
    ]
    report = {
        "schema_version": 2,
        "artifact": "windows-photos-fast-capture-batch",
        "images": str(source_dir),
        "output_dir": str(destination),
        "mode": "frames-only" if frames_only else "full",
        "parallel": requested,
        "effective_parallel": count,
        "display_scale": display_scale,
        "effective_display_scale": effective_scale,
        "foreground": foreground,
        "pages": results,
        "no_blue_boxes": no_blue_boxes,
        "selection_review_required": review_required,
        "complete": bool(results)
        and all(
            # failed / fallback-manga pages are EXPECTED outcomes: they land
            # in the review gate (or the manga fallback path) and must not
            # turn the whole batch into an exit-code failure.
            item["status"] in {
                "captured", "reused", "black-page", "no-text",
                "failed", "fallback-manga",
            }
            for item in results
        ),
    }
    _write_json_atomic(destination / "batch.capture.json", report)
    return report


def recapture_photos_columns(
    capture_dir: str | Path,
    fusion_dir: str | Path | None = None,
    *,
    pages: set[str] | None = None,
    timeout: float = 20,
) -> dict:
    """Re-capture Photos text for conflict columns only.

    Loads each fast-mask capture manifest, opens the source image in Photos,
    and click-copies the columns flagged for review by the fusion stage
    (inventory unmatched, Photos text conflict, low confidence, fallback).
    Only the affected columns are touched; the patched manifest keeps every
    other column byte-for-byte stable.
    """
    from pywinauto import keyboard, mouse  # noqa: F401  (used by fallback helper)

    capture_root = Path(capture_dir).resolve()
    fusion_root = Path(fusion_dir).resolve() if fusion_dir else None
    results = []
    window = None
    pages_dir = capture_root.iterdir()
    for page_dir in sorted(pages_dir, key=lambda item: item.name.lower()):
        if not page_dir.is_dir():
            continue
        page = page_dir.name
        if pages is not None and page not in pages:
            continue
        manifest_path = page_dir / "capture.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        source = manifest.get("source", {}).get("path")
        if not source or not os.path.isfile(source):
            results.append({"page": page, "status": "skipped", "reason": "missing-source"})
            continue
        target_ids: set[str] = set()
        if fusion_root is not None:
            fusion_path = fusion_root / f"{page}.ocr.json"
            if fusion_path.is_file():
                fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
                for item in fusion.get("manual_review_queue", []):
                    if item.get("status") in {"pending", "bypassed"}:
                        target_ids.add(str(item.get("column_id") or ""))
        if not target_ids:
            results.append({"page": page, "status": "no-conflicts"})
            continue
        try:
            window = _open_photos(source, timeout)
            base, active, blocks, active_control = _prepare_overlay(window, timeout)
            viewport = (active.shape[1], active.shape[0])
            image = Image.open(source).convert("RGB")
            recaptured = []
            for column in manifest.get("columns", []):
                if str(column.get("id") or "") not in target_ids:
                    continue
                before = column.get("text", "")
                _photos_fallback_copy(active_control, column, viewport, image.size)
                if column.get("text") and column.get("text") != before:
                    column["text_source"] = "photos-recapture"
                    column["needs_review"] = False
                    recaptured.append({
                        "id": column["id"],
                        "before": before,
                        "after": column["text"],
                    })
            capture = manifest.setdefault("capture", {})
            capture["recapture_columns"] = len(recaptured)
            _write_json_atomic(manifest_path, manifest)
            results.append({
                "page": page,
                "status": "recaptured" if recaptured else "no-change",
                "recaptured_columns": recaptured,
            })
        except Exception as error:
            results.append({"page": page, "status": "failed", "error": str(error)})
            _close_window_wait(window)
            window = None
    report = {
        "schema_version": 1,
        "artifact": "windows-photos-fast-capture-recapture",
        "capture_dir": str(capture_root),
        "fusion_dir": str(fusion_root),
        "pages": results,
        "complete": bool(results)
        and all(item["status"] in {"recaptured", "no-change", "no-conflicts"} for item in results),
    }
    _write_json_atomic(capture_root / "recapture.json", report)
    return report


def _tight_ink_bbox(image: np.ndarray, bbox: Sequence[int], pad: int = 0) -> list[int]:
    """Shrink a column box to its dark glyph pixels for display masks."""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return [x1, y1, x2, y2]
    ink = region < 120
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return [x1, y1, x2, y2]
    return [
        max(0, x1 + int(cols.min()) - pad),
        max(0, y1 + int(rows.min()) - pad),
        min(image.shape[1], x1 + int(cols.max()) + 1 + pad),
        min(image.shape[0], y1 + int(rows.max()) + 1 + pad),
    ]


def _thumbnail(image: np.ndarray, size: int = 48) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _display_matches(target_small: np.ndarray, probe) -> bool:
    """Whether the Photos control currently displays the requested image."""
    probe_small = _thumbnail(np.array(probe))
    target_flat = target_small.ravel().astype(np.float32)
    probe_flat = probe_small.ravel().astype(np.float32)
    if float(np.std(target_flat)) == 0 or float(np.std(probe_flat)) == 0:
        return False
    correlation = float(np.corrcoef(target_flat, probe_flat)[0, 1])
    # 0.6 instead of 0.8: parallel-window rendering jitter (Photos still
    # animating the fit-in-view / scan-text dim) can push an identical
    # page's 48px thumbnail below 0.8; 0.6 keeps the guard meaningful
    # while not mis-killing pages whose window is genuinely showing the
    # right image.  The later blue-box detection double-checks the page
    # content anyway.
    return correlation > 0.6


def _prepare_overlay(
    window, timeout: float, screen_capture: bool = False,
    fixed_window_rect: tuple[int, int, int, int] | None = None,
):
    """Reset any previous OCR session and capture base/active frames.

    Returns ``(base, active, blocks, active_control)`` where ``blocks`` are the
    white-outline OCR islands visible in ``active``.  With ``screen_capture``
    the frames are grabbed from the actual screen region so composition-layer
    overlays (gray scan-text mask, highlighted islands) are visible.
    """
    # OCR mode replaces D2DImageControl with OcrControl.  Reset it first so a
    # previous run or manual Photos session cannot break automatic capture.
    if _try_get_control(window, "OcrControl") is not None:
        _get_control(window, "OcrButton").invoke()
        time.sleep(0.5)
    base_control = _get_control(window, "D2DImageControl")
    if fixed_window_rect is not None:
        _ensure_window_rect(window, fixed_window_rect)
    if screen_capture:
        base = _capture_control_region(window, base_control)
    else:
        base = _capture(base_control)
    button = _get_control(window, "OcrButton")
    button.invoke()

    deadline = time.monotonic() + (max(timeout, 30.0) if screen_capture else timeout)
    active = None
    blocks = []
    while time.monotonic() < deadline:
        with _INPUT_LOCK:
            window.set_focus()
        if fixed_window_rect is not None:
            _ensure_window_rect(window, fixed_window_rect)
        try:
            active_control = _get_control(window, "OcrControl", timeout)
        except RuntimeError as error:
            # Scan-text mode was requested but the OCR control never
            # appeared: Photos found no text on this page.
            raise PhotosNoTextError(
                "Photos found no text (OcrControl never appeared)"
            ) from error
        if screen_capture:
            active = _capture_control_region(window, active_control)
        else:
            active = _capture(active_control)
        if active.shape != base.shape:
            try:
                active, base = _align_frames(active, base)
            except ValueError:
                # Photos changed its layout between the base grab and the
                # scan-text switch (the OCR panel resized the window):
                # re-grab the base frame so both frames share the current
                # layout, then keep polling until the sizes agree.
                try:
                    fresh_base_control = _get_control(window, "D2DImageControl", 0.6)
                    if screen_capture:
                        base = _capture_control_region(window, fresh_base_control)
                    else:
                        base = _capture(fresh_base_control)
                except Exception:
                    pass
                if active.shape != base.shape:
                    time.sleep(0.3)
                    continue
        try:
            blocks = detect_overlay_blocks(base, active)
        except ValueError:
            # base/active still disagree (e.g. the window is mid-resize):
            # re-grab the base frame on the next pass.
            time.sleep(0.3)
            continue
        if blocks:
            break
        time.sleep(0.15)
    if not blocks or active is None:
        raise PhotosNoTextError("Photos OCR overlay did not produce detectable text")
    return base, active, blocks, active_control


def capture_photos_columns(path: str, timeout: float = 20, debug_dir: str | None = None) -> dict:
    from pywinauto import keyboard, mouse

    source = os.path.abspath(path)
    image = Image.open(source).convert("RGB")
    window = _open_photos(source, timeout)
    base, active, blocks, active_control = _prepare_overlay(window, timeout)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(base).save(os.path.join(debug_dir, "photos_base.png"))
        Image.fromarray(active).save(os.path.join(debug_dir, "photos_ocr.png"))

    rect = active_control.rectangle()
    seen: list[dict] = []
    click_index = 0
    click_groups = [
        (block_index, _projection_clicks(active, block))
        for block_index, block in enumerate(blocks)
    ]
    # Photos sometimes preserves most of a white manga page as one OCR island.
    # Probe likely glyph centres so separate columns within that island remain
    # selectable; copied text and geometry de-duplication discard repeated hits.
    click_groups.append((-1, _glyph_clicks(base)))
    for block_index, clicks in click_groups:
        for click in clicks:
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
            # Character-centre probing intentionally hits each vertical column
            # several times. Avoid another screenshot when the click still lies
            # inside a geometry already captured for the same copied text.
            if any(
                prior["text"] == text
                and prior["display_bbox"][0] - 6 <= click[0] <= prior["display_bbox"][2] + 6
                and prior["display_bbox"][1] - 6 <= click[1] <= prior["display_bbox"][3] + 6
                for prior in seen
            ):
                continue
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
    ignored_inventory_noise = list(missing_counts.elements())
    # On sparse punctuation Photos occasionally emits one unstable kanji/kana
    # (the same 19.jpg run alternated between 仄 and し). Do not treat a lone,
    # unlocatable OCR hallucination as manga dialogue; all multi-character
    # columns remain strict and captured extras are never accepted.
    if (
        sum(missing_counts.values()) == 1
        and not extra_counts
        and len(next(iter(missing_counts))) == 1
    ):
        inventory_counts -= missing_counts
        missing_counts.clear()
    else:
        ignored_inventory_noise = []
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
            "ignored_inventory_noise": ignored_inventory_noise,
        },
        "columns": columns,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "recapture":
        parser = argparse.ArgumentParser(description="Re-capture conflict columns from Windows Photos")
        parser.add_argument("recapture", nargs="?")
        parser.add_argument("--capture-dir", required=True)
        parser.add_argument("--fusion-dir")
        parser.add_argument("--pages")
        parser.add_argument("--timeout", type=float, default=20)
        args = parser.parse_args(raw_args)
        result = recapture_photos_columns(
            args.capture_dir,
            args.fusion_dir,
            pages=_parse_pages(args.pages),
            timeout=args.timeout,
        )
        print(Path(args.capture_dir).resolve() / "recapture.json")
        return 0 if result["complete"] else 2
    if raw_args and raw_args[0] == "ingest":
        parser = argparse.ArgumentParser(description="Ingest approved Windows Photos capture artifacts")
        parser.add_argument("ingest", nargs="?")
        parser.add_argument("--images", required=True)
        parser.add_argument("--pages")
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args(raw_args)
        result = ingest_photos_batch(
            args.images,
            args.output_dir,
            pages=_parse_pages(args.pages),
            resume=args.resume,
        )
        print(Path(args.output_dir).resolve() / "batch.capture.json")
        return 0 if result["complete"] else 2
    if raw_args and raw_args[0] == "batch":
        parser = argparse.ArgumentParser(description="Batch capture from Windows Photos")
        parser.add_argument("batch", nargs="?")
        parser.add_argument("--images", required=True)
        parser.add_argument("--pages")
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--quick", action="store_true")
        parser.add_argument(
            "--fast",
            action="store_true",
            help="fast masked-island capture with a persistent PaddleOCR session",
        )
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--timeout", type=float, default=20)
        parser.add_argument("--paddle-device", default="cpu")
        parser.add_argument(
            "--max-attempts",
            type=int,
            default=3,
            help="per-page capture attempts (default 3; first success exits)",
        )
        parser.add_argument(
            "--no-ask-fallback",
            action="store_true",
            help="do not ask for manga fallback when a page fails",
        )
        parser.add_argument(
            "--frames-only",
            action="store_true",
            help="capture only frames + blue boxes (no masked OCR / columns)",
        )
        parser.add_argument(
            "--ocr-engine",
            choices=["auto", "windows-ocr", "paddle"],
            default="auto",
            help="masked-page OCR engine (default auto: Windows OCR then Paddle fallback; forced engines fail loudly)",
        )
        parser.add_argument(
            "--parallel",
            type=int,
            default=1,
            help="parallel Photos windows (tiled side by side, default 1)",
        )
        parser.add_argument(
            "--display-scale",
            type=int,
            default=100,
            help="pin every window to this percent of its page's size "
                 "(default 100 = fixed portrait tiles; e.g. 25 shows the "
                 "whole page at 25%% zoom, image file never resized)",
        )
        parser.add_argument(
            "--layout-row-gap",
            type=int,
            default=None,
            help="fixed per-column spacing (px) in --display-scale layout; "
                 "default spreads the row across the usable width",
        )
        parser.add_argument(
            "--layout-bottom-shift-up",
            type=int,
            default=0,
            help="lift the last row of windows this many px off the bottom "
                 "edge in --display-scale layout (default 0)",
        )
        parser.add_argument(
            "--layout-region",
            default="",
            help="constrain the window layout to WxH px (e.g. 2400x1400 on a "
                 "2560x1600 screen keeps right/bottom margins); empty = full "
                 "virtual screen",
        )
        parser.add_argument(
            "--box-delta",
            type=int,
            default=8,
            help="delta fallback: per-channel RGB change threshold between "
                 "the two frames (default 8)",
        )
        parser.add_argument(
            "--box-tolerance",
            type=float,
            default=5.0,
            help="strict overlay formula: per-channel match slack (default 5)",
        )
        parser.add_argument(
            "--box-area-ratio",
            type=float,
            default=0.20,
            help="whole-page drop: connected region area fraction of the "
                 "frame above which the box is discarded (default 0.20)",
        )
        parser.add_argument(
            "--no-fixed-window",
            action="store_true",
            help="do not pin Photos windows to fixed screen tiles",
        )
        args = parser.parse_args(raw_args)
        if args.fast:
            # ``--box-delta`` is accepted for CLI compatibility but has no
            # effect any more: detection is a single strict-formula path.
            if args.box_delta != 8:
                print(
                    "warning: --box-delta is ignored "
                    "(strict formula only, no delta fallback)",
                    file=sys.stderr,
                )
            result = capture_photos_fast_batch(
                args.images,
                args.output_dir,
                pages=_parse_pages(args.pages),
                timeout=args.timeout,
                resume=args.resume,
                paddle_device=args.paddle_device,
                max_attempts=args.max_attempts,
                ask_fallback=not args.no_ask_fallback,
                parallel=args.parallel,
                frames_only=args.frames_only,
                ocr_engine=args.ocr_engine,
                fixed_window=not args.no_fixed_window,
                display_scale=args.display_scale,
                row_gap_px=args.layout_row_gap,
                bottom_shift_up=args.layout_bottom_shift_up,
                layout_region=args.layout_region,
                box_delta=args.box_delta,
                box_tolerance=args.box_tolerance,
                box_area_ratio=args.box_area_ratio,
            )
        else:
            result = capture_photos_batch(
                args.images,
                args.output_dir,
                pages=_parse_pages(args.pages),
                timeout=args.timeout,
                resume=args.resume,
            )
        print(Path(args.output_dir).resolve() / "batch.capture.json")
        return 0 if result["complete"] else 2

    parser = argparse.ArgumentParser(description="Capture column OCR from Windows Photos")
    parser.add_argument("image")
    parser.add_argument("-o", "--output")
    parser.add_argument("--debug-dir")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="mask islands and OCR them with PaddleOCR instead of click-copy",
    )
    parser.add_argument(
        "--screenshot",
        help="save the highlighted OCR overlay screenshot to this PNG path",
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="do not fall back to Photos click-copy for unrecognized columns",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "windows-ocr", "paddle"],
        default="auto",
        help="masked-page OCR engine (default auto: Windows OCR then Paddle fallback; forced engines fail loudly)",
    )
    parser.add_argument("--paddle-device", default="cpu")
    args = parser.parse_args(raw_args)
    if args.fast:
        result = capture_photos_fast(
            args.image,
            args.timeout,
            args.debug_dir,
            args.screenshot,
            args.skip_fallback,
            args.paddle_device,
            args.ocr_engine,
        )
    else:
        result = capture_photos_columns(args.image, args.timeout, args.debug_dir)
    output = args.output or str(Path(args.image).with_suffix(".photos-ocr.json"))
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
