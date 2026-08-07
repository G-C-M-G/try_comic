"""One-click manga translation.

Stage 1 (capture-only, default): parallel Photos frames + blue-box capture
with fixed tiled windows; every page gets a ``{page}.bluebox_check.png``
annotation (islands gold, blue boxes blue) so the user can inspect the OCR
units before anything else runs.

Stage 2 (--process): background batch -- diff+blue-box islands, manga-ocr
per blue box (group join), Sakura translation (guarded), erase + vertical/
horizontal typeset, merged PDF.

--auto runs stage 2 immediately after stage 1.

Architecture contract (see docs/ONE_CLICK_ARCHITECTURE.md):

* The Photos blue box is the atomic OCR unit and cannot be split.
* manga-ocr reads ONLY blue boxes, cropped from the whole-page masked image
  (``bluebox_masked.png``: blue boxes expanded by 6 px keep their pixels,
  everything else is filled with ``--bluebox-fill``).  Islands WITHOUT blue
  boxes are never OCRed -- the flat fill area makes manga-ocr hallucinate
  (observed: pure-black crops read as "それでも、").  No-box islands stay
  untouched: no read, no translation, no erasure, no typesetting.
* Erasure erases only the boxed islands (a dataset subset is passed to
  ``erase_island_image``); unboxed islands keep their original pixels.
  Each boxed island's canonical "one box" (its blue-box group expanded
  6 px) replaces the island bbox for erasure and typesetting: the whole
  box is erased, so rendered text never lands on un-erased art.
* Typesetting (island_typeset.py): single layout path -- a centred block
  inside the island bbox (the canonical box), ratio A_text / A_island in
  [0.6, 1.1].  The block must fit entirely inside the box; the font size
  descends below 28px (down to a 12px absolute floor) instead of being
  truncated.  Glyph size <= the PADDED box dimensions: vertical size <=
  narrowest pad-6 box width, horizontal size <= shortest pad-6 box
  height.  Vertical columns are top-aligned, horizontal rows
  left-aligned; the block is centred on the island bbox centre.
* A blue box read empty twice is treated as a false selection and dropped.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

IMAGES = Path(r"D:\workplace\opencode\try_comic\漫画翻译\原文\甲书")
BLUEBOX_FILL = (0, 0, 0)  # fill colour outside blue boxes (--bluebox-fill)


def _load_api_config() -> dict:
    """Local API keys/config (config/api_keys.json, git-ignored)."""
    path = ROOT / "config" / "api_keys.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _parse_pages(value: str | None) -> set[str] | None:
    if not value:
        return None
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


def stage1_capture(
    images: Path, out_root: Path, pages: set[str] | None, parallel: int,
    display_scale: int, row_gap_px: int | None, bottom_shift_up: int,
    layout_region: str = "", box_delta: int = 8, box_tolerance: float = 5.0,
    box_area_ratio: float = 0.20,
) -> bool:
    """Parallel frames-only capture + blue-box check images."""
    capture_dir = out_root / "capture"
    check_dir = out_root / "check"
    capture_dir.mkdir(parents=True, exist_ok=True)
    check_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ROOT / "venv" / "Scripts" / "python.exe"),
        str(ROOT / "photos_capture.py"),
        "batch", "--fast",
        "--images", str(images),
        "--output-dir", str(capture_dir),
        "--frames-only",
        "--parallel", str(parallel),
        "--display-scale", str(display_scale),
        "--resume",
        "--box-delta", str(box_delta),
        "--box-tolerance", str(box_tolerance),
        "--box-area-ratio", str(box_area_ratio),
    ]
    if row_gap_px is not None:
        cmd += ["--layout-row-gap", str(row_gap_px)]
    if bottom_shift_up:
        cmd += ["--layout-bottom-shift-up", str(bottom_shift_up)]
    if layout_region:
        cmd += ["--layout-region", layout_region]
    if pages:
        cmd += ["--pages", ",".join(sorted(pages))]
    from manga_localizer.progress import ProgressFile

    progress = ProgressFile(out_root)
    progress.set_stage("capture", total=len(pages) if pages else 0)
    progress.log(f"capturing {len(pages) if pages else 'all'} pages (parallel={parallel})")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    captured = 0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(line)
        progress.log(line)
        if "capture" in line and "page" in line.lower():
            captured += 1
            progress.progress(min(captured, len(pages) or 1), item=line[:60])
    code = proc.wait()
    if code != 0:
        progress.fail(f"photos_capture exited {code}")
        return False
    progress.progress(len(pages) or 0, item="capture done")

    # check images: blue boxes (blue, drawn padded like OCR uses them) +
    # page_scale (red) per page.  No highlight-island detection: only the
    # Photos blue boxes matter.
    pad = 6
    for page_dir in sorted(capture_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        manifest_path = page_dir / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        capture = manifest.get("capture", {})
        if not capture.get("complete"):
            continue
        source = manifest.get("source", {})
        image_size = (int(source["width"]), int(source["height"]))
        image_path = Path(source["path"])
        if not image_path.is_file():
            continue
        selection_blocks = capture.get("selection_blocks") or []
        # Check image = the original page with blue boxes drawn directly on
        # top (no dimming/filling): review shows true pixel brightness.
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for block in selection_blocks:
            x1, y1, x2, y2 = block["image_bbox"]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image_size[0], x2 + pad)
            y2 = min(image_size[1], y2 + pad)
            draw.rectangle([x1, y1, x2, y2], outline="#4363d8", width=3)
            draw.text((x1, min(image.height - 14, y2 + 2)), block["id"],
                      fill="#4363d8")
        image.save(check_dir / f"{page_dir.name}.bluebox_check.png")
    print(f"blue-box check images -> {check_dir}")
    progress.done(ok=True)
    return True

    # Report pages that will not be processed because no blue boxes (or no
    # usable capture) were found: written to a report file, printed and
    # surfaced with a message box.
    report_rows: list[tuple[str, str]] = []
    for page_dir in sorted(capture_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        manifest_path = page_dir / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        capture = manifest.get("capture", {})
        if capture.get("complete") and not (capture.get("selection_blocks") or []):
            reason = capture.get("selection_boxes_status") or "no blue boxes detected"
            report_rows.append((page_dir.name, reason))
        elif capture.get("black_page"):
            report_rows.append((page_dir.name, "black page (no content)"))
        elif capture.get("no_text"):
            report_rows.append((page_dir.name, "no text detected"))
        elif capture.get("fallback_manga"):
            report_rows.append((page_dir.name, "capture failed (manga fallback)"))
        elif not capture.get("complete"):
            report_rows.append(
                (page_dir.name, capture.get("error") or "capture failed")
            )
    if report_rows:
        report_path = out_root / "BLUEBOX_REPORT.md"
        lines = ["# 蓝框缺失报告", ""]
        lines.append("以下页面未识别到蓝框（或采集未完成），将保持原样不处理：")
        lines.append("")
        for page, reason in report_rows:
            lines.append(f"- 第 {page} 页：{reason}")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = "、".join(page for page, _ in report_rows)
        print("")
        print("=" * 60)
        print(f"!! 蓝框缺失: {len(report_rows)} 页 -> {report_path}")
        for page, reason in report_rows:
            print(f"   - 第 {page} 页: {reason}")
        print("=" * 60)
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                f"以下页面未识别到蓝框，将保持原样不处理：\n{summary}\n\n"
                f"详见 {report_path}",
                "蓝框缺失报告",
                0x00000040 | 0x00001000,  # MB_ICONINFORMATION | MB_SYSTEMMODAL
            )
        except Exception:
            pass
    else:
        print("all captured pages have blue boxes")
    return True


def build_bluebox_masked(
    image: np.ndarray, blue_boxes: list[dict], pad: int = 6, fill=(0, 0, 0)
) -> np.ndarray:
    """One whole-page masked image: every blue box (page_scale excluded)
    expanded by *pad* px keeps its original pixels, everything else is
    filled with *fill* (default black, customizable)."""
    masked = np.full(image.shape, fill, dtype=np.uint8)
    for box in blue_boxes:
        if box.get("page_scale"):
            continue
        x1, y1, x2, y2 = box["image_bbox"]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return masked


def blue_boxes_to_lines(
    selection_blocks_path, pad: int = 6, exclude: set[str] | None = None
) -> list:
    """Blue boxes (page_scale excluded) expanded by *pad* px as pseudo
    OcrLines -- the typeset column positions follow these columns.  ``exclude``
    drops specific box ids (false selections dropped by the OCR passes)."""
    from manga_localizer.island_dry_run import OcrLine

    payload = json.loads(Path(selection_blocks_path).read_text(encoding="utf-8-sig"))
    excluded = exclude or set()
    lines = []
    for box in payload.get("selection_blocks") or []:
        if box.get("page_scale") or box.get("id") in excluded:
            continue
        x1, y1, x2, y2 = box["image_bbox"]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = x2 + pad
        y2 = y2 + pad
        lines.append(OcrLine(
            polygon=((float(x1), float(y1)), (float(x2), float(y1)),
                     (float(x2), float(y2)), (float(x1), float(y2))),
            text="",
        ))
    return lines


def _load_readings_cache(out_root: Path) -> dict:
    """Load the persisted OCR readings cache (``{crop_id: text}``).

    Only non-empty readings are stored, so boxes that read empty are
    re-OCRed on every run (the two-pass false-selection check keeps its
    behaviour).  A missing or malformed cache file yields ``{}``.
    """
    path = out_root / "ocr_readings.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(value, str) and value.strip()
    }


def _store_readings_cache(out_root: Path, readings: dict) -> None:
    """Merge ``readings`` into the persisted OCR cache (atomic replace).

    The write merges with the on-disk state, so parallel segment
    processes do not lose each other's entries (the last writer wins only
    for entries written concurrently between load and store).
    """
    merged = _load_readings_cache(out_root)
    merged.update(
        {key: text for key, text in readings.items() if str(text or "").strip()}
    )
    path = out_root / "ocr_readings.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _select_crop_misses(crops: list[dict], cached: dict) -> list[dict]:
    """Crops without a cached non-empty reading (these need OCR)."""
    return [
        crop for crop in crops
        if not str((cached.get(crop["id"]) or "")).strip()
    ]


def run_manga(
    crops: list[dict],
    out_root: Path,
    progress: "ProgressFile | None" = None,
    *,
    ai_ocr_url: str = "",
    ai_ocr_model: str = "",
    ai_ocr_api_key: str = "",
) -> dict:
    """OCR every crop: AI vision model when configured, else manga-ocr.

    The AI path (OpenAI-compatible vision, e.g. GLM-4V / Qwen-VL) posts
    each crop as an image message; without a configured vision endpoint
    manga-ocr runs in one subprocess (model loads once).

    OCR readings are persisted to ``out_root/ocr_readings.json`` (non-empty
    only): crops with a cached reading skip the OCR pass entirely, so
    re-running stage 2 (re-grouping, re-typesetting, PDF merge) costs no
    re-recognition.  Boxes that read empty are never cached and are OCRed
    again on every run (the two-pass false-selection check).
    """
    cached = _load_readings_cache(out_root)
    misses = _select_crop_misses(crops, cached)
    if misses and (ai_ocr_url and ai_ocr_model and ai_ocr_api_key):
        from manga_localizer.ai_ocr import ai_ocr_readings

        readings = ai_ocr_readings(
            misses, ai_ocr_url, ai_ocr_model, ai_ocr_api_key,
            progress=progress,
        )
    elif misses:
        readings = _run_manga_subprocess(misses, out_root, progress)
    else:
        readings = {}
    _store_readings_cache(out_root, readings)
    result = {}
    for crop in crops:
        result[crop["id"]] = str(
            readings.get(crop["id"]) or cached.get(crop["id"]) or ""
        )
    return result


def _run_manga_subprocess(
    crops: list[dict],
    out_root: Path,
    progress: "ProgressFile | None" = None,
) -> dict:
    script = """\
import json, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image
from manga_ocr import MangaOcr
crops = json.load(open(sys.argv[1], encoding="utf-8"))
mocr = MangaOcr()
for entry in crops:
    source, bbox = entry["source"], tuple(entry["bbox"])
    image = None
    for attempt in range(3):
        try:
            image = Image.open(source).convert("RGB")
            break
        except OSError as error:
            if attempt == 2:
                raise OSError(f"cannot read crop source {source!r}: {error}")
            time.sleep(0.3)
    crop = image.crop(bbox)
    text = str(mocr(crop) or "").strip()
    print(json.dumps({"id": entry["id"], "text": text}, ensure_ascii=False))
    sys.stdout.flush()
"""
    crops_path = out_root / f"crops.{os.getpid()}.json"
    crops_path.write_text(json.dumps(crops, ensure_ascii=False), encoding="utf-8")
    python = ROOT / "venv" / "Scripts" / "python.exe"
    env = dict(__import__("os").environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not env.get("MANGA_OCR_GPU"):
        # manga-ocr runs on the CPU by default: the Sakura translation
        # model owns the GPU, and three parallel OCR subprocesses on CUDA
        # exhaust its VRAM (llama_decode returns -1 / logits assert).
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.Popen(
        [str(python), "-c", script, str(crops_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=False, env=env,
    )
    assert proc.stdout is not None
    readings: dict[str, str] = {}
    done = 0
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        readings[payload["id"]] = str(payload.get("text") or "")
        done += 1
        if progress is not None:
            progress.sub_progress(done, len(crops), payload["id"])
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    code = proc.wait(timeout=3600)
    if code != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"manga-ocr failed: {detail}")
    return readings


def _source_image_for(images: Path, page: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images / f"{page}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _group_boxes_by_overlap(
    boxes: Sequence[dict],
    *,
    pad: int = 6,
    gap_ratio: float = 0.20,
    overlap_ratio: float = 0.40,
) -> list[tuple[tuple[int, int, int, int], list[str]]]:
    """Group blue boxes (image-space) into OCR recognition units.

    A box is vertical when its height >= width, horizontal otherwise.
    Boxes NEVER merge along the reading direction (Photos' blue boxes are
    the trusted atomic units); they only merge side-by-side on the
    PERPENDICULAR axis when they are close AND aligned.  All distances
    are measured on the PADDED boxes (each blue box expanded by *pad* px,
    the same expansion the canonical "one box" uses):

    * two VERTICAL boxes merge when the padded horizontal gap is below
      20% of the smaller padded box's width AND their padded vertical
      ranges overlap by more than 40% of the LARGER padded box's height;
    * two HORIZONTAL boxes merge when the padded vertical gap is below
      20% of the smaller padded box's height AND their padded horizontal
      ranges overlap by more than 40% of the LARGER padded box's width;
    * mixed-orientation pairs never merge.

    Returns ``[(group_bbox, [member_ids])]`` sorted top-to-bottom,
    right-to-left.  ``boxes`` must carry ``id`` and ``image_bbox``.
    """
    atomic = [
        box for box in boxes
        if box.get("id") and not box.get("page_scale")
        and isinstance(box.get("image_bbox"), (list, tuple)) and len(box["image_bbox"]) == 4
    ]
    ordered = sorted(
        atomic, key=lambda box: (box["image_bbox"][1], -box["image_bbox"][2])
    )
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    padded: dict[int, tuple[int, int, int, int]] = {}
    for index, box in enumerate(ordered):
        x1, y1, x2, y2 = box["image_bbox"]
        padded[index] = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    for i in range(len(ordered)):
        a = ordered[i]
        ax1, ay1, ax2, ay2 = a["image_bbox"]
        a_vertical = (ay2 - ay1) >= (ax2 - ax1)
        pa = padded[i]
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            bx1, by1, bx2, by2 = b["image_bbox"]
            b_vertical = (by2 - by1) >= (bx2 - bx1)
            if a_vertical != b_vertical:
                continue
            pb = padded[j]
            min_pw = min(pa[2] - pa[0], pb[2] - pb[0])
            min_ph = min(pa[3] - pa[1], pb[3] - pb[1])
            if min_pw <= 0 or min_ph <= 0:
                continue
            x_gap = max(pa[0], pb[0]) - min(pa[2], pb[2])
            y_gap = max(pa[1], pb[1]) - min(pa[3], pb[3])
            x_overlap = min(pa[2], pb[2]) - max(pa[0], pb[0])
            y_overlap = min(pa[3], pb[3]) - max(pa[1], pb[1])
            if a_vertical:
                if x_gap < gap_ratio * min_pw and y_overlap > overlap_ratio * max(
                    pa[3] - pa[1], pb[3] - pb[1]
                ):
                    union(i, j)
            else:
                if y_gap < gap_ratio * min_ph and x_overlap > overlap_ratio * max(
                    pa[2] - pa[0], pb[2] - pb[0]
                ):
                    union(i, j)

    members_by_root: dict[int, list[str]] = {}
    for index, box in enumerate(ordered):
        members_by_root.setdefault(find(index), []).append(box["id"])
    result = []
    for root, member_ids in members_by_root.items():
        members = [ordered[i] for i in range(len(ordered)) if find(i) == root]
        x1 = min(box["image_bbox"][0] for box in members)
        y1 = min(box["image_bbox"][1] for box in members)
        x2 = max(box["image_bbox"][2] for box in members)
        y2 = max(box["image_bbox"][3] for box in members)
        result.append(((x1, y1, x2, y2), member_ids))
    return sorted(result, key=lambda item: (item[0][1], -item[0][2]))


def build_bluebox_dataset(
    manifest_path, page_dir, source_path, min_island_area: float = 800.0, pad: int = 2
) -> list[dict]:
    """islands.json purely from the Photos blue boxes (original image space).

    No highlight-island detection: the atomic blue boxes are re-grouped by
    the TWO-AXIS 60% overlap rule (``_group_boxes_by_overlap``) -- every
    group is one island, padded ``pad`` px, and its own independent bbox.
    Raises ValueError when the page has no usable blue boxes -- the caller
    keeps the page untouched then.  Also writes selection_blocks.json
    (typeset column positions + the re-grouped groups) and a copy of the
    original image.
    """
    import shutil

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    capture = payload.get("capture", {})
    blocks = capture.get("selection_blocks") or []
    source = payload.get("source", {})
    image_w = int(source["width"])
    image_h = int(source["height"])
    atomic_ids = {item["id"] for item in blocks if not item.get("page_scale")}
    grouped = _group_boxes_by_overlap(blocks)
    groups = [
        {
            "id": f"selg_{index:03d}",
            "image_bbox": [int(value) for value in bbox],
            "members": member_ids,
        }
        for index, (bbox, member_ids) in enumerate(grouped, 1)
        if member_ids
    ]
    islands: list[dict] = []
    for group in groups:
        members = [member for member in group["members"] if member in atomic_ids]
        if not members:
            continue
        box = group.get("image_bbox")
        if not box or len(box) != 4:
            continue
        x1 = max(0, int(box[0]) - pad)
        y1 = max(0, int(box[1]) - pad)
        x2 = min(image_w, int(box[2]) + pad)
        y2 = min(image_h, int(box[3]) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        area = float((x2 - x1) * (y2 - y1))
        if area < min_island_area:
            continue
        islands.append({
            "id": f"island_{len(islands) + 1:03d}",
            "display_bbox": [x1, y1, x2, y2],
            "display_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "display_area": int(area),
            "image_bbox": [x1, y1, x2, y2],
            "image_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "image_area": round(area, 1),
            "from_selection_group": True,
        })
    if not islands:
        raise ValueError(
            f"page {Path(page_dir).name}: no usable blue boxes "
            "(nothing to process, page kept original)"
        )
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "islands.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "page": Path(page_dir).name,
                "image_size": [image_w, image_h],
                "contours": False,
                "fill_mode": "bbox",
                "min_island_area": min_island_area,
                "blue_boxes_only": True,
                "islands": islands,
                "dropped_tiny": [],
                "splits": [],
                "timing": {
                    "blue_boxes_only": True,
                    "island_count": len(islands),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (page_dir / "selection_blocks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "page": Path(page_dir).name,
                "image_size": [image_w, image_h],
                "selection_blocks": blocks,
                "selection_groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(source_path, page_dir / "original.jpg")
    return islands


def build_page_texts(units, readings, dropped_boxes):
    """Join blue-box manga readings into per-island sentence texts.

    Dropped boxes (false selections) never join a sentence and never set a
    direction.  Islands whose members all read empty get an empty text so
    the caller keeps them untouched (no erasure, no typesetting).
    Returns ``(page_texts, page_directions)``.
    """
    page_texts: dict[str, dict[str, str]] = {}
    page_directions: dict[str, dict[str, str]] = {}
    for page, unit in units.items():
        by_island = unit["islands"]
        box_by_id = unit["blocks"]
        dropped = dropped_boxes.get(page, set())
        texts, directions = {}, {}
        for iid, island in by_island.items():
            ix1, iy1, ix2, iy2 = island["image_bbox"]
            joined = []
            horizontal = False
            for gid, group in unit["groups"].items():
                members = []
                for mid in group["members"]:
                    box = box_by_id.get(mid)
                    if box is None or mid in dropped:
                        continue
                    bx1, by1, bx2, by2 = box["image_bbox"]
                    if ix1 <= (bx1 + bx2) // 2 <= ix2 and iy1 <= (by1 + by2) // 2 <= iy2:
                        members.append((box, readings.get(f"{page}:box:{mid}", "")))
                if members:
                    horiz = sum(
                        1 for box, _ in members
                        if (box["image_bbox"][2] - box["image_bbox"][0])
                        >= (box["image_bbox"][3] - box["image_bbox"][1])
                    )
                    if horiz * 2 >= len(members):
                        horizontal = True
                        members.sort(key=lambda item: (item[0]["image_bbox"][1], item[0]["image_bbox"][0]))
                    else:
                        members.sort(key=lambda item: -item[0]["image_bbox"][0])
                    joined.append("".join(text for _, text in members))
            texts[iid] = "".join(joined) if joined else ""
            directions[iid] = "horizontal" if horizontal else "vertical"
        page_texts[page] = texts
        page_directions[page] = directions
    return page_texts, page_directions


def stage2_process(
    images: Path,
    out_root: Path,
    pages: list[str],
    *,
    ai_url: str = "https://api.deepseek.com/v1",
    ai_model: str = "deepseek-chat",
    ai_api_key: str = "",
    ai_ocr_url: str = "",
    ai_ocr_model: str = "",
    ai_ocr_api_key: str = "",
    resume: bool = False,
    no_pdf: bool = False,
    min_font_size: int | None = None,
    font_name: str = "",
    stroke_divisor: int | None = None,
    row_gap_divisor: int | None = None,
    column_gap_divisor: int | None = None,
    tracking_divisor: int | None = None,
    text_color: str = "",
    stroke_color: str = "",
    background_color: str = "",
    max_workers: int = 1,
    progress: "ProgressFile | None" = None,
) -> bool:
    """Background: blue boxes -> manga -> translate -> typeset -> PDF.

    Translation prefers the AI (OpenAI-compatible, e.g. DeepSeek) whole-page
    batch; islands whose numbered line is missing/malformed fall back to
    the local Sakura endpoint.  Without an API key everything goes through
    local Sakura.  ``ai_ocr_*`` enable an OpenAI-compatible vision model to
    read the blue-box crops instead of manga-ocr.  ``resume`` skips pages
    whose typeset output already exists.  ``no_pdf`` skips the final PDF
    merge (typeset images only).  ``progress`` publishes run.json.
    """
    from island_extract import build_fallback_dataset
    from manga_localizer.island_contract import load_island_dataset
    from manga_localizer.island_translate import (
        IslandTranslateConfig,
        IslandTranslationCache,
        PageBatchCache,
        translate_island_page,
        translate_page_batch,
    )
    from manga_localizer.translator import load_glossary

    if progress is None:
        from manga_localizer.progress import ProgressFile

        progress = ProgressFile(out_root)

    capture_dir = out_root / "capture"
    dataset_dir = out_root / "dataset"
    typeset_dir = out_root / "typeset"
    typeset_dir.mkdir(parents=True, exist_ok=True)

    fallback = _fallback_pages_for(capture_dir, pages)
    # Islands come ONLY from the Photos blue boxes (no highlight-island
    # detection): one island per selection group, original image space.
    progress.set_stage("dataset", total=len(pages))
    for index, page in enumerate(pages):
        progress.progress(index, item=f"page {page}")
        page_dir = dataset_dir / page
        if (page_dir / "islands.json").is_file():
            continue
        source = _source_image_for(images, page)
        if source is None:
            continue
        if page in fallback:
            with Image.open(source) as probe:
                image_size = probe.size
            build_fallback_dataset(source, page_dir, image_size)
            continue
        manifest_path = capture_dir / page / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            build_bluebox_dataset(manifest_path, page_dir, source)
        except ValueError as error:
            # No usable blue boxes: nothing to process, the page stays
            # original (kept out of the OCR/translate/typeset loops).
            print(f"{page}: {error}")

    active_pages = [
        page for page in pages
        if (dataset_dir / page / "islands.json").is_file()
    ]

    # manga reads: blue boxes only, from the blue-box masked image so
    # neighbour text can never leak into a crop
    crops, units = [], {}
    for page in active_pages:
        page_dir = dataset_dir / page
        islands = json.loads((page_dir / "islands.json").read_text(encoding="utf-8-sig"))
        sel_path = page_dir / "selection_blocks.json"
        selection = (
            json.loads(sel_path.read_text(encoding="utf-8-sig"))
            if sel_path.is_file()
            else None
        )
        blocks = (selection or {}).get("selection_blocks") or []
        groups = (selection or {}).get("selection_groups") or []
        source = str(page_dir / "original.jpg")
        box_by_id = {b["id"]: b for b in blocks}
        # whole-page masked image: blue boxes (pad 2) keep pixels, rest fill
        if blocks:
            original = np.array(Image.open(source).convert("RGB"))
            masked = build_bluebox_masked(
                original, blocks, pad=6,
                fill=tuple(int(v) for v in BLUEBOX_FILL),
            )
            masked_path = page_dir / "bluebox_masked.png"
            Image.fromarray(masked).save(masked_path)
            source = str(masked_path)
        for block in blocks:
            if block.get("page_scale"):
                continue
            crops.append({
                "id": f"{page}:box:{block['id']}",
                "source": source,
                "bbox": [
                    max(0, block["image_bbox"][0] - 6),
                    max(0, block["image_bbox"][1] - 6),
                    block["image_bbox"][2] + 6,
                    block["image_bbox"][3] + 6,
                ],
            })
        # islands without blue boxes are never OCRed: outside the blue
        # boxes everything stays original (no read, no translation, no
        # erasure, no typesetting).  Reading the fill area would only
        # invite manga-ocr hallucinations.
        units[page] = {"islands": {i["id"]: i for i in islands["islands"]},
                       "blocks": box_by_id, "groups": {g["id"]: g for g in groups}}
    progress.set_stage("ocr", total=len(crops))
    progress.sub_progress(0, len(crops), "manga-ocr")
    readings = run_manga(
        crops, out_root, progress=progress,
        ai_ocr_url=ai_ocr_url, ai_ocr_model=ai_ocr_model,
        ai_ocr_api_key=ai_ocr_api_key,
    )

    # Second pass: reread the boxes manga-ocr read empty once.  A box that
    # stays empty is a false selection -- dropped everywhere (never joined
    # into a sentence, never erased, never typeset).
    dropped_boxes: dict[str, set[str]] = {page: set() for page in active_pages}
    rereads = []
    for crop in crops:
        page = crop["id"].split(":", 1)[0]
        if readings.get(crop["id"]):
            continue
        rereads.append(
            {
                "id": f"{page}:box:{crop['id'].split(':box:', 1)[1]}:reread",
                "source": crop["source"],
                "bbox": crop["bbox"],
            }
        )
    if rereads:
        progress.log(f"rereading {len(rereads)} empty boxes")
        reread_readings = run_manga(
            rereads, out_root, progress=progress,
            ai_ocr_url=ai_ocr_url, ai_ocr_model=ai_ocr_model,
            ai_ocr_api_key=ai_ocr_api_key,
        )
        for crop in rereads:
            page, _, rest = crop["id"].partition(":box:")
            box_id = rest.removesuffix(":reread")
            key = f"{page}:box:{box_id}"
            reread_text = reread_readings.get(crop["id"], "")
            if reread_text:
                readings[key] = reread_text
            else:
                dropped_boxes[page].add(box_id)

    page_texts, page_directions = build_page_texts(units, readings, dropped_boxes)

    glossary = load_glossary(ROOT / "config" / "glossary.json")
    cache = IslandTranslationCache(out_root / "translation_cache")
    batch_cache = PageBatchCache(out_root / "translation_batch_cache")
    parsed_text = _parse_color(text_color)
    parsed_stroke = _parse_color(stroke_color)
    parsed_background = _parse_color(background_color)

    def _translate_one_page(page: str) -> dict[str, str]:
        """Whole-page translate: AI batch preferred, local Sakura fallback."""
        dataset = load_island_dataset(dataset_dir / page / "islands.json")
        texts = page_texts[page]
        page_translations: dict[str, str] = {}
        fallback_texts: dict[str, str] = {}
        if ai_api_key and texts:
            try:
                batch = translate_page_batch(
                    dataset,
                    texts,
                    IslandTranslateConfig(
                        precision="online",
                        online_base_url=ai_url,
                        online_model=ai_model,
                        online_api_key=ai_api_key,
                        glossary_path=str(ROOT / "config" / "glossary.json"),
                    ),
                    glossary=glossary,
                    cache=batch_cache,
                )
                page_translations.update(batch.translations)
                fallback_texts = {island_id: texts[island_id] for island_id in batch.failures}
                if batch.parse_errors:
                    print(f"  {page}: AI parse errors={len(batch.parse_errors)} fallback={len(batch.failures)}")
            except Exception as error:  # noqa: BLE001 - AI failure degrades to local
                print(f"  {page}: AI batch failed ({error}), local fallback")
                fallback_texts = dict(texts)
        else:
            fallback_texts = dict(texts)
        if fallback_texts:
            result = translate_island_page(
                dataset,
                fallback_texts,
                IslandTranslateConfig(
                    precision="local",
                    sakura_base_url="http://127.0.0.1:8080/v1",
                    sakura_model="sukinishiro",
                    glossary_path=str(ROOT / "config" / "glossary.json"),
                ),
                glossary=glossary,
                cache=cache,
            )
            for unit in result.units:
                page_translations[unit.island_id] = unit.translated_text or ""
        return page_translations

    def _worker(page: str) -> bool:
        """One page: translate then typeset.  Returns True on failure."""
        try:
            translations_page = _translate_one_page(page)
        except Exception as error:  # noqa: BLE001 - per-page isolation
            print(f"!! translate {page} FAILED: {error}")
            progress.log(f"translate {page} failed: {error}")
            return True
        out_png = typeset_dir / f"{page}.typeset.png"
        if resume and out_png.is_file():
            progress.log(f"resume: skip {page} (typeset exists)")
            return False
        try:
            erase_and_typeset_page_with(
                dataset_dir, typeset_dir, page, translations_page,
                page_directions.get(page), dropped_boxes.get(page, set()),
                min_font_size=min_font_size, font_name=font_name,
                stroke_divisor=stroke_divisor, row_gap_divisor=row_gap_divisor,
                column_gap_divisor=column_gap_divisor,
                tracking_divisor=tracking_divisor,
                text_color=parsed_text, stroke_color=parsed_stroke,
                background_color=parsed_background,
            )
        except Exception as error:  # noqa: BLE001 - one bad page must not
            # kill the batch; the page keeps its original in the PDF.
            print(f"!! typeset {page} FAILED: {error}")
            progress.log(f"typeset {page} failed: {error}")
            return True
        print(f"typeset {page} -> {typeset_dir / f'{page}.typeset.png'}")
        return False

    import threading
    from concurrent.futures import ThreadPoolExecutor

    progress.set_stage("typeset", total=len(active_pages))
    failed_pages: list[str] = []
    state_lock = threading.Lock()
    done_count = 0

    def _on_page_done(page: str, failed: bool) -> None:
        nonlocal done_count
        with state_lock:
            done_count += 1
            if failed:
                failed_pages.append(page)
        progress.progress(done_count, item=f"page {page}")

    workers = max(1, int(max_workers))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker, page): page for page in active_pages
            }
            for future in futures:
                page = futures[future]
                try:
                    failed = future.result()
                except Exception as error:  # noqa: BLE001
                    failed = True
                    print(f"!! page {page} worker error: {error}")
                _on_page_done(page, failed)
    else:
        for page in active_pages:
            _on_page_done(page, _worker(page))

    # Pages without blue boxes stay untouched: the PDF keeps them as the
    # original image.
    untouched = [page for page in pages if page not in active_pages]
    if untouched:
        print("")
        print("=" * 60)
        print(f"!! 以下页面无蓝框/未处理，PDF 中保持原图: {', '.join(untouched)}")
        print("=" * 60)
    for page in pages:
        if page in active_pages:
            continue
        source = dataset_dir / page / "original.jpg"
        if not source.is_file():
            source = _source_image_for(images, page)
        if source is not None:
            import shutil

            shutil.copyfile(source, typeset_dir / f"{page}.typeset.png")

    typeset_files = [
        typeset_dir / f"{page}.typeset.png"
        for page in pages
        if (typeset_dir / f"{page}.typeset.png").is_file()
    ]
    if not typeset_files:
        print("no pages to assemble into a PDF")
        progress.done(ok=not failed_pages)
        return True
    if no_pdf:
        print(f"--no-pdf: typeset images only ({len(typeset_files)} pages, PDF skipped)")
        if failed_pages:
            print(f"!! 失败页（嵌字图缺失）: {', '.join(failed_pages)}")
        progress.done(ok=not failed_pages)
        return True
    progress.set_stage("pdf", total=1)
    progress.progress(0, item="merge PDF")
    pdf_path = out_root / f"{images.name}.pdf"
    try:
        import img2pdf

        pdf_path.write_bytes(img2pdf.convert([str(path) for path in typeset_files]))
    except Exception as error:  # noqa: BLE001 - Pillow PDF save is broken on
        # this Pillow build (no JPEG encoder); img2pdf is the primary path.
        progress.fail(f"PDF merge failed: {error}")
        raise RuntimeError(f"PDF merge failed: {error}") from error
    progress.progress(1, item=str(pdf_path))
    print(f"-> {pdf_path} ({len(typeset_files)} pages)")
    if failed_pages:
        print(f"!! 失败页（PDF 中保持原图）: {', '.join(failed_pages)}")
    progress.done(ok=not failed_pages)
    return True


def _fallback_pages_for(capture_dir: Path, pages: list[str]) -> set[str]:
    fallback = set()
    for page in pages:
        manifest_path = capture_dir / page / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if manifest.get("capture", {}).get("fallback_manga"):
            fallback.add(page)
    return fallback


def _review_required_pages(capture_dir: Path, pages: list[str]) -> list[str]:
    """Pages whose capture produced no blue pixels after three Ctrl+A
    attempts.  Hard rule: these pages are NOT processed downstream until
    the user reviews them and confirms (--confirm-no-boxes) they really
    have no selection boxes."""
    review = []
    for page in pages:
        manifest_path = capture_dir / page / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        capture = manifest.get("capture", {})
        if capture.get("selection_review_required") and not capture.get(
            "selection_review_confirmed"
        ):
            review.append(page)
    return review


def _confirm_no_boxes(capture_dir: Path, pages: list[str]) -> None:
    """Mark pages as user-reviewed: they genuinely have no blue boxes and
    may be closed (skipped downstream without further warnings)."""
    for page in pages:
        manifest_path = capture_dir / page / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        capture = manifest.setdefault("capture", {})
        capture["selection_review_confirmed"] = True
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _skip_pages(capture_dir: Path, pages: list[str]) -> set[str]:
    """Pages with no usable capture (black / no-text / failed) are skipped
    silently; fallback_manga pages stay active.  Review-required pages are
    skipped but listed loudly for the user to audit."""
    skipped = set()
    for page in pages:
        manifest_path = capture_dir / page / "capture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        capture = manifest.get("capture", {})
        if capture.get("fallback_manga"):
            continue
        if not capture.get("complete"):
            skipped.add(page)
        elif (
            capture.get("selection_review_required")
            and not capture.get("selection_review_confirmed")
        ):
            skipped.add(page)
    return skipped


def build_union_overrides(dataset, boxes, pad: int = 6) -> dict:
    """Per-island blue-box union masks (island-local coords).

    Each island's mask is the union of its blue boxes expanded by *pad* px
    (clamped to the island bbox); ``area`` is the mask pixel count.  Glyph
    size caps follow the PADDED box dimensions: vertical font size
    (glyph width) <= narrowest padded box width, horizontal font size
    (glyph height) <= shortest padded box height.

    ``bbox`` is the island's canonical "one box": the union of its blue
    boxes expanded by *pad* px per side, in IMAGE coordinates, unclamped.
    """
    union_overrides = {}
    for island in dataset.islands:
        ix1, iy1, ix2, iy2 = island.image_bbox
        local_boxes = []
        min_w = min_h = None
        for b in boxes:
            x1, y1, x2, y2 = b["image_bbox"]
            if ix1 <= (x1 + x2) // 2 <= ix2 and iy1 <= (y1 + y2) // 2 <= iy2:
                px1 = max(0, x1 - pad - ix1)
                py1 = max(0, y1 - pad - iy1)
                px2 = min(ix2 - ix1, x2 + pad - ix1)
                py2 = min(iy2 - iy1, y2 + pad - iy1)
                local_boxes.append((px1, py1, px2, py2))
                box_w = px2 - px1
                box_h = py2 - py1
                if min_w is None or box_w < min_w:
                    min_w = box_w
                if min_h is None or box_h < min_h:
                    min_h = box_h
        if not local_boxes:
            continue
        mask = np.zeros((iy2 - iy1, ix2 - ix1), dtype=np.uint8)
        for px1, py1, px2, py2 in local_boxes:
            mask[py1:py2, px1:px2] = 255
        raw = [
            (b["image_bbox"][0] - pad, b["image_bbox"][1] - pad,
             b["image_bbox"][2] + pad, b["image_bbox"][3] + pad)
            for b in boxes
            if ix1 <= (b["image_bbox"][0] + b["image_bbox"][2]) // 2 <= ix2
            and iy1 <= (b["image_bbox"][1] + b["image_bbox"][3]) // 2 <= iy2
        ]
        union_overrides[island.id] = {
            "mask": mask,
            "area": float(np.count_nonzero(mask)),
            "bbox": (
                min(item[0] for item in raw),
                min(item[1] for item in raw),
                max(item[2] for item in raw),
                max(item[3] for item in raw),
            ),
            "glyph_width_limit": float(min_w),
            "glyph_height_limit": float(min_h),
        }
    return union_overrides


def _parse_color(value: str) -> tuple[int, int, int] | None:
    """Parse 'R,G,B' into a tuple, or None for 'none'/empty."""
    if not value or value.strip().lower() in ("none", "null"):
        return None
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise ValueError(f"invalid colour {value!r} (expected R,G,B or none)") from None
    if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
        raise ValueError(f"invalid colour {value!r} (expected R,G,B or none)")
    return tuple(parts)  # type: ignore[return-value]


def erase_and_typeset_page_with(
    dataset_dir, typeset_dir, page, translations, directions, dropped_boxes=None,
    min_font_size: int | None = None,
    font_name: str | None = None,
    stroke_divisor: int | None = None,
    row_gap_divisor: int | None = None,
    column_gap_divisor: int | None = None,
    tracking_divisor: int | None = None,
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    background_color: tuple[int, int, int] | None = None,
):
    import dataclasses

    import numpy as np

    from manga_localizer.eraser import ErasureConfig
    from manga_localizer.island_contract import load_island_dataset
    from manga_localizer.island_erasure import erase_island_image
    from manga_localizer.island_typeset import TypesetConfig, typeset_island_page

    dropped = set(dropped_boxes or ())
    dataset = load_island_dataset(dataset_dir / page / "islands.json")
    # column positions follow the blue boxes (pad 6), page_scale and
    # dropped (false selection) boxes excluded
    selection_json = Path(dataset_dir) / page / "selection_blocks.json"
    blue_lines = (
        blue_boxes_to_lines(selection_json, pad=6, exclude=dropped)
        if selection_json.is_file()
        else []
    )
    boxes = []
    if selection_json.is_file():
        payload = json.loads(selection_json.read_text(encoding="utf-8-sig"))
        boxes = [
            b for b in payload.get("selection_blocks") or []
            if not b.get("page_scale") and b.get("id") not in dropped
        ]
    boxed_ids = set()
    for island in dataset.islands:
        ix1, iy1, ix2, iy2 = island.image_bbox
        for b in boxes:
            x1, y1, x2, y2 = b["image_bbox"]
            if ix1 <= (x1 + x2) // 2 <= ix2 and iy1 <= (y1 + y2) // 2 <= iy2:
                boxed_ids.add(island.id)
                break
    # Erase only the islands that carry blue boxes AND have a translation:
    # islands whose boxes all read empty (false selections) or whose text
    # could not be read keep their original pixels untouched -- never erase
    # what we cannot replace.
    erase_ids = {island_id for island_id in boxed_ids if translations.get(island_id)}
    if not erase_ids:
        # No erasure-worthy islands: nothing to erase or typeset, the page
        # stays exactly as the original.
        with Image.open(dataset_dir / page / "original.jpg") as image:
            image.convert("RGB").save(typeset_dir / f"{page}.typeset.png")
        return None
    with Image.open(dataset_dir / page / "original.jpg") as image:
        source = image.convert("RGB").copy()
    # union masks: each island's blue boxes (pad 6) union, island-local
    # coords; glyph size caps follow the PADDED box dimensions (vertical:
    # font width <= narrowest padded box width; horizontal: font height <=
    # shortest padded box height)
    union_overrides = build_union_overrides(dataset, boxes, pad=6) if boxes else {}
    config = TypesetConfig(
        min_font_size=min_font_size if min_font_size else TypesetConfig.min_font_size,
        font_name=font_name or TypesetConfig.font_name,
        stroke_divisor=stroke_divisor or TypesetConfig.stroke_divisor,
        row_gap_divisor=row_gap_divisor or TypesetConfig.row_gap_divisor,
        column_gap_divisor=column_gap_divisor or TypesetConfig.column_gap_divisor,
        tracking_divisor=tracking_divisor or TypesetConfig.tracking_divisor,
        text_color=text_color or TypesetConfig.text_color,
        stroke_color=stroke_color or TypesetConfig.stroke_color,
        background_color=background_color,
    )
    # Canonical "one box": each boxed island's blue-box union expanded
    # 6 px becomes its island bbox for erasure and typesetting -- the
    # erasure range, the layout bounds and the reported block bbox all
    # refer to the same box, so text never lands on un-erased art.
    canonical = {}
    for island in dataset.islands:
        box = (union_overrides.get(island.id) or {}).get("bbox")
        if box is not None:
            canonical[island.id] = tuple(int(value) for value in box)
    canonical_dataset = dataclasses.replace(
        dataset,
        islands=tuple(
            dataclasses.replace(island, image_bbox=canonical.get(island.id, island.image_bbox))
            for island in dataset.islands
        ),
    )
    active_dataset = dataclasses.replace(
        canonical_dataset,
        islands=tuple(
            island for island in canonical_dataset.islands if island.id in erase_ids
        ),
    )
    # Erase the WHOLE canonical box of every erasure-worthy island (plus
    # the detected glyph ink): whatever the typesetter draws inside the
    # box is always on erased art.
    extra_regions = {
        island_id: canonical[island_id]
        for island_id in erase_ids
        if island_id in canonical
    }
    erased = erase_island_image(
        source, active_dataset, config=ErasureConfig(), min_island_area=800.0,
        extra_regions=extra_regions,
    ).image
    result = typeset_island_page(
        erased, canonical_dataset, translations, blue_lines, config=config,
        direction_overrides=directions,
        union_overrides=union_overrides,
    )
    result.image.save(typeset_dir / f"{page}.typeset.png")
    return result


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="One-click manga translation")
    parser.add_argument("--images", default=str(IMAGES), help="source image directory")
    parser.add_argument("--pages", help="page list e.g. 1-24 or 1,3,5")
    parser.add_argument("--parallel", type=int, default=3, help="parallel Photos windows")
    parser.add_argument(
        "--display-scale",
        type=int,
        default=25,
        help="pin every Photos window to this percent of its page's size "
             "(default 25: whole page fully visible at fixed 25%% zoom, "
             "image file never resized)",
    )
    parser.add_argument(
        "--layout-row-gap",
        type=int,
        default=None,
        help="fixed per-column spacing (px) in --display-scale layout",
    )
    parser.add_argument(
        "--layout-bottom-shift-up",
        type=int,
        default=0,
        help="lift the last row of windows this many px off the bottom "
             "edge in --display-scale layout (default 0)",
    )
    parser.add_argument("--capture-only", action="store_true",
                        help="stage 1 only (capture + blue-box check images)")
    parser.add_argument("--process", action="store_true",
                        help="stage 2 only (background processing)")
    parser.add_argument("--auto", action="store_true",
                        help="run stage 2 immediately after stage 1")
    parser.add_argument("--bluebox-fill", default="0,0,0",
                        help="fill colour outside blue boxes (R,G,B) for the "
                             "masked manga input, default 0,0,0 (black)")
    parser.add_argument(
        "--confirm-no-boxes",
        help="comma-separated pages that the user reviewed and confirmed to "
             "have no blue boxes (they may be closed/skipped downstream)",
    )
    parser.add_argument(
        "--ai-url",
        default="https://api.deepseek.com/v1",
        help="OpenAI-compatible AI base URL for whole-page translation "
             "(default DeepSeek)",
    )
    parser.add_argument(
        "--ai-model",
        default="deepseek-chat",
        help="AI model name (default deepseek-chat)",
    )
    parser.add_argument(
        "--ai-api-key",
        default="",
        help="AI API key (or DEEPSEEK_API_KEY env).  When set, translation "
             "prefers the AI whole-page batch and falls back to local Sakura; "
             "without it everything goes through local Sakura.",
    )
    parser.add_argument(
        "--ai-ocr-url",
        default="",
        help="OpenAI-compatible vision base URL for AI OCR (e.g. GLM-4V / "
             "Qwen-VL); empty = local manga-ocr",
    )
    parser.add_argument(
        "--ai-ocr-model",
        default="",
        help="AI OCR vision model name",
    )
    parser.add_argument(
        "--ai-ocr-api-key",
        default="",
        help="AI OCR API key (or AI_OCR_API_KEY env)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="stage2: skip pages whose typeset output already exists",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="stage2: skip the final PDF merge (typeset images only)",
    )
    parser.add_argument(
        "--layout-region",
        default="",
        help="constrain capture window layout to WxH px (e.g. 2400x1400); "
             "empty = full virtual screen",
    )
    parser.add_argument(
        "--box-delta",
        type=int,
        default=8,
        help="delta fallback: per-channel RGB change threshold (default 8)",
    )
    parser.add_argument(
        "--box-tolerance",
        type=float,
        default=5.0,
        help="strict overlay formula match slack (default 5)",
    )
    parser.add_argument(
        "--box-area-ratio",
        type=float,
        default=0.20,
        help="whole-page drop: frame-area fraction (default 0.20)",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=28,
        help="typeset font floor in px (default 28)",
    )
    parser.add_argument(
        "--font",
        default="msyh.ttc",
        help="typeset font file name (msyh.ttc / simhei.ttf / simkai.ttf / "
             "simfang.ttf / simsun.ttc)",
    )
    parser.add_argument(
        "--stroke-divisor",
        type=int,
        default=10,
        help="stroke width = font_size // N (default 10; larger = thinner)",
    )
    parser.add_argument(
        "--row-gap-divisor",
        type=int,
        default=4,
        help="horizontal row gap = font_size // N (default 4)",
    )
    parser.add_argument(
        "--column-gap-divisor",
        type=int,
        default=4,
        help="vertical column gap = font_size // N (default 4)",
    )
    parser.add_argument(
        "--tracking-divisor",
        type=int,
        default=16,
        help="character tracking = font_size // N (default 16)",
    )
    parser.add_argument(
        "--text-color",
        default="",
        help="typeset text colour R,G,B (default 0,0,0 black)",
    )
    parser.add_argument(
        "--stroke-color",
        default="",
        help="typeset stroke colour R,G,B (default 255,255,255 white)",
    )
    parser.add_argument(
        "--background-color",
        default="",
        help="per-glyph background colour R,G,B or 'none' (default none)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="parallel translate+typeset page workers (default 1; max 3)",
    )
    args = parser.parse_args(argv)
    global BLUEBOX_FILL
    try:
        parts = [int(part.strip()) for part in args.bluebox_fill.split(",")]
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            BLUEBOX_FILL = tuple(parts)
    except ValueError:
        pass

    images = Path(args.images).resolve()
    out_root = ROOT / "outputs" / "oneclick" / images.name
    if args.confirm_no_boxes:
        confirmed = sorted(_parse_pages(args.confirm_no_boxes) or set())
        if not confirmed:
            print("no pages to confirm", file=sys.stderr)
            return 2
        _confirm_no_boxes(out_root / "capture", confirmed)
        print(f"confirmed no-blue-box pages: {', '.join(confirmed)}")
        return 0
    pages_set = _parse_pages(args.pages)
    pages = sorted(
        (path.stem for path in images.iterdir()
         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
         and (pages_set is None or path.stem in pages_set)),
        key=lambda item: int(item) if item.isdigit() else item.lower(),
    )
    if not pages:
        print(f"no images found in {images}", file=sys.stderr)
        return 2

    if not args.process:
        ok = stage1_capture(images, out_root, pages_set, args.parallel,
                            args.display_scale, args.layout_row_gap,
                            args.layout_bottom_shift_up,
                            layout_region=args.layout_region,
                            box_delta=args.box_delta,
                            box_tolerance=args.box_tolerance,
                            box_area_ratio=args.box_area_ratio)
        if not ok:
            print("capture failed", file=sys.stderr)
            return 2
        skipped = _skip_pages(out_root / "capture", pages)
        print(f"capture done: {len(pages)} pages ({len(skipped)} skipped: "
              f"black/no-text/failed) -> {out_root / 'check'}")
        review = _review_required_pages(out_root / "capture", pages)
        if review:
            print("!! " + "=" * 60)
            print(f"!! 待人工审核（三次采集无蓝色像素）: {', '.join(review)}")
            print("!! 请查看这些页的截图确认是否真的无蓝色选区框；")
            print("!! 确认无框后运行:")
            print(f"!!   python one_click_translate.py --images \"{images}\" "
                  f"--confirm-no-boxes {','.join(review)}")
            print("!! " + "=" * 60)
        if not args.auto:
            print("inspect the blue-box check images, then run: "
                  f"python one_click_translate.py --images \"{images}\" --process")
            return 0
    review = _review_required_pages(out_root / "capture", pages)
    if review:
        print("!! " + "=" * 60)
        print(f"!! 待人工审核页（未确认，已跳过处理）: {', '.join(review)}")
        print("!! 审核确认无框后运行:")
        print(f"!!   python one_click_translate.py --images \"{images}\" "
              f"--confirm-no-boxes {','.join(review)}")
        print("!! " + "=" * 60)
    active_pages = [p for p in pages if p not in _skip_pages(out_root / "capture", pages)]
    if not active_pages:
        print("no capturable pages (all black)", file=sys.stderr)
        return 2
    _config = _load_api_config()
    return 0 if stage2_process(
        images,
        out_root,
        active_pages,
        ai_url=args.ai_url,
        ai_model=args.ai_model,
        ai_api_key=args.ai_api_key
        or os.environ.get("DEEPSEEK_API_KEY", "")
        or _config.get("deepseek_api_key", ""),
        ai_ocr_url=args.ai_ocr_url or _config.get("ai_ocr_url", ""),
        ai_ocr_model=args.ai_ocr_model or _config.get("ai_ocr_model", ""),
        ai_ocr_api_key=args.ai_ocr_api_key
        or os.environ.get("AI_OCR_API_KEY", "")
        or _config.get("ai_ocr_api_key", ""),
        resume=args.resume,
        no_pdf=args.no_pdf,
        min_font_size=args.min_font_size,
        font_name=args.font,
        stroke_divisor=args.stroke_divisor,
        row_gap_divisor=args.row_gap_divisor,
        column_gap_divisor=args.column_gap_divisor,
        tracking_divisor=args.tracking_divisor,
        text_color=args.text_color,
        stroke_color=args.stroke_color,
        background_color=args.background_color,
        max_workers=max(1, min(3, args.max_workers)),
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
