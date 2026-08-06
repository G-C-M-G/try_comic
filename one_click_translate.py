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
* Typesetting (island_typeset.py): the block must first fit inside the blue
  box UNION (pad 6), ratio A_text / A_union in [0.6, 1.1] (hard cap 1.2);
  on failure it falls back to the outer bounding rectangle with A_rect as
  the ratio base.  Glyph size <= ORIGINAL blue box dimensions: vertical
  size <= narrowest box width, horizontal size <= shortest box height.
  Column positions follow the pad-6 blue boxes; the block centre is the
  group-level bounding rectangle centre.
* A blue box read empty twice is treated as a false selection and dropped.
"""

from __future__ import annotations

import io
import json
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

IMAGES = Path(r"C:\Users\YLP\Desktop\软件\漫画翻译\原文\甲书")
BLUEBOX_FILL = (0, 0, 0)  # fill colour outside blue boxes (--bluebox-fill)


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


def stage1_capture(images: Path, out_root: Path, pages: set[str] | None, parallel: int, display_scale: int, row_gap_px: int | None, bottom_shift_up: int) -> bool:
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
    ]
    if row_gap_px is not None:
        cmd += ["--layout-row-gap", str(row_gap_px)]
    if bottom_shift_up:
        cmd += ["--layout-bottom-shift-up", str(bottom_shift_up)]
    if pages:
        cmd += ["--pages", ",".join(sorted(pages))]
    completed = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    print(output[-2000:])
    if completed.returncode != 0:
        return False

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


def run_manga(crops: list[dict], out_root: Path) -> dict:
    """manga-ocr every crop in one subprocess (model loads once)."""
    script = """\
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image
from manga_ocr import MangaOcr
crops = json.load(open(sys.argv[1], encoding="utf-8"))
mocr = MangaOcr()
for entry in crops:
    image = Image.open(entry["source"]).convert("RGB")
    x1, y1, x2, y2 = entry["bbox"]
    crop = image.crop((x1, y1, x2, y2))
    text = str(mocr(crop) or "").strip()
    print(json.dumps({"id": entry["id"], "text": text}, ensure_ascii=False))
    sys.stdout.flush()
"""
    crops_path = out_root / "crops.json"
    crops_path.write_text(json.dumps(crops, ensure_ascii=False), encoding="utf-8")
    python = ROOT / "venv" / "Scripts" / "python.exe"
    env = dict(__import__("os").environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    completed = subprocess.run(
        [str(python), "-c", script, str(crops_path)],
        capture_output=True, text=False, timeout=3600, check=False, env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"manga-ocr failed: {detail}")
    readings = {}
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        readings[payload["id"]] = str(payload.get("text") or "")
    return readings


def _source_image_for(images: Path, page: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images / f"{page}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_bluebox_dataset(
    manifest_path, page_dir, source_path, min_island_area: float = 800.0, pad: int = 2
) -> list[dict]:
    """islands.json purely from the Photos blue boxes (original image space).

    No highlight-island detection: every island is one selection group (the
    atomic blue boxes Photos produced), padded ``pad`` px.  Raises
    ValueError when the page has no usable blue boxes -- the caller keeps
    the page untouched then.  Also writes selection_blocks.json (typeset
    column positions) and a copy of the original image.
    """
    import shutil

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    capture = payload.get("capture", {})
    blocks = capture.get("selection_blocks") or []
    groups = capture.get("selection_groups") or []
    source = payload.get("source", {})
    image_w = int(source["width"])
    image_h = int(source["height"])
    atomic_ids = {item["id"] for item in blocks if not item.get("page_scale")}
    islands: list[dict] = []
    for group in groups:
        members = [
            member for member in group.get("members", [])
            if member in atomic_ids
        ]
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


def stage2_process(images: Path, out_root: Path, pages: list[str]) -> bool:
    """Background: blue boxes -> manga -> translate -> typeset -> PDF."""
    from island_extract import build_fallback_dataset
    from manga_localizer.island_contract import load_island_dataset
    from manga_localizer.island_translate import (
        IslandTranslateConfig,
        IslandTranslationCache,
        translate_island_page,
    )
    from manga_localizer.translator import load_glossary

    capture_dir = out_root / "capture"
    dataset_dir = out_root / "dataset"
    typeset_dir = out_root / "typeset"
    typeset_dir.mkdir(parents=True, exist_ok=True)

    fallback = _fallback_pages_for(capture_dir, pages)
    # Islands come ONLY from the Photos blue boxes (no highlight-island
    # detection): one island per selection group, original image space.
    for page in pages:
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
    readings = run_manga(crops, out_root)

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
        reread_readings = run_manga(rereads, out_root)
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
    translations = {}
    for page in active_pages:
        dataset = load_island_dataset(dataset_dir / page / "islands.json")
        result = translate_island_page(
            dataset,
            page_texts[page],
            IslandTranslateConfig(
                precision="local",
                sakura_base_url="http://127.0.0.1:8080/v1",
                sakura_model="sukinishiro",
                glossary_path=str(ROOT / "config" / "glossary.json"),
            ),
            glossary=glossary,
            cache=cache,
        )
        translations[page] = dict(result.translations)
        print(f"translate {page}: {len(result.units)} islands")

    for page in active_pages:
        erase_and_typeset_page_with(
            dataset_dir, typeset_dir, page, translations[page],
            page_directions.get(page), dropped_boxes.get(page, set()),
        )
        print(f"typeset {page} -> {typeset_dir / f'{page}.typeset.png'}")

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

    images_out = [
        Image.open(typeset_dir / f"{page}.typeset.png").convert("RGB")
        for page in pages
        if (typeset_dir / f"{page}.typeset.png").is_file()
    ]
    if not images_out:
        print("no pages to assemble into a PDF")
        return True
    pdf_path = out_root / f"{images.name}.pdf"
    images_out[0].save(pdf_path, save_all=True, append_images=images_out[1:], resolution=100)
    print(f"-> {pdf_path} ({len(images_out)} pages)")
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


def _skip_pages(capture_dir: Path, pages: list[str]) -> set[str]:
    """Pages with no usable capture (black / no-text / failed) are skipped
    silently; fallback_manga pages stay active."""
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
    return skipped


def build_union_overrides(dataset, boxes, pad: int = 6) -> dict:
    """Per-island blue-box union masks (island-local coords).

    Each island's mask is the union of its blue boxes expanded by *pad* px
    (clamped to the island bbox); ``area`` is the mask pixel count.  Glyph
    size caps follow the PADDED box dimensions: vertical font size
    (glyph width) <= narrowest padded box width, horizontal font size
    (glyph height) <= shortest padded box height.
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
        union_overrides[island.id] = {
            "mask": mask,
            "area": float(np.count_nonzero(mask)),
            "glyph_width_limit": float(min_w),
            "glyph_height_limit": float(min_h),
        }
    return union_overrides


def erase_and_typeset_page_with(
    dataset_dir, typeset_dir, page, translations, directions, dropped_boxes=None
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
    active_dataset = dataclasses.replace(
        dataset,
        islands=tuple(
            island for island in dataset.islands if island.id in erase_ids
        ),
    )
    with Image.open(dataset_dir / page / "original.jpg") as image:
        erased = erase_island_image(
            image, active_dataset, config=ErasureConfig(), min_island_area=800.0
        ).image
    # union masks: each island's blue boxes (pad 6) union, island-local
    # coords; glyph size caps follow the PADDED box dimensions (vertical:
    # font width <= narrowest padded box width; horizontal: font height <=
    # shortest padded box height)
    union_overrides = build_union_overrides(dataset, boxes, pad=6) if boxes else {}
    result = typeset_island_page(
        erased, dataset, translations, blue_lines, config=TypesetConfig(),
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
                            args.layout_bottom_shift_up)
        if not ok:
            print("capture failed", file=sys.stderr)
            return 2
        skipped = _skip_pages(out_root / "capture", pages)
        print(f"capture done: {len(pages)} pages ({len(skipped)} skipped: "
              f"black/no-text/failed) -> {out_root / 'check'}")
        if not args.auto:
            print("inspect the blue-box check images, then run: "
                  f"python one_click_translate.py --images \"{images}\" --process")
            return 0
    active_pages = [p for p in pages if p not in _skip_pages(out_root / "capture", pages)]
    if not active_pages:
        print("no capturable pages (all black)", file=sys.stderr)
        return 2
    return 0 if stage2_process(images, out_root, active_pages) else 2


if __name__ == "__main__":
    raise SystemExit(main())
