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

from manga_localizer.errors import ErasureError

from manga_localizer.ai_ocr import DEFAULT_WORKERS as AI_OCR_DEFAULT_WORKERS
from manga_localizer.text_rules import (
    is_ocr_commentary,
    is_sfx_like,
    short_source,
)
from manga_localizer.thresholds import (
    PAGE_FONT_CEILING_RATIO,
    SFX_MAX_FONT_SIZE,
    SHORT_SOURCE_MAX_CHARS,
    STROKE_OVERFLOW_MARGIN,
    SFX_GLOSSARY_MIN_COUNT,
    SFX_MIN_SIDE_PAGE,
    SFX_MIN_SIDE_RATIO,
    WHOLE_PAGE_AREA_RATIO,
    bbox_pad as source_bbox_pad,
    glyph_margin as source_glyph_margin,
)

BLUEBOX_FILL = (0, 0, 0)  # fill colour outside blue boxes (--bluebox-fill)


def _venv_python() -> Path:
    """Interpreter for the capture/OCR subprocesses.

    Prefers the repo-root venv (its dependency set differs from
    paddle_venv).  Falls back to the running interpreter when venv/ is
    absent so a plain checkout still works; a frozen build has no usable
    interpreter and must fail loudly instead of spawning the exe itself.
    """
    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return venv_python
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            "找不到 venv\\Scripts\\python.exe：打包版依赖仓库根目录的 venv/（见 README 环境一节）"
        )
    return Path(sys.executable)


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
    # Tolerate full-width punctuation (users type "4，6" / "4、6" in an
    # IME): normalize to the half-width separator before splitting, so a
    # full-width comma cannot become part of a page name and silently
    # match nothing ("no images found").
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


def _pages_from_dataset(out_root: Path, pages_set: set[str] | None) -> list[str]:
    """只跑 stage2 时的页清单：取自既有 dataset（页名 = 原图 stem）。

    用户把 `--images` 指到产物目录（`outputs/oneclick/{书名}`）时用它继续——
    产物即契约，stage2 的输入本来就是 dataset，不需要原图。
    """
    dataset_dir = out_root / "dataset"
    if not dataset_dir.is_dir():
        return []
    return sorted(
        (
            path.name for path in dataset_dir.iterdir()
            if path.is_dir() and (path / "islands.json").is_file()
            and (pages_set is None or path.name in pages_set)
        ),
        key=_page_sort_key,
    )


def _output_dir_hint(images: Path) -> str:
    """`--images` 指到产物目录时的可执行提示（最常见的误操作）。"""
    markers = [name for name in ("capture", "dataset", "check", "typeset")
               if (images / name).is_dir()]
    if not markers:
        return ""
    return (
        f"提示：{images} 看起来是**输出目录**（含 {'/'.join(markers)} 产物），"
        "不是原图目录。\n"
        f"      --images 要指向原图（如 漫画翻译/原文/{{{images.name}}}），书名取该目录名；\n"
        f"      产物在 outputs/oneclick/{images.name}/ 下，重采需要原图目录，\n"
        "      只重做 stage2 可以继续指产物目录（需要其中有 dataset/*/islands.json）。"
    )


def _is_black_page(image_path: Path, dark_ratio: float = 0.90) -> bool:
    """整页几乎全黑（无内容）——与 photos_capture._is_black_page 同判据。"""
    try:
        arr = np.asarray(Image.open(image_path).convert("L"))
    except (OSError, ValueError):
        return False
    if arr.size == 0:
        return False
    return float((arr < 32).mean()) > dark_ratio


def _direction_groups(bboxes: Sequence[Sequence[int]]) -> dict[str, list[int]]:
    """按方向分组：横框（宽≥高）/ 竖框。"""
    groups: dict[str, list[int]] = {"h": [], "v": []}
    for index, box in enumerate(bboxes):
        key = "h" if (box[2] - box[0]) >= (box[3] - box[1]) else "v"
        groups[key].append(index)
    return groups


def _direction_baseline(
    bboxes: Sequence[Sequence[int]], key: str, *,
    min_group: int = 3, page_median: float | None = None,
) -> float:
    """某方向的短边基准；该方向样本 < ``min_group`` 时回落全页中位数。

    "短边 ≈ 字号"是这套判据的地基（横竖都成立），基准按方向分开算才不会被
    另一种字号污染；样本太少时基准无法自证，回落全页（判据说明见
    ``_filter_large_sfx``）。
    """
    if page_median is None:
        page_median = float(np.median(
            [min(b[2] - b[0], b[3] - b[1]) for b in bboxes]
        ))
    indices = _direction_groups(bboxes)[key]
    if len(indices) < min_group:
        return page_median
    return float(np.median([
        min(bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1])
        for i in indices
    ]))


def oversized_box_ids(
    boxes: Sequence[dict], *,
    ratio: float = SFX_MIN_SIDE_RATIO, min_group: int = 3,
    page_short_side: float | None = None,
    page_floor: float = SFX_MIN_SIDE_PAGE,
) -> set[str]:
    """短边远超同方向基准的框（大概率拟声/效果字）。

    它们**不参与合并**：把这种框和相邻的竖排框并成一组，组的外接矩形会变得很大、
    字号也跟着撑大（实测页 7/11/19 的巨型字号都是这么来的）；而且原图里它本来就
    是一个独立的超大文字，不该和别的列共享一个排版容器。

    **必须同时超过绝对下限**（``page_short_side × page_floor``，默认页短边 8%）：
    只看倍数的话，小字页面（页 97 常规字号 22px）的门槛会低到 33px，页上几乎所有
    框都被判成"过大"→ 全被压到 30px、还互相不合并（实测封顶框 23% → 加下限后 8%）。
    这与采集期剔除大拟音的判据同形（那条也要求同时超过页短边 10%）。
    """
    if len(boxes) < 3:
        return set()
    bboxes = [list(b["image_bbox"]) for b in boxes]
    page_median = float(np.median(
        [min(b[2] - b[0], b[3] - b[1]) for b in bboxes]
    ))
    floor_px = (page_floor * float(page_short_side)) if page_short_side else 0.0
    locked: set[str] = set()
    for box in boxes:
        bb = box["image_bbox"]
        key = "h" if (bb[2] - bb[0]) >= (bb[3] - bb[1]) else "v"
        baseline = _direction_baseline(bboxes, key, min_group=min_group,
                                       page_median=page_median)
        short = min(bb[2] - bb[0], bb[3] - bb[1])
        if baseline > 0 and short > ratio * baseline and short > floor_px:
            locked.add(str(box.get("id") or ""))
    return locked


def oversized_single_box_islands(
    dataset, boxes: Sequence[dict], *,
    ratio: float = SFX_MIN_SIDE_RATIO, min_group: int = 3,
    page_floor: float = SFX_MIN_SIDE_PAGE,
) -> set[str]:
    """**未被合并的框**里，短边远超同方向平均值的岛（大概率是拟声词）。

    只认单框岛：合并过的组是"一句被拆成几段"，短边大是句子长的结果，不是
    字号大。这类单框在原图里就是巨大的拟声/效果字（比页面常规字号大两倍
    以上），中文照原尺寸放大只会盖过对白 → 字号封到 --sfx-max-font-size。
    阈值复用大拟音筛框的 ``--sfx-min-side-ratio``（默认 2.0× 同方向基准）。
    """
    if len(boxes) < 3:
        return set()
    bboxes = [list(b["image_bbox"]) for b in boxes]
    page_median = float(np.median(
        [min(b[2] - b[0], b[3] - b[1]) for b in bboxes]
    ))
    floor_px = page_floor * float(min(dataset.image_size))
    capped: set[str] = set()
    for island in dataset.islands:
        ix1, iy1, ix2, iy2 = island.image_bbox
        members = [
            b for b in boxes
            if ix1 <= (b["image_bbox"][0] + b["image_bbox"][2]) // 2 <= ix2
            and iy1 <= (b["image_bbox"][1] + b["image_bbox"][3]) // 2 <= iy2
        ]
        if len(members) != 1:
            continue
        box = members[0]["image_bbox"]
        key = "h" if (box[2] - box[0]) >= (box[3] - box[1]) else "v"
        baseline = _direction_baseline(bboxes, key, min_group=min_group,
                                       page_median=page_median)
        short = min(box[2] - box[0], box[3] - box[1])
        if baseline > 0 and short > ratio * baseline and short > floor_px:
            capped.add(island.id)
    return capped


def _filter_large_sfx(
    bboxes: list[list[int]], image_w: int, image_h: int, *,
    ratio: float = 2.0, page_floor: float = 0.10, min_group: int = 3,
) -> tuple[list[list[int]], list[tuple[list[int], int, str]]]:
    """剔除大号拟音/效果字框，返回 ``(保留, [(框, 短边px, 基准说明)])``。

    判据用 **min(w,h)（短边）** 而不是单边长度：文字行无论横竖，短边都≈字号，
    逐页稳定；而大拟声词两个方向都大，短边会远超基准。**不能用宽度**——横排行
    天然就宽（实测本册最宽的对白框 0.11 页宽，而竖排短边中位仅 0.038 页宽），
    按宽度筛必然误伤横排。

    **基准按方向分组**（横框比横框、竖框比竖框），避免两种字号混在一起互相
    污染。但某方向样本 < ``min_group`` 时无法自证基准——此时回落到全页短边
    中位数。这一步是必需的：漫画以竖排为主，待删的大拟声词往往就是该页**仅有**
    的横框，若严格"横比横"，基准会等于离群值本身而全部漏删（实测页 32：横框
    仅 1 个且就是那个 385px 的大字，基准 385 → 阈值 962 → 一个都删不掉）。

    另需同时超过 ``page_floor × min(页宽, 页高)`` 的绝对下限，避免在只有
    两三个框的小页上误删。
    """
    if len(bboxes) < 3:
        return list(bboxes), []
    short = [min(box[2] - box[0], box[3] - box[1]) for box in bboxes]
    page_median = float(np.median(short))
    floor_px = page_floor * min(image_w, image_h)

    kept: list[list[int]] = []
    dropped: list[tuple[list[int], int, str]] = []
    for key, indices in _direction_groups(bboxes).items():
        baseline = _direction_baseline(
            bboxes, key, min_group=min_group, page_median=page_median,
        )
        basis = (
            f"{'横' if key == 'h' else '竖'}框基准{baseline:.0f}px"
            if len(indices) >= min_group
            else f"仅{len(indices)}个{'横' if key == 'h' else '竖'}框,回落全页基准{baseline:.0f}px"
        )
        for index in indices:
            box = bboxes[index]
            if short[index] > ratio * baseline and short[index] > floor_px:
                dropped.append((box, short[index], basis))
            else:
                kept.append(box)
    kept.sort(key=lambda box: (box[1], -box[2]))
    return kept, dropped


def capture_boxes_with_paddle(
    images: Path, out_root: Path, pages: set[str] | None,
    *, device: str = "auto", box_area_ratio: float = WHOLE_PAGE_AREA_RATIO,
    min_side: int = 6,
) -> bool:
    """stage1 的另一种框来源：PaddleOCR 全页检测（默认）。

    产出与 Photos 蓝框**完全相同的 capture.json 契约**（source + capture.
    selection_blocks / selection_groups / complete / 状态字段），所以后续
    每一步——check 核对图、dataset 构建、填黑图、20%/40% 分组、stage2——
    都零改动复用。相比 Photos 路径：无窗口自动化、无焦点/时序问题、
    离线确定性、GPU 下 32 页约 7 秒。

    检测到的框是**文本行**级（Photos 对白页也是按句切分，两者粒度接近；
    但大字/疏排页面 Photos 会退化成逐字，Paddle 仍是整行——见 开发记录）。
    """
    capture_dir = _capture_source_dir(out_root, "paddle")
    capture_dir.mkdir(parents=True, exist_ok=True)
    raw_output = out_root / "paddle_boxes.json"
    worker = ROOT / "paddle_ocr_worker.py"
    cmd = [
        str(ROOT / "paddle_venv" / "Scripts" / "python.exe"),
        "-u", str(worker), str(images), str(raw_output), "--device", device,
    ]
    if pages:
        cmd += ["--pages", ",".join(sorted(pages))]
    print(f"[box-source=paddle] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line)
    code = proc.wait()
    if code != 0 or not raw_output.is_file():
        print(f"paddle box detection failed (exit {code})", file=sys.stderr)
        return False

    payload = json.loads(raw_output.read_text(encoding="utf-8"))
    print(f"[box-source=paddle] backend={payload.get('backend')} "
          f"device={payload.get('device')}")
    written = 0
    for entry in payload.get("pages") or []:
        source_path = Path(entry["source"])
        try:
            with Image.open(source_path) as handle:
                image_w, image_h = handle.size
        except (OSError, ValueError):
            continue
        black_page = _is_black_page(source_path)
        image_area = max(1, image_w * image_h)
        candidates = []
        for prediction in entry.get("predictions") or []:
            polygon = prediction.get("polygon") or []
            if len(polygon) < 3:
                continue
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(image_w, max(xs)), min(image_h, max(ys))
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            # 整页大框：不是原子文本单位（与 Photos 的 20% 过滤同规则）
            if (x2 - x1) * (y2 - y1) > box_area_ratio * image_area:
                continue
            candidates.append((y1, -x2, [x1, y1, x2, y2]))

        # 大拟音/效果字过滤放在共用的数据集构建阶段（build_bluebox_dataset），
        # 这样 Photos 与 Paddle 两条路径共用同一套规则；采集期只负责检出框。
        # id 稳定分配：按阅读序（上→下，右→左），与 Photos 的编号习惯一致
        candidates.sort(key=lambda item: (item[0], item[1]))
        blocks = [
            {
                "id": f"sel_{index:03d}",
                "display_bbox": list(box),
                "image_bbox": list(box),
                "page_scale": False,
            }
            for index, (_, _, box) in enumerate(candidates, 1)
        ]
        groups = [
            {
                "id": f"selg_{index:03d}",
                "image_bbox": [int(value) for value in bbox],
                "display_bbox": [int(value) for value in bbox],
                "members": member_ids,
            }
            for index, (bbox, member_ids) in enumerate(
                _group_boxes_by_overlap(blocks), 1
            )
            if member_ids
        ]
        page_dir = capture_dir / source_path.stem
        page_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "backend": payload.get("backend") or "PaddleOCR",
            "source": {
                "path": str(source_path),
                "width": image_w,
                "height": image_h,
            },
            "capture": {
                "app": "paddle-ocr",
                "method": "full-page text detection (box-source=paddle)",
                "frames_only": False,
                "box_source": "paddle",
                "device": payload.get("device"),
                "viewport_size": [image_w, image_h],
                "island_count": len(blocks),
                "selection_boxes_status": "ok" if blocks else "no_boxes",
                "selection_missing": not blocks,
                # 硬规则沿用：检测不到任何框的页需人工确认后才进入 stage2
                "selection_review_required": not blocks,
                "selection_blocks": blocks,
                "selection_groups": groups,
                "black_page": black_page,
                "complete": True,
            },
        }
        (page_dir / "capture.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1
        print(f"paddle capture page {source_path.stem}: {len(blocks)} boxes"
              + ("  [black page]" if black_page else ""))
    print(f"paddle box capture done: {written} page(s) -> {capture_dir}")
    return written > 0


def stage1_capture(
    images: Path, out_root: Path, pages: set[str] | None, parallel: int,
    display_scale: int, row_gap_px: int | None, bottom_shift_up: int,
    layout_region: str = "", box_delta: int = 8, box_tolerance: float = 5.0,
    box_area_ratio: float = WHOLE_PAGE_AREA_RATIO, overwrite: bool = False,
    ocr_engine: str = "auto", paddle_device: str = "auto",
    box_source: str = "paddle",
    row_spacing_px: int | None = None,
    right_gap_px: int = 0, scan_wait: float = 0.5,
    scan_settle: float = 1.3, select_wait: float = 0.45,
    regrab_wait: float = 0.6, focus_clear_pct: float = 4.0,
) -> bool:
    """Parallel frames-only capture + blue-box check images.

    ``overwrite`` purges the selected pages' old artifacts (capture,
    check image, dataset, OCR/translation cache keys) first, so a
    re-capture replaces the old result instead of being skipped.
    """
    check_dir = out_root / "check"
    check_dir.mkdir(parents=True, exist_ok=True)
    # 采集结果按来源分开存放：capture/paddle/ 与 capture/photos/ 互不覆盖
    capture_dir = _capture_source_dir(out_root, "photos")
    capture_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        target = sorted(pages) if pages else [
            path.name for path in capture_dir.iterdir() if path.is_dir()
        ]
        _purge_page_artifacts(out_root, target, capture=True)
        print(f"overwrite: purged old artifacts of {len(target)} page(s)")

    from manga_localizer.progress import ProgressFile

    progress = ProgressFile(out_root)
    progress.set_stage("capture", total=len(pages) if pages else 0)
    if box_source == "paddle":
        # PaddleOCR 全页检测：离线、无窗口自动化（默认）
        progress.log(f"paddle full-page detection: "
                     f"{len(pages) if pages else 'all'} page(s)")
        ok = capture_boxes_with_paddle(
            images, out_root, pages,
            device=paddle_device, box_area_ratio=box_area_ratio,
        )
        if not ok:
            progress.fail("paddle box capture failed")
            return False
        progress.progress(len(pages) if pages else 0, item="paddle capture done")
    else:
        progress.log(f"capturing {len(pages) if pages else 'all'} pages "
                     f"(parallel={parallel})")
        cmd = [
            str(_venv_python()),
            "-u",  # unbuffered: the child must flush layout/progress lines
                   # immediately so a killed run leaves a meaningful trace in
                   # run.log instead of an empty buffer that looks like the
                   # capture never started.
            str(ROOT / "photos_capture.py"),
            "batch", "--fast",
            "--images", str(images),
            "--output-dir", str(capture_dir),
            "--frames-only",
            "--parallel", str(parallel),
            "--display-scale", str(display_scale),
            "--box-delta", str(box_delta),
            "--box-tolerance", str(box_tolerance),
            "--box-area-ratio", str(box_area_ratio),
        ]
        if ocr_engine != "auto":
            cmd += ["--ocr-engine", ocr_engine]
        if paddle_device != "auto":
            cmd += ["--paddle-device", paddle_device]
        if not overwrite:
            # Skip pages that were already captured (unless overwriting).
            cmd += ["--resume"]
        if row_gap_px is not None:
            cmd += ["--layout-row-gap", str(row_gap_px)]
        if row_spacing_px is not None:
            cmd += ["--layout-row-spacing", str(row_spacing_px)]
        if right_gap_px:
            cmd += ["--layout-right-gap", str(right_gap_px)]
        if scan_wait != 0.5:
            cmd += ["--scan-wait", str(scan_wait)]
        if scan_settle != 1.3:
            cmd += ["--scan-settle", str(scan_settle)]
        if select_wait != 0.45:
            cmd += ["--select-wait", str(select_wait)]
        if regrab_wait != 0.6:
            cmd += ["--regrab-wait", str(regrab_wait)]
        if focus_clear_pct != 4.0:
            cmd += ["--focus-clear-pct", str(focus_clear_pct)]
        if bottom_shift_up:
            cmd += ["--layout-bottom-shift-up", str(bottom_shift_up)]
        if layout_region:
            cmd += ["--layout-region", layout_region]
        if pages:
            cmd += ["--pages", ",".join(sorted(pages))]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
                total = len(pages) if pages else 0
                progress.progress(min(captured, total or 1), item=line[:60])
        code = proc.wait()
        if code != 0:
            progress.fail(f"photos_capture exited {code}")
            return False
        progress.progress(len(pages) if pages else 0, item="capture done")

    # check images：画的是 capture.json 的 selection_blocks（扩 6px），
    # 按来源分目录存放（check/paddle/ 红框、check/photos/ 蓝框）
    pad = 6
    for source, page, manifest_path in _iter_capture_pages(out_root):
        manifest = _load_capture(manifest_path)
        if manifest is None:
            continue
        capture = manifest.get("capture", {})
        if not capture.get("complete"):
            continue
        source_meta = manifest.get("source", {})
        image_size = (int(source_meta["width"]), int(source_meta["height"]))
        image_path = Path(source_meta["path"])
        if not image_path.is_file():
            continue
        color = "#e6194b" if source == "paddle" else "#4363d8"
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for block in capture.get("selection_blocks") or []:
            x1, y1, x2, y2 = block["image_bbox"]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image_size[0], x2 + pad)
            y2 = min(image_size[1], y2 + pad)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, min(image.height - 14, y2 + 2)), block["id"], fill=color)
        target_dir = _check_source_dir(out_root, source)
        target_dir.mkdir(parents=True, exist_ok=True)
        image.save(target_dir / f"{page}.bluebox_check.png")
    print(f"blue-box check images -> {check_dir}")

    # 报告：没有可用框的页（按来源列出）
    report_rows: list[tuple[str, str, str]] = []
    for source, page, manifest_path in _iter_capture_pages(out_root):
        manifest = _load_capture(manifest_path)
        if manifest is None:
            continue
        capture = manifest.get("capture", {})
        if capture.get("complete") and not (capture.get("selection_blocks") or []):
            reason = capture.get("selection_boxes_status") or "no blue boxes detected"
            report_rows.append((source, page, reason))
        elif capture.get("black_page"):
            report_rows.append((source, page, "black page (no content)"))
        elif capture.get("no_text"):
            report_rows.append((source, page, "no text detected"))
        elif capture.get("fallback_manga"):
            report_rows.append((source, page, "capture failed (manga fallback)"))
        elif not capture.get("complete"):
            report_rows.append(
                (source, page, capture.get("error") or "capture failed")
            )
    if report_rows:
        report_path = out_root / "BLUEBOX_REPORT.md"
        lines = ["# 蓝框缺失报告", ""]
        lines.append("以下页面未识别到蓝框（或采集未完成），将保持原样不处理：")
        lines.append("")
        for source, page, reason in report_rows:
            lines.append(f"- [{source}] 第 {page} 页：{reason}")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = "、".join(
            f"{page}({source})" for _source, page, _reason in report_rows
        )
        print("")
        print("=" * 60)
        print(f"!! 蓝框缺失: {len(report_rows)} 页 -> {report_path}")
        for source, page, reason in report_rows:
            print(f"   - [{source}] 第 {page} 页: {reason}")
        print("=" * 60)
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                "以下页面未识别到蓝框，将保持原样不处理：\n"
                f"{summary}\n\n详见 {report_path}",
                "蓝框缺失报告",
                0x00000040 | 0x00001000,
            )
        except Exception:
            pass
    else:
        print("all captured pages have blue boxes")
    progress.done(ok=True)
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


def _write_cache_json(path: Path, payload: dict) -> None:
    """Atomic JSON replace (tmp + rename), the cache-write convention."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_translations(out_root: Path) -> dict:
    """Persisted per-page translations ``{page: {island_id: text}}``.

    Written by the translate stage (and every full ``--process`` run), so
    the typeset stage can run standalone on already-translated pages.
    A missing or malformed file yields ``{}``.
    """
    path = out_root / "translations.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for page, items in payload.items():
        if not isinstance(items, dict):
            continue
        result[str(page)] = {str(iid): str(text) for iid, text in items.items()}
    return result


def _store_translations(out_root: Path, translations: dict) -> None:
    """Persist per-page translations, **按页合并**（原子整文件写）。

    只覆盖本次处理的页，其它页原样保留 —— 原先整文件替换，导致
    ``--pages 1`` 或 ``--stage translate --pages X`` 这种子集运行会把其余页
    的译文整本抹掉（实测踩到：跑单页验证后 31 页译文消失，后续嵌字只剩 1 页）。
    需要"丢弃某页旧译文"时由 ``_purge_page_artifacts(translations=True)`` 先行
    删除该页键，再写入新值，语义不受影响。
    """
    merged = dict(_load_translations(out_root))
    for page, per_island in (translations or {}).items():
        merged[str(page)] = dict(per_island or {})
    _write_cache_json(out_root / "translations.json", merged)


def _purge_page_artifacts(out_root: Path, pages: list[str], *,
                          capture: bool = False, dataset: bool = False,
                          readings: bool = True, translations: bool = True) -> None:
    """Delete one page's artifacts and drop its cache keys (--overwrite).

    Re-running a stage with ``--overwrite`` must start clean: a re-captured
    page's old dataset / readings / translations would otherwise be reused
    by downstream stages and mix stale text with fresh blue boxes.  All
    removals are idempotent; pages not in ``pages`` are never touched.
    """
    import shutil

    selected = set(pages)
    if capture:
        for page in pages:
            target = out_root / "dataset" / page
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            for source in CAPTURE_SOURCES:
                target = _capture_source_dir(out_root, source) / page
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                check_png = _check_source_dir(out_root, source) / f"{page}.bluebox_check.png"
                if check_png.is_file():
                    check_png.unlink(missing_ok=True)
    if dataset:
        for page in pages:
            target = out_root / "dataset" / page
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
    if readings:
        path = out_root / "ocr_readings.json"
        if path.is_file():
            payload = _load_readings_cache(out_root)
            stale = [key for key in payload if key.split(":", 1)[0] in selected]
            if stale:
                _backup_before_purge(path)
                for key in stale:
                    del payload[key]
                _write_cache_json(path, payload)
    if translations:
        path = out_root / "translations.json"
        if path.is_file():
            payload = _load_translations(out_root)
            stale = [page for page in payload if page in selected]
            if stale:
                _backup_before_purge(path)
                for page in stale:
                    del payload[page]
                _write_cache_json(path, payload)


def _backup_before_purge(path: Path) -> Path:
    """清空产物前先留一份 ``*.bak``（`--overwrite` 的安全网）。

    `--overwrite` 在 stage2 开头就删掉选中页的读数/译文，而重新生成它们要跑完
    OCR + 翻译。中途失败（进程被杀、断网、Key 失效、额度不足）就会**只丢不进**：
    实测踩到过一次——24 页译文被清空，只剩一个 ``{}``，等于白花钱重翻。
    有了这份备份，最坏情况也只是回滚到本次运行前。
    """
    import shutil

    backup = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copyfile(path, backup)
    except OSError:
        return path
    print(f"!! --overwrite 前已备份：{backup.name}（失败可回滚）")
    return backup



def _page_units(dataset_dir: Path, page: str) -> dict | None:
    """islands / blocks / groups for one page from disk.

    Returns ``None`` when the page has no dataset (never captured, or no
    blue boxes).  Mirrors the shape ``build_page_texts`` expects.
    """
    page_dir = dataset_dir / page
    islands_path = page_dir / "islands.json"
    if not islands_path.is_file():
        return None
    islands = json.loads(islands_path.read_text(encoding="utf-8-sig"))
    sel_path = page_dir / "selection_blocks.json"
    selection = (
        json.loads(sel_path.read_text(encoding="utf-8-sig"))
        if sel_path.is_file() else None
    )
    blocks = (selection or {}).get("selection_blocks") or []
    groups = (selection or {}).get("selection_groups") or []
    box_by_id = {b["id"]: b for b in blocks}
    return {
        "islands": {i["id"]: i for i in islands["islands"]},
        "blocks": box_by_id,
        "groups": {g["id"]: g for g in groups},
    }


def _load_page_texts(out_root: Path, pages: list[str]) -> tuple[dict, dict, dict]:
    """Rebuild page texts / directions / dropped boxes from disk.

    Derives exactly what the OCR stage produced: readings come from
    ``ocr_readings.json`` and a box missing from the cache was read empty
    twice (false selection) and is dropped -- the two-pass check is
    already folded into the cache.  Returns
    ``(page_texts, page_directions, dropped_boxes)`` for pages that have a
    dataset; pages without one are omitted.
    """
    readings = _load_readings_cache(out_root)
    dataset_dir = out_root / "dataset"
    units: dict[str, dict] = {}
    for page in pages:
        unit = _page_units(dataset_dir, page)
        if unit is not None:
            units[page] = unit
    dropped: dict[str, set[str]] = {}
    for page, unit in units.items():
        dropped[page] = {
            box_id for box_id in unit["blocks"]
            if f"{page}:box:{box_id}" not in readings
        }
    texts, directions = build_page_texts(units, readings, dropped)
    return texts, directions, dropped


def run_manga(
    crops: list[dict],
    out_root: Path,
    progress: "ProgressFile | None" = None,
    *,
    ai_ocr_url: str = "",
    ai_ocr_model: str = "",
    ai_ocr_api_key: str = "",
    ai_ocr_workers: int = AI_OCR_DEFAULT_WORKERS,
    stage2_engine: str = "manga-ocr",
) -> tuple[dict, set[str]]:
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
    fallback_ids: set[str] = set()
    if stage2_engine == "ai-ocr":
        if not (ai_ocr_url and ai_ocr_model and ai_ocr_api_key):
            raise RuntimeError(
                "--stage2-engine ai-ocr 需要配置 AI OCR："
                "--ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key"
            )
        if misses:
            from manga_localizer.ai_ocr import ai_ocr_readings

            readings = ai_ocr_readings(
                misses, ai_ocr_url, ai_ocr_model, ai_ocr_api_key,
                progress=progress, max_workers=ai_ocr_workers,
            )
            # 失败的框回退 **manga-ocr**（不是 Sakura —— Sakura 是翻译层）。
            # 视觉模型读不出的往往是风格化小字，本地 OCR 至少还能给出候选，
            # 由调用方按"可疑读数"规则决定要不要用。
            failed = [
                crop for crop in misses
                if not str(readings.get(crop["id"]) or "").strip()
            ]
            if failed:
                if progress is not None:
                    progress.log(f"ai-ocr 失败 {len(failed)} 框 → 回退 manga-ocr")
                readings.update(_run_manga_subprocess(failed, out_root, progress))
                fallback_ids = {crop["id"] for crop in failed}
        else:
            # all crops cached: nothing to read, even in forced ai-ocr mode
            readings = {}
    elif misses and (ai_ocr_url and ai_ocr_model and ai_ocr_api_key):
        from manga_localizer.ai_ocr import ai_ocr_readings

        readings = ai_ocr_readings(
            misses, ai_ocr_url, ai_ocr_model, ai_ocr_api_key,
            progress=progress, max_workers=ai_ocr_workers,
        )
    elif misses:
        readings = _run_manga_subprocess(misses, out_root, progress)
    else:
        readings = {}
    # 说明性输出不是读数：'图片中没有任何日文文本' 这类描述若当读数用，会一路
    # 翻成中文塞进原框（实测两个岛）。命中即视为空 → 回退 manga-ocr；仍读不出
    # 就按假选框丢弃、保留原图。
    dropped_commentary = [
        key for key, value in readings.items() if is_ocr_commentary(str(value))
    ]
    for key in dropped_commentary:
        del readings[key]
    if dropped_commentary and progress is not None:
        progress.log(f"丢弃 {len(dropped_commentary)} 条 OCR 说明性输出（不是文字）")
    _store_readings_cache(out_root, readings)
    result = {}
    for crop in crops:
        value = str(readings.get(crop["id"]) or cached.get(crop["id"]) or "")
        result[crop["id"]] = "" if is_ocr_commentary(value) else value
    return result, fallback_ids


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
    python = _venv_python()
    env = dict(__import__("os").environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if env.get("MANGA_OCR_CPU") or str(env.get("MANGA_OCR_GPU", "")).lower() in {
        "0", "false", "no"
    }:
        # 唯一需要 CPU 的场景是**本地 Sakura**：它按设计吃满 8GB 卡的显存，
        # 两者同时驻留会让 llama_decode 失败（返回 -1 / logits assert）。
        # 用远程 AI 翻译（DeepSeek 等）时 Sakura 不在跑，GPU 空闲，默认就用。
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.Popen(
        [str(python), "-c", script, str(crops_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=False, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
    gap_ratio: float = 0.30,
    overlap_ratio: float = 0.40,
    no_merge_ids: Sequence[str] = (),
) -> list[tuple[tuple[int, int, int, int], list[str]]]:
    """Group blue boxes (image-space) into OCR recognition units.

    A box is vertical when its height >= width, horizontal otherwise.
    Boxes NEVER merge along the reading direction (Photos' blue boxes are
    the trusted atomic units); they only merge side-by-side on the
    PERPENDICULAR axis when they are close AND aligned.

    **合并判据用原始框几何**（只有排布事实，不含 OCR 用的扩边量）；*pad*
    只用于**组的外接矩形**（= 成员框扩边后的并集，下游据此决定填黑/擦字/
    嵌字的范围）。顺序即：筛框 → 编组（原始框）→ 取扩框外接矩形。

    Merge test on the raw boxes -- only the text layout, never the OCR
    padding (padding is an engineering allowance, not a layout fact):

    * two VERTICAL boxes merge when the raw horizontal gap is below
      30% of the smaller box's width AND their vertical ranges overlap by
      more than 40% of the LARGER box's height;
    * two HORIZONTAL boxes merge when the raw vertical gap is below
      30% of the smaller box's height AND their horizontal ranges overlap
      by more than 40% of the LARGER box's width;
    * mixed-orientation pairs never merge.

    (30% on raw geometry replaces the earlier "20% on padded geometry":
    padding used to add ~2*pad px to every gap -- a large constant
    relative to small boxes.  The raw form is scale-invariant.  Measured
    on 图源样张 (532 boxes): 0.6/0.5/0.4/0.3/0.2 -> 261/265/266/271/274
    groups; lower ratios split multi-column sentences apart, and 0.3 was
    chosen by review of the affected pages.)

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

    # 合并判据用**原始框**几何：编组回答的是"这些框是不是同一个文本单位"，
    # 应该只看文字本身的排布，不该被"给 OCR 的扩边量"影响（扩边是工程补偿，
    # 不是排版事实）。顺序因此是：筛框 → 编组（原始框）→ 再按扩框取外接矩形。
    locked = {str(value) for value in no_merge_ids}
    for i in range(len(ordered)):
        a = ordered[i]
        if locked and str(a.get("id") or "") in locked:
            # 过大单框：各成一岛，绝不与相邻列合并（见 oversized_box_ids）
            continue
        ax1, ay1, ax2, ay2 = a["image_bbox"]
        a_vertical = (ay2 - ay1) >= (ax2 - ax1)
        pa = (ax1, ay1, ax2, ay2)
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if locked and str(b.get("id") or "") in locked:
                continue
            bx1, by1, bx2, by2 = b["image_bbox"]
            b_vertical = (by2 - by1) >= (bx2 - bx1)
            if a_vertical != b_vertical:
                continue
            pb = (bx1, by1, bx2, by2)
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
        # 组的外接矩形 = 成员框**扩边后**的并集：编组本身按原始框判，但下游
        # 拿这个矩形是要"多少像素"（填黑/擦字/嵌字的范围都在它之内）
        x1 = min(box["image_bbox"][0] - pad for box in members)
        y1 = min(box["image_bbox"][1] - pad for box in members)
        x2 = max(box["image_bbox"][2] + pad for box in members)
        y2 = max(box["image_bbox"][3] + pad for box in members)
        result.append(((x1, y1, x2, y2), member_ids))
    return sorted(result, key=lambda item: (item[0][1], -item[0][2]))


def _rects_intersect(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _yield_step(small, big, limit: float, gap: int = 1):
    """把 small 推离 big 所需的最小位移（两轴取小），|位移| 受 limit 约束。"""
    candidates = []
    for dx in ((big[0] - gap) - small[2], (big[2] + gap) - small[0]):
        if abs(dx) <= limit:
            candidates.append((abs(dx), dx, 0))
    for dy in ((big[1] - gap) - small[3], (big[3] + gap) - small[1]):
        if abs(dy) <= limit:
            candidates.append((abs(dy), 0, dy))
    if not candidates:
        return None
    _, dx, dy = min(candidates)
    return dx, dy


def _shifted(rect, shift):
    return (rect[0] + shift[0], rect[1] + shift[1],
            rect[2] + shift[0], rect[3] + shift[1])


def _resolve_group_yields(rects, *, limit_ratio: float = 0.5,
                          gap: int = 1) -> list[list[int]]:
    """小组让位：组矩形相交时把**面积小**的组推开，返回每组的位移。

    组矩形（成员框并集 ±bbox_pad）就是嵌字的布局边界，两框相交即两段译文
    会互相干扰。规则：只对更大的组让位；沿"移动量最小"的轴推开到不相交
    （四周各留 gap）；位移上限 = limit_ratio × 自身较短边，避免挪飞。
    推不开（上限内无法分离）就保持原位——宁可轻微重叠也不把文字甩到远处。
    """
    shifts = [[0, 0] for _ in rects]
    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rects]
    order = sorted(range(len(rects)), key=lambda index: -areas[index])
    for rank, index in enumerate(order):
        for bigger in order[:rank]:
            current = _shifted(rects[index], shifts[index])
            target = _shifted(rects[bigger], shifts[bigger])
            if not _rects_intersect(current, target):
                continue
            limit = limit_ratio * min(current[2] - current[0], current[3] - current[1])
            step = _yield_step(current, target, limit, gap)
            if step is None:
                continue
            shifts[index][0] += step[0]
            shifts[index][1] += step[1]
    return shifts


def build_bluebox_dataset(
    manifest_path, page_dir, source_path, min_island_area: float = 800.0, pad: int = 0,
    drop_large_boxes: bool = True,
    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE,
) -> list[dict]:
    """islands.json purely from the Photos blue boxes (original image space).

    No highlight-island detection: the atomic blue boxes are re-grouped by
    the two-axis gap/overlap rule (``_group_boxes_by_overlap``) -- every
    group is one island, padded ``pad`` px, and its own independent bbox.

    **两条框来源共用同一套过滤**（`manga_localizer.thresholds`）：整页大框
    （面积 > 20% 页）在采集期已丢；这里再丢**大号拟音/效果字**（短边 >
    2.0×同方向基准 且 > 页短边 10%）。放在共用阶段而非采集期，Photos 与
    Paddle 两条路径才会得到同一结果。
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
    # 共用的大拟音/效果字过滤（Photos 与 Paddle 同一规则；见 thresholds）
    if drop_large_boxes and blocks:
        atomic_boxes = [b for b in blocks if not b.get("page_scale")]
        if len(atomic_boxes) >= 3:
            kept_boxes, dropped_large = _filter_large_sfx(
                [list(b["image_bbox"]) for b in atomic_boxes], image_w, image_h,
                ratio=sfx_min_side_ratio, page_floor=sfx_min_side_page,
            )
            if dropped_large:
                keep_set = {tuple(box) for box in kept_boxes}
                removed = {
                    b["id"] for b in atomic_boxes
                    if tuple(b["image_bbox"]) not in keep_set
                }
                blocks = [
                    b for b in blocks
                    if b.get("page_scale") or b["id"] not in removed
                ]
                print(
                    f"page {Path(page_dir).name}: 过滤 {len(dropped_large)} 个大拟音/"
                    f"效果字框（短边 > {sfx_min_side_ratio}×同方向基准 且 > "
                    f"{sfx_min_side_page:.0%} 页短边）"
                )

    atomic_ids = {item["id"] for item in blocks if not item.get("page_scale")}
    # 过大单框（短边 ≫ 同方向基准，大概率拟声/效果字）不参与合并：并进竖排组会
    # 把容器和字号一起撑大（实测页 7/11/19），而且它本来就是独立的超大文字。
    oversized = oversized_box_ids(
        [item for item in blocks if not item.get("page_scale")],
        ratio=sfx_min_side_ratio,
        page_short_side=float(min(image_w, image_h)),
    )
    if oversized:
        print(
            f"page {Path(page_dir).name}: {len(oversized)} 个大单框不参与合并"
            f"（短边 > {sfx_min_side_ratio}×同方向基准）"
        )
    grouped = _group_boxes_by_overlap(blocks, no_merge_ids=oversized)
    groups = [
        {
            "id": f"selg_{index:03d}",
            "image_bbox": [int(value) for value in bbox],
            "members": member_ids,
        }
        for index, (bbox, member_ids) in enumerate(grouped, 1)
        if member_ids
    ]
    # 小组让位：先算每组的"嵌字布局矩形"（成员框并集 ±bbox_pad），相交时让
    # 面积小的组推开。位移只记进数据集（typeset_shift），**不动成员框本身**——
    # OCR 裁剪与擦字仍以原框为准，只有嵌字的布局框被挪开。
    boxes_by_id = {b["id"]: b for b in blocks}
    source_kind = str(payload.get("capture", {}).get("box_source") or "photos")
    bbox_pad = source_bbox_pad(source_kind)
    group_rects = []
    for group in groups:
        member_boxes = [boxes_by_id[mid]["image_bbox"] for mid in group["members"]
                        if mid in boxes_by_id]
        if not member_boxes:
            group_rects.append(None)
            continue
        group_rects.append([
            min(b[0] for b in member_boxes) - bbox_pad,
            min(b[1] for b in member_boxes) - bbox_pad,
            max(b[2] for b in member_boxes) + bbox_pad,
            max(b[3] for b in member_boxes) + bbox_pad,
        ])
    shiftable = [r for r in group_rects if r is not None]
    shifts = _resolve_group_yields(shiftable)
    shift_iter = iter(shifts)
    group_shifts = []
    for rect in group_rects:
        group_shifts.append([0, 0] if rect is None else next(shift_iter))
    for group, shift in zip(groups, group_shifts):
        if shift[0] or shift[1]:
            group["typeset_shift"] = list(shift)

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
        island_record = {
            "id": f"island_{len(islands) + 1:03d}",
            "display_bbox": [x1, y1, x2, y2],
            "display_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "display_area": int(area),
            "image_bbox": [x1, y1, x2, y2],
            "image_center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "image_area": round(area, 1),
            "from_selection_group": True,
        }
        shift = group.get("typeset_shift")
        if shift:
            # 让位位移：stage2 据此把**布局框**挪开（擦字范围会一并覆盖让位处）
            island_record["typeset_shift"] = list(shift)
        islands.append(island_record)
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
                # 框来源决定字号上限的补偿量（见 build_union_overrides）：
                # Photos 蓝框紧贴笔画（上限 = 框宽+12），Paddle 检测框自带边距
                # （上限 = 框宽+8，各减 2px/边抵消松边）
                "box_source": str(payload.get("capture", {}).get("box_source") or "photos"),
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
    ai_ocr_workers: int = AI_OCR_DEFAULT_WORKERS,
    stage2_engine: str = "manga-ocr",
    box_source: str = "paddle",
    drop_large_boxes: bool = True,
    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE,
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
    sfx_glossary_min_count: int = 2,
    sfx_max_font_size: int | None = None,
    short_source_max_chars: int = SHORT_SOURCE_MAX_CHARS,
    page_font_ceiling_ratio: float = PAGE_FONT_CEILING_RATIO,
    progress: "ProgressFile | None" = None,
    overwrite: bool = False,
) -> bool:
    """Full stage 2: dataset -> OCR -> translate -> typeset -> PDF.

    Translation prefers the AI (OpenAI-compatible, e.g. DeepSeek) whole-page
    batch; islands whose numbered line is missing/malformed fall back to
    the local Sakura endpoint.  Without an API key everything goes through
    local Sakura.  ``ai_ocr_*`` enable an OpenAI-compatible vision model to
    read the blue-box crops instead of manga-ocr.  ``resume`` skips pages
    whose typeset output already exists.  ``no_pdf`` skips the final PDF
    merge (typeset images only).  ``overwrite`` purges the selected pages'
    dataset + cache keys first, so every stage redoes its work from
    scratch.  ``progress`` publishes run.json.

    The same stages are exposed individually through ``--stage``
    (``_run_stage2_single``); this function is their composition and
    shares their implementations, so full runs and split runs produce
    identical artifacts.
    """
    if progress is None:
        from manga_localizer.progress import ProgressFile

        progress = ProgressFile(out_root)

    active_pages = _stage2_dataset(
        images, out_root, pages, progress, overwrite=overwrite,
        box_source=box_source,
        drop_large_boxes=drop_large_boxes,
        sfx_min_side_ratio=sfx_min_side_ratio,
        sfx_min_side_page=sfx_min_side_page,
    )
    page_texts, page_directions, dropped_boxes = _stage2_ocr(
        out_root, active_pages, progress=progress, overwrite=overwrite,
        ai_ocr_url=ai_ocr_url, ai_ocr_model=ai_ocr_model,
        ai_ocr_api_key=ai_ocr_api_key, ai_ocr_workers=ai_ocr_workers,
        stage2_engine=stage2_engine,
    )
    translations = _stage2_translate(
        out_root, active_pages, page_texts, progress=progress,
        overwrite=overwrite, ai_url=ai_url, ai_model=ai_model,
        ai_api_key=ai_api_key, max_workers=max_workers,
        sfx_glossary_min_count=sfx_glossary_min_count,
    )
    return _stage2_typeset(
        out_root, images, pages, active_pages, translations,
        page_directions, dropped_boxes, progress=progress, resume=resume,
        no_pdf=no_pdf, min_font_size=min_font_size,
        sfx_sources=page_texts, sfx_max_font_size=sfx_max_font_size,
        short_source_max_chars=short_source_max_chars,
        sfx_min_side_ratio=sfx_min_side_ratio,
        page_font_ceiling_ratio=page_font_ceiling_ratio,
        font_name=font_name,
        stroke_divisor=stroke_divisor, row_gap_divisor=row_gap_divisor,
        column_gap_divisor=column_gap_divisor,
        tracking_divisor=tracking_divisor, text_color=text_color,
        stroke_color=stroke_color, background_color=background_color,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# capture / check 的目录布局
#   capture/{source}/{page}/capture.json     source = paddle | photos
#   check/{source}/{page}.bluebox_check.png
#   dataset/{page}/...   typeset/{page}.typeset.png   （单套，由当前来源生成）
# 两条框来源**各自独立存放**，切来源不会覆盖对方的采集结果；dataset/typeset
# 只有一套，由本次运行选用的来源生成。所有读写都走下面这几个解析函数。
# ---------------------------------------------------------------------------

CAPTURE_SOURCES = ("paddle", "photos")


def _capture_source_dir(out_root: Path, source: str) -> Path:
    return Path(out_root) / "capture" / source


def _check_source_dir(out_root: Path, source: str) -> Path:
    return Path(out_root) / "check" / source


def _capture_source_dirs(out_root: Path) -> list[tuple[str, Path]]:
    """已存在的来源目录（按 CAPTURE_SOURCES 顺序，paddle 在前）。"""
    root = Path(out_root) / "capture"
    return [
        (source, root / source)
        for source in CAPTURE_SOURCES
        if (root / source).is_dir()
    ]


def _find_capture_page(
    out_root: Path, page: str, preferred: str = ""
) -> tuple[str, Path] | None:
    """定位某页的 capture.json：优先 ``preferred`` 来源，其次另一来源。

    返回 ``(source, manifest_path)``；两个来源都没有则 ``None``。
    """
    order = [s for s in CAPTURE_SOURCES if s == preferred]
    order += [s for s in CAPTURE_SOURCES if s != preferred]
    for source in order:
        manifest = _capture_source_dir(out_root, source) / page / "capture.json"
        if manifest.is_file():
            return source, manifest
    return None


def _page_sort_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name.lower())


def _iter_capture_pages(out_root: Path) -> list[tuple[str, str, Path]]:
    """遍历所有来源下的页，返回 ``(source, page, manifest_path)``。"""
    rows = []
    for source, base in _capture_source_dirs(out_root):
        for page_dir in sorted(base.iterdir(), key=lambda d: _page_sort_key(d.name)):
            manifest = page_dir / "capture.json"
            if page_dir.is_dir() and manifest.is_file():
                rows.append((source, page_dir.name, manifest))
    return rows


def _load_capture(manifest_path: Path) -> dict | None:
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _mark_review_required(out_root: Path, page: str, reason: str) -> None:

    """把一页标记为"待人工审核"（写进该页**所有来源**的 capture.json）。

    用于**筛框之后**才发现无框可用的页：这类页在采集期是有框的，所以不会被
    采集期的审核标记覆盖到，必须在这里补记，否则会被静默跳过。
    两个来源都写：审核状态与"框来自哪条路径"无关，审核页读哪个来源都能看到。
    """
    for source in CAPTURE_SOURCES:
        manifest_path = _capture_source_dir(out_root, source) / page / "capture.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_capture(manifest_path)
        if manifest is None:
            continue
        capture = manifest.setdefault("capture", {})
        capture["selection_review_required"] = True
        capture["selection_review_reason"] = reason
        capture["selection_empty_after_filter"] = True
        try:
            temporary = manifest_path.with_name(manifest_path.name + ".tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(manifest_path)
        except OSError:
            pass


def _stage2_dataset(images: Path, out_root: Path, pages: list[str],
                    progress: "ProgressFile", *,
                    overwrite: bool = False,
                    box_source: str = "paddle",
                    drop_large_boxes: bool = True,
                    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
                    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE) -> list[str]:
    """Build islands from capture (one island per blue-box selection
    group).  Returns the pages that ended up with a dataset (active)."""
    from island_extract import build_fallback_dataset

    dataset_dir = out_root / "dataset"
    if overwrite:
        _purge_page_artifacts(out_root, pages, dataset=True,
                              readings=True, translations=True)
    fallback = _fallback_pages_for(out_root, pages)
    # Islands come ONLY from the Photos blue boxes (no highlight-island
    # detection): one island per selection group, original image space.
    progress.set_stage("dataset", total=len(pages))
    for index, page in enumerate(pages):
        progress.progress(index, item=f"page {page}")
        page_dir = dataset_dir / page
        # 本次运行选用的来源优先；该来源没有这页时回退另一来源（并说明），
        # 这样"用 paddle 跑 stage2、但只有 photos 采集"也能继续。
        found = _find_capture_page(out_root, page, preferred=box_source)
        if found is None:
            continue
        used_source, manifest_path = found
        if used_source != box_source:
            print(
                f"{page}: 没有 {box_source} 采集结果，改用 {used_source} 采的框"
            )
        islands_path = page_dir / "islands.json"
        if islands_path.is_file():
            # A stale dataset is the classic silent bug: the page was
            # RE-captured (new capture.json with different blue boxes)
            # but the old islands/selection_blocks were reused, so the
            # OCR crops no longer match the check image.  Only reuse the
            # dataset when it is NEWER than the capture manifest; a
            # re-captured page must rebuild its islands from scratch.
            try:
                manifest_mtime = manifest_path.stat().st_mtime
                islands_mtime = islands_path.stat().st_mtime
            except OSError:
                manifest_mtime = islands_mtime = 0.0
            if islands_mtime >= manifest_mtime:
                continue
            import shutil

            shutil.rmtree(page_dir, ignore_errors=True)
            print(
                f"{page}: capture re-done after dataset, rebuilding islands "
                "(check image and OCR now share the same blue boxes)"
            )
        source = _source_image_for(images, page)
        if source is None:
            continue
        if page in fallback:
            with Image.open(source) as probe:
                image_size = probe.size
            build_fallback_dataset(source, page_dir, image_size)
            continue
        if not manifest_path.is_file():
            continue
        try:
            build_bluebox_dataset(
                manifest_path, page_dir, source,
                drop_large_boxes=drop_large_boxes,
                sfx_min_side_ratio=sfx_min_side_ratio,
                sfx_min_side_page=sfx_min_side_page,
            )
        except ValueError as error:
            # No usable blue boxes: nothing to process, the page stays
            # original (kept out of the OCR/translate/typeset loops).
            # **但要把这页送进审核页**：筛框（微框/整页大框/大拟音）之后可能
            # 一个框都不剩，静默跳过会让人以为"这页没问题"。标记审核，人工
            # 在审核页可选择"重新识别（Paddle）"或"重采（Photos）"。
            print(f"{page}: {error}")
            _mark_review_required(
                out_root, page, f"筛框后无框（{error}）",
            )

    return [
        page for page in pages
        if (dataset_dir / page / "islands.json").is_file()
    ]


def _stage2_ocr(out_root: Path, pages: list[str], *,
                progress: "ProgressFile", overwrite: bool = False,
                ai_ocr_url: str = "", ai_ocr_model: str = "",
                ai_ocr_api_key: str = "",
                ai_ocr_workers: int = AI_OCR_DEFAULT_WORKERS,
                stage2_engine: str = "manga-ocr") -> tuple[dict, dict, dict]:
    """OCR the blue boxes (manga/AI) and derive page texts/directions.

    Reads ONLY blue boxes, cropped from the whole-page masked image so
    neighbour text can never leak into a crop.  A box read empty twice is
    a false selection and is dropped.  Returns
    ``(page_texts, page_directions, dropped_boxes)``.
    """
    if overwrite:
        # Full re-read: purge the selected pages' readings/translations so
        # a re-captured page can never mix fresh blue boxes with stale OCR.
        _purge_page_artifacts(out_root, pages, readings=True, translations=True)
    dataset_dir = out_root / "dataset"
    # manga reads: blue boxes only, from the blue-box masked image so
    # neighbour text can never leak into a crop
    crops, units = [], {}
    for page in pages:
        unit = _page_units(dataset_dir, page)
        if unit is None:
            continue
        page_dir = dataset_dir / page
        blocks = list(unit["blocks"].values())
        source = str(page_dir / "original.jpg")
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
        units[page] = unit
    progress.set_stage("ocr", total=len(crops))
    progress.sub_progress(0, len(crops), "manga-ocr")
    readings, fallback_ids = run_manga(
        crops, out_root, progress=progress,
        ai_ocr_url=ai_ocr_url, ai_ocr_model=ai_ocr_model,
        ai_ocr_api_key=ai_ocr_api_key, ai_ocr_workers=ai_ocr_workers,
        stage2_engine=stage2_engine,
    )

    # Second pass: reread the boxes manga-ocr read empty once.  A box that
    # stays empty is a false selection -- dropped everywhere (never joined
    # into a sentence, never erased, never typeset).
    dropped_boxes: dict[str, set[str]] = {page: set() for page in units}
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
        reread_readings, reread_fallbacks = run_manga(
            rereads, out_root, progress=progress,
            ai_ocr_url=ai_ocr_url, ai_ocr_model=ai_ocr_model,
            ai_ocr_api_key=ai_ocr_api_key, ai_ocr_workers=ai_ocr_workers,
            stage2_engine=stage2_engine,
        )
        fallback_ids |= reread_fallbacks
        for crop in rereads:
            page, _, rest = crop["id"].partition(":box:")
            box_id = rest.removesuffix(":reread")
            key = f"{page}:box:{box_id}"
            reread_text = reread_readings.get(crop["id"], "")
            if reread_text:
                readings[key] = reread_text
            else:
                dropped_boxes[page].add(box_id)

    dropped_boxes = _demote_suspicious_fallbacks(
        readings, dropped_boxes, fallback_ids, progress,
    )
    page_texts, page_directions = build_page_texts(units, readings, dropped_boxes)
    return page_texts, page_directions, dropped_boxes


def _demote_suspicious_fallbacks(
    readings: dict[str, str],
    dropped_boxes: dict[str, set[str]],
    fallback_ids: set[str],
    progress: "ProgressFile",
) -> dict[str, set[str]]:
    """AI OCR 失败回退 manga-ocr 后，可疑读数一律不翻译（保留原文）。

    AI 视觉模型和图内文字识别模型都没把握的框，多半是风格化拟声词/语气词
    （``ぎゅっ♡`` 被 manga-ocr 读成 ``どうし`` 再译成"为什么"）。这种读数
    进翻译只会产出语义全错的中文并擦掉原文，代价比保留日文大，所以按
    "假选框"处理：不翻译、不擦字、不嵌字，页面保持原样。
    """
    from manga_localizer.text_rules import suspicious_short_kana

    demoted = 0
    for crop_id in sorted(fallback_ids):
        page, _, rest = crop_id.partition(":box:")
        box_id = rest.removesuffix(":reread")
        if box_id in dropped_boxes.get(page, set()):
            continue
        if suspicious_short_kana(readings.get(f"{page}:box:{box_id}", "")):
            dropped_boxes.setdefault(page, set()).add(box_id)
            demoted += 1
    if demoted:
        progress.log(f"回退读数可疑，保留原文 {demoted} 框（不翻译）")
    return dropped_boxes


def _translate_one_page(page: str, texts: dict[str, str], *,
                        dataset_dir: Path, glossary, cache, batch_cache,
                        ai_url: str, ai_model: str,
                        ai_api_key: str) -> dict[str, str]:
    """Whole-page translate: AI batch preferred, local Sakura fallback."""
    from manga_localizer.island_contract import load_island_dataset
    from manga_localizer.island_translate import (
        IslandTranslateConfig,
        translate_island_page,
        translate_page_batch,
    )

    dataset = load_island_dataset(dataset_dir / page / "islands.json")
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


def _stage2_translate(out_root: Path, pages: list[str],
                      texts: dict[str, dict[str, str]], *,
                      progress: "ProgressFile", overwrite: bool = False,
                      ai_url: str = "https://api.deepseek.com/v1",
                      ai_model: str = "deepseek-chat", ai_api_key: str = "",
                      max_workers: int = 1,
                      sfx_glossary_min_count: int = 2) -> dict[str, dict[str, str]]:
    """Translate every given page and persist ``translations.json``.

    AI whole-page batch first, local Sakura fallback; the per-page result
    is written atomically at the end (re-running is free when the caches
    hit).  Returns ``{page: {island_id: text}}``.
    """
    from concurrent.futures import ThreadPoolExecutor

    from manga_localizer.island_translate import (
        IslandTranslationCache,
        PageBatchCache,
    )
    from manga_localizer.translator import load_glossary

    if overwrite:
        # Discard the old translations of the selected pages: they must be
        # re-derived (cache hits return the same text, so this is cheap).
        _purge_page_artifacts(out_root, pages, readings=False, translations=True)
    glossary = load_glossary(ROOT / "config" / "glossary.json")
    cache = IslandTranslationCache(out_root / "translation_cache")
    batch_cache = PageBatchCache(out_root / "translation_batch_cache")
    dataset_dir = out_root / "dataset"
    translations: dict[str, dict[str, str]] = {}
    failed: list[str] = []

    def _translate(page: str) -> tuple[str, dict[str, str] | None]:
        try:
            return page, _translate_one_page(
                page, texts[page], dataset_dir=dataset_dir,
                glossary=glossary, cache=cache, batch_cache=batch_cache,
                ai_url=ai_url, ai_model=ai_model, ai_api_key=ai_api_key,
            )
        except Exception as error:  # noqa: BLE001 - per-page isolation
            print(f"!! translate {page} FAILED: {error}")
            progress.log(f"translate {page} failed: {error}")
            return page, None

    progress.set_stage("translate", total=len(pages))
    results: list[tuple[str, dict[str, str] | None]] = []
    workers = max(1, int(max_workers))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_translate, page) for page in pages]
            for future in futures:
                results.append(future.result())
    else:
        results = [_translate(page) for page in pages]
    for page, result in results:
        if result is not None:
            translations[page] = result
        else:
            failed.append(page)
    _store_translations(out_root, translations)
    if failed:
        print(f"!! 翻译失败页（未写入 translations.json）: {', '.join(failed)}")
        backup = out_root / "translations.json.bak"
        if backup.is_file():
            print(
                f"!! 旧译文仍在 {backup.name}：本次失败的页没有新译文，"
                f"要回滚就把 {backup.name} 改名回 translations.json"
            )
    _sync_sfx_glossary(out_root, texts, translations, progress,
                       min_count=sfx_glossary_min_count)
    return translations


def _sync_sfx_glossary(out_root: Path, texts: dict[str, dict[str, str]],
                       translations: dict[str, dict[str, str]],
                       progress: "ProgressFile",
                       *, min_count: int = 2) -> dict:
    """把本次译好、且译法稳定的拟声词收录进 ``config/glossary.json``。

    收录后术语表 sha 变化 → 页级翻译缓存签名失效 → 下次运行（含全书重跑）
    统一按收录的译法走。失败（读不了/写不了）不影响翻译结果。
    """
    from manga_localizer.glossary_auto import (
        collect_observations,
        empty_report,
        sync_sfx_terms,
    )

    if min_count <= 0:
        return empty_report()
    observations = collect_observations(texts, translations)
    if not observations:
        return empty_report()
    try:
        report = sync_sfx_terms(
            ROOT / "config" / "glossary.json", observations, min_count=min_count,
        )
    except OSError as error:  # noqa: BLE001 - 术语库写失败不该影响出图
        progress.log(f"拟声词术语库更新失败: {error}")
        return empty_report()
    for key in ("added", "updated", "removed"):
        items = report.get(key) or []
        if items:
            progress.log(f"拟声词术语库 {key}: {'、'.join(items)}")
    if report.get("removed"):
        # 译法冲突 → 撤出，避免锁死可能是错的译法
        print(f"  拟声词译法不稳定，已撤出术语库: {'、'.join(report['removed'])}")
    if report.get("added") or report.get("updated"):
        print("  拟声词已收录进术语库（下次运行全书统一）："
              + "、".join(report.get("added", []) + report.get("updated", [])))
    return report

def _stage2_typeset(out_root: Path, images: Path, pages: list[str],
                    active_pages: list[str],
                    translations: dict[str, dict[str, str]],
                    page_directions: dict, dropped_boxes: dict, *,
                    progress: "ProgressFile", resume: bool = False,
                    no_pdf: bool = False, min_font_size: int | None = None,
                    sfx_sources: dict[str, dict[str, str]] | None = None,
                    sfx_max_font_size: int | None = None,
                    short_source_max_chars: int = SHORT_SOURCE_MAX_CHARS,
                    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
                    page_font_ceiling_ratio: float = PAGE_FONT_CEILING_RATIO,
                    font_name: str = "", stroke_divisor: int | None = None,
                    row_gap_divisor: int | None = None,
                    column_gap_divisor: int | None = None,
                    tracking_divisor: int | None = None,
                    text_color: str = "", stroke_color: str = "",
                    background_color: str = "", max_workers: int = 1) -> bool:
    """Erase + typeset every translated page, then assemble the PDF.

    Pages without translations (never translated, or translate failed) are
    kept as the original image in the PDF -- unless a stale typeset.png
    already exists, which is kept as-is.
    """
    import shutil
    import threading
    from concurrent.futures import ThreadPoolExecutor

    dataset_dir = out_root / "dataset"
    typeset_dir = out_root / "typeset"
    typeset_dir.mkdir(parents=True, exist_ok=True)
    parsed_text = _parse_color(text_color)
    parsed_stroke = _parse_color(stroke_color)
    parsed_background = _parse_color(background_color)

    def _worker(page: str) -> bool:
        """One page: typeset with its translation.  True on failure."""
        translations_page = translations.get(page)
        if not translations_page:
            # Standalone typeset on a page that was never translated: the
            # page keeps its original (listed with the untouched pages).
            progress.log(f"skip {page}: no translations (run translate first)")
            return False
        out_png = typeset_dir / f"{page}.typeset.png"
        if resume and out_png.is_file():
            progress.log(f"resume: skip {page} (typeset exists)")
            return False
        try:
            erase_and_typeset_page_with(
                dataset_dir, typeset_dir, page, translations_page,
                page_directions.get(page), dropped_boxes.get(page, set()),
                min_font_size=min_font_size,
                sources=(sfx_sources or {}).get(page),
                sfx_max_font_size=sfx_max_font_size,
                short_source_max_chars=short_source_max_chars,
                sfx_min_side_ratio=sfx_min_side_ratio,
                font_name=font_name,
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

    # Every page without a fresh typeset image keeps the original in the
    # PDF: untouched pages (no blue boxes), never-translated pages, and
    # failed pages that have no stale image to fall back on.
    kept_original: list[str] = []
    for page in pages:
        out_png = typeset_dir / f"{page}.typeset.png"
        if out_png.is_file():
            continue
        source = dataset_dir / page / "original.jpg"
        if not source.is_file():
            source = _source_image_for(images, page)
        if source is not None:
            shutil.copyfile(source, out_png)
            kept_original.append(page)
    if kept_original:
        print("")
        print("=" * 60)
        print(f"!! 无嵌字结果、PDF 中保持原图的页: {', '.join(kept_original)}")
        print("=" * 60)

    # Assemble the PDF from EVERY existing typeset image in the directory
    # (natural page order), not just the pages of this run: when the user
    # re-processes a subset ("处理页 41,55,58"), the PDF must still contain
    # the whole book -- the other pages' typeset files are already there.
    typeset_files = sorted(
        (
            path for path in typeset_dir.glob("*.typeset.png")
            if path.is_file()
        ),
        key=lambda path: (
            (0, int(path.stem))
            if path.stem.isdigit() else (1, path.stem.lower())
        ),
    )
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


def _run_stage2_single(stage: str, images: Path, out_root: Path,
                       pages: list[str], kwargs: dict, *,
                       overwrite: bool) -> bool:
    """Run exactly one stage of stage 2 on ``pages`` (--stage).

    Pages whose prerequisite artifacts are missing are skipped with a
    warning; ``kwargs`` holds the shared stage-2 options.
    """
    from manga_localizer.progress import ProgressFile

    progress = ProgressFile(out_root)
    dataset_dir = out_root / "dataset"
    if stage == "ocr":
        active_pages = _stage2_dataset(
            images, out_root, pages, progress, overwrite=overwrite,
            box_source=kwargs.get("box_source", "paddle"),
            drop_large_boxes=kwargs.get("drop_large_boxes", True),
            sfx_min_side_ratio=kwargs.get("sfx_min_side_ratio", SFX_MIN_SIDE_RATIO),
        page_font_ceiling_ratio=kwargs.get(
            "page_font_ceiling_ratio", PAGE_FONT_CEILING_RATIO),
            sfx_min_side_page=kwargs.get("sfx_min_side_page", SFX_MIN_SIDE_PAGE),
        )
        if not active_pages:
            print("no pages with a dataset to OCR (capture first)")
            progress.done(ok=True)
            return True
        _stage2_ocr(
            out_root, active_pages, progress=progress, overwrite=overwrite,
            ai_ocr_url=kwargs["ai_ocr_url"],
            ai_ocr_model=kwargs["ai_ocr_model"],
            ai_ocr_api_key=kwargs["ai_ocr_api_key"],
            stage2_engine=kwargs.get("stage2_engine", "manga-ocr"),
        )
        progress.done(ok=True)
        return True
    if stage == "translate":
        dataset_pages = [
            page for page in pages
            if (dataset_dir / page / "islands.json").is_file()
        ]
        page_texts, _directions, _dropped = _load_page_texts(
            out_root, dataset_pages)
        qualified = [
            page for page in dataset_pages
            if any(page_texts.get(page, {}).values())
        ]
        skipped = sorted(set(dataset_pages) - set(qualified))
        if skipped:
            print("!! " + "=" * 60)
            print(f"!! 无 OCR 结果的页（先跑 --stage ocr）: {', '.join(skipped)}")
            print("!! " + "=" * 60)
        if not qualified:
            print("no pages to translate (no OCR readings)")
            progress.done(ok=True)
            return True
        _stage2_translate(
            out_root, qualified, page_texts, progress=progress,
            overwrite=overwrite, ai_url=kwargs["ai_url"],
            ai_model=kwargs["ai_model"], ai_api_key=kwargs["ai_api_key"],
            max_workers=kwargs["max_workers"],
            sfx_glossary_min_count=kwargs.get("sfx_glossary_min_count", 2),
        )
        progress.done(ok=True)
        return True
    # typeset
    translations = _load_translations(out_root)
    dataset_pages = [
        page for page in pages
        if (dataset_dir / page / "islands.json").is_file()
    ]
    translated = [page for page in dataset_pages if translations.get(page)]
    skipped = sorted(set(dataset_pages) - set(translated))
    if skipped:
        print("!! " + "=" * 60)
        print(f"!! 无翻译结果的页（先跑 --stage translate，将保持原图）: "
              f"{', '.join(skipped)}")
        print("!! " + "=" * 60)
    page_texts, page_directions, dropped_boxes = _load_page_texts(
        out_root, dataset_pages)
    return _stage2_typeset(
        out_root, images, pages, translated, translations,
        page_directions, dropped_boxes, progress=progress,
        resume=kwargs["resume"], no_pdf=kwargs["no_pdf"],
        min_font_size=kwargs["min_font_size"],
        sfx_sources=page_texts,
        sfx_max_font_size=kwargs.get("sfx_max_font_size"),
        short_source_max_chars=kwargs.get(
            "short_source_max_chars", SHORT_SOURCE_MAX_CHARS),
        sfx_min_side_ratio=kwargs.get("sfx_min_side_ratio", SFX_MIN_SIDE_RATIO),
        font_name=kwargs["font_name"],
        stroke_divisor=kwargs["stroke_divisor"],
        row_gap_divisor=kwargs["row_gap_divisor"],
        column_gap_divisor=kwargs["column_gap_divisor"],
        tracking_divisor=kwargs["tracking_divisor"],
        text_color=kwargs["text_color"], stroke_color=kwargs["stroke_color"],
        background_color=kwargs["background_color"],
        max_workers=kwargs["max_workers"],
    )



def _page_manifests(out_root: Path, page: str) -> list[dict]:
    """该页在**所有来源**下的 capture.json（按 CAPTURE_SOURCES 顺序）。"""
    manifests = []
    for source in CAPTURE_SOURCES:
        manifest_path = _capture_source_dir(out_root, source) / page / "capture.json"
        manifest = _load_capture(manifest_path) if manifest_path.is_file() else None
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def _fallback_pages_for(out_root: Path, pages: list[str]) -> set[str]:
    """任一来源用了 manga 保底的页。"""
    fallback = set()
    for page in pages:
        if any(
            m.get("capture", {}).get("fallback_manga")
            for m in _page_manifests(out_root, page)
        ):
            fallback.add(page)
    return fallback


def _review_required_pages(out_root: Path, pages: list[str]) -> list[str]:
    """采集产出无框（三条路径：无蓝色像素 / 无文本 / 筛框后空）的页。

    硬规则：这些页在人工确认（``--confirm-no-boxes``）前**不进下游**。
    任一来源标了待审即算待审——审核状态与框来自哪条路径无关。
    """
    review = []
    for page in pages:
        for manifest in _page_manifests(out_root, page):
            capture = manifest.get("capture", {})
            if capture.get("selection_review_required") and not capture.get(
                "selection_review_confirmed"
            ):
                review.append(page)
                break
    return review


def _confirm_no_boxes(out_root: Path, pages: list[str]) -> None:
    """标记为已人工复核：确实没有框，可以关闭（下游静默跳过）。"""
    for page in pages:
        for source in CAPTURE_SOURCES:
            manifest_path = _capture_source_dir(out_root, source) / page / "capture.json"
            manifest = _load_capture(manifest_path) if manifest_path.is_file() else None
            if manifest is None:
                continue
            capture = manifest.setdefault("capture", {})
            capture["selection_review_confirmed"] = True
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def _skip_pages(out_root: Path, pages: list[str]) -> set[str]:
    """无可用采集的页（黑页 / 无文本 / 采集失败 / 待审）静默或大声跳过。

    **任一来源可用即不跳过**：另一条路径还留着好框，不该因为这条失败就丢掉。
    """
    skipped = set()
    for page in pages:
        manifests = _page_manifests(out_root, page)
        if not manifests:
            continue
        usable = False
        for manifest in manifests:
            capture = manifest.get("capture", {})
            if capture.get("fallback_manga"):
                usable = True
                continue
            if not capture.get("complete"):
                continue
            if (
                capture.get("selection_review_required")
                and not capture.get("selection_review_confirmed")
            ):
                continue
            usable = True
        if not usable:
            skipped.add(page)
    return skipped


def build_union_overrides(dataset, boxes, pad: int = 6,
                          glyph_margin: int | None = None,
                          bbox_pad: int | None = None) -> dict:
    """Per-island blue-box union masks (island-local coords).

    Each island's mask is the union of its blue boxes expanded by *pad* px
    (clamped to the island bbox); ``area`` is the mask pixel count.

    Glyph size caps follow the padded box dimensions: vertical font size
    (glyph width) <= narrowest padded box width, horizontal font size
    (glyph height) <= shortest padded box height.  **These use the
    UNCLAMPED padded size** -- the island-bbox clamp exists only to keep
    the mask in island-local coordinates, and letting it feed the caps
    would make the font size follow the island bbox (a bigger group rect
    would silently enlarge the typeset text).

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
                # 字号上限取**未裁剪**的扩框尺寸：px/py 是给掩膜用的岛内局部
                # 坐标，会被岛 bbox 裁掉边缘；若拿它算字号上限，就会产生
                # "岛 bbox 一变、字号跟着变"的无关耦合（组外接矩形加个 pad
                # 都会把字号顶上去）。字号上限只该反映框本身的尺寸。
                # 字号上限 = 框尺寸 + 固定余量（不随 pad 变）：Photos 蓝框
                # 紧贴笔画用 +12；Paddle 检测框自带约 2px/边余量，用 +5 抵消
                margin = 2 * pad if glyph_margin is None else glyph_margin
                box_w = (x2 - x1) + margin
                box_h = (y2 - y1) + margin
                if min_w is None or box_w < min_w:
                    min_w = box_w
                if min_h is None or box_h < min_h:
                    min_h = box_h
        if not local_boxes:
            continue
        mask = np.zeros((iy2 - iy1, ix2 - ix1), dtype=np.uint8)
        for px1, py1, px2, py2 in local_boxes:
            mask[py1:py2, px1:px2] = 255
        bp = pad if bbox_pad is None else bbox_pad
        raw = [
            (b["image_bbox"][0] - bp, b["image_bbox"][1] - bp,
             b["image_bbox"][2] + bp, b["image_bbox"][3] + bp)
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


# 擦字缓存版本号：擦字算法/配置语义变化时 +1，旧缓存自动失效
ERASE_CACHE_VERSION = 1


def _erase_cache_key(source_sha256: str, extra_regions: dict, min_island_area: float,
                     config) -> str:
    """擦字结果的缓存键：只包含**决定擦字输出**的全部输入。

    原图内容 + 每岛擦字矩形 + 擦字配置 + 面积下限 + 算法版本。任一变化即失效，
    所以改框/换图/改配置都不会串用旧结果。擦字是整条链里最贵的一步
    （实测 12~13s/页，其中 FSR 修复占 ~87%），缓存后重跑嵌字（改排版、改编组、
    重合并 PDF）不必再擦。
    """
    import hashlib

    payload = {
        "version": ERASE_CACHE_VERSION,
        "source_sha256": source_sha256,
        "regions": sorted(
            (str(island_id), [int(v) for v in rect])
            for island_id, rect in (extra_regions or {}).items()
        ),
        "min_island_area": float(min_island_area),
        "config": {
            "style_override": getattr(config, "style_override", "auto"),
            "fill_ring_radius": getattr(config, "fill_ring_radius", None),
            "inpaint_padding": getattr(config, "inpaint_padding", None),
            "mask": str(getattr(config, "mask", "")),
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _erased_image_for(page_dir: Path, source, active_dataset, extra_regions,
                      config, min_island_area: float):
    """取擦字结果：命中缓存直接读，否则擦完写缓存。

    缓存文件与键文件都是**可删的中间产物**：删了就重擦一次，结果一致。
    """
    import hashlib

    from manga_localizer.island_erasure import erase_island_image

    cache_png = page_dir / "erased.png"
    cache_key_file = page_dir / "erased.key"
    source_path = page_dir / "original.jpg"
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        source_sha256 = ""
    key = _erase_cache_key(source_sha256, extra_regions, min_island_area, config)
    if cache_png.is_file() and cache_key_file.is_file():
        try:
            if cache_key_file.read_text(encoding="utf-8").strip() == key:
                with Image.open(cache_png) as cached:
                    if cached.size == source.size:
                        return cached.convert("RGB").copy()
        except (OSError, ValueError):
            pass

    erased = erase_island_image(
        source, active_dataset, config=config, min_island_area=min_island_area,
        extra_regions=extra_regions,
    ).image
    try:
        erased.save(cache_png)
        cache_key_file.write_text(key, encoding="utf-8")
    except OSError:
        pass
    return erased


def erase_and_typeset_page_with(
    dataset_dir, typeset_dir, page, translations, directions, dropped_boxes=None,
    min_font_size: int | None = None,
    sources: dict[str, str] | None = None,
    sfx_max_font_size: int | None = None,
    short_source_max_chars: int = SHORT_SOURCE_MAX_CHARS,
    font_name: str | None = None,
    stroke_divisor: int | None = None,
    row_gap_divisor: int | None = None,
    column_gap_divisor: int | None = None,
    tracking_divisor: int | None = None,
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    background_color: tuple[int, int, int] | None = None,
    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE,
    page_font_ceiling_ratio: float = PAGE_FONT_CEILING_RATIO,
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
    # 字号上限的边距按框来源补偿：Paddle 检测框自带约 2px/边 的余量，
    # 用 +8（pad 4）而非蓝框的 +12（pad 6），抵消松边导致的字号偏大
    _box_source = ""
    try:
        _box_source = str(
            json.loads(selection_json.read_text(encoding="utf-8")).get("box_source") or ""
        )
    except (OSError, ValueError):
        _box_source = ""
    glyph_margin = source_glyph_margin(_box_source)
    bbox_pad = source_bbox_pad(_box_source)
    union_overrides = (
        build_union_overrides(dataset, boxes, pad=6, glyph_margin=glyph_margin,
                              bbox_pad=bbox_pad)
        if boxes else {}
    )
    config = TypesetConfig(
        min_font_size=min_font_size if min_font_size else TypesetConfig.min_font_size,
        sfx_max_font_size=(
            sfx_max_font_size
            if sfx_max_font_size is not None
            else TypesetConfig.sfx_max_font_size
        ),
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
    layout = {}
    for island in dataset.islands:
        box = (union_overrides.get(island.id) or {}).get("bbox")
        if box is None:
            continue
        canonical[island.id] = tuple(int(value) for value in box)
        # 小组让位：数据集里记录的 typeset_shift 只挪**布局框**（嵌字范围），
        # 擦字仍以不挪的 canonical box 为准（两者下面的 extra_regions 都覆盖）
        raw_shift = island.raw.get("typeset_shift") or (0, 0)
        try:
            dx, dy = int(raw_shift[0]), int(raw_shift[1])
        except (TypeError, ValueError, IndexError):
            dx, dy = 0, 0
        base = canonical[island.id]
        layout[island.id] = (
            (base[0] + dx, base[1] + dy, base[2] + dx, base[3] + dy)
            if (dx or dy) else base
        )
    canonical_dataset = dataclasses.replace(
        dataset,
        islands=tuple(
            dataclasses.replace(island, image_bbox=layout.get(island.id, island.image_bbox))
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
    # 擦字范围 = canonical box **再放宽 STROKE_OVERFLOW_MARGIN**：文字描边向外
    # 画，贴着框边时会伸出布局框（裁剪掩膜同样放宽），这一带必须先擦干净，
    # 否则描边会画在原图笔画上。
    _m = STROKE_OVERFLOW_MARGIN
    extra_regions = {
        island_id: (
            canonical[island_id][0] - _m, canonical[island_id][1] - _m,
            canonical[island_id][2] + _m, canonical[island_id][3] + _m,
        )
        for island_id in erase_ids
        if island_id in canonical
    }
    # 让位过的岛：擦字要同时覆盖"原位"和"让位后的布局框"，否则译文会压在
    # 没擦掉的原图笔画上
    for island_id in list(extra_regions):
        moved = layout.get(island_id)
        if not moved or moved == extra_regions[island_id]:
            continue
        base = extra_regions[island_id]
        extra_regions[island_id] = (
            min(base[0], moved[0]), min(base[1], moved[1]),
            max(base[2], moved[2]), max(base[3], moved[3]),
        )
    erased = _erased_image_for(
        dataset_dir / page, source, active_dataset, extra_regions,
        ErasureConfig(), 800.0,
    )
    # 字号封顶的岛：判据用**日文原文**（不是译文）——(a) 拟声词（短假名/符号），
    # (b) 短原文（≤ SHORT_SOURCE_MAX_CHARS 字，默认 6，用户要求"少于 6 个字就
    # 严格限制"）。两者都按 SFX_MAX_FONT_SIZE 封顶：漫画里 2~3 个字的台词，
    # 蓝框往往很大（原文字本来就写得大），字号跟框走会嵌出一个 80~90px 的中文
    # 短句，与整页文字失衡。只封顶不放大。
    capped_islands = {
        island_id for island_id, source_text in (sources or {}).items()
        if is_sfx_like(str(source_text))
        or short_source(str(source_text), short_source_max_chars)
    }
    # 几何判据（不需要原文）：未合并的单框、短边远超同方向平均 → 大概率拟声词
    capped_islands |= oversized_single_box_islands(
        dataset, boxes, ratio=sfx_min_side_ratio,
        page_floor=sfx_min_side_page,
    )
    # 整页字号上限：≤ ratio × 该页常规字号（蓝框短边中位数）。各岛"各自撑满"
    # 会让同页出现 2 倍级差（实测页 7：131px vs 基准 66px），这里按页压回来。
    font_ceiling_px = None
    if page_font_ceiling_ratio and page_font_ceiling_ratio > 0 and boxes:
        import statistics as _statistics

        baseline = float(_statistics.median([
            min(b["image_bbox"][2] - b["image_bbox"][0],
                b["image_bbox"][3] - b["image_bbox"][1])
            for b in boxes
        ]))
        font_ceiling_px = max(1, int(round(page_font_ceiling_ratio * baseline)))
    result = typeset_island_page(
        erased, canonical_dataset, translations, blue_lines, config=config,
        direction_overrides=directions,
        union_overrides=union_overrides,
        capped_islands=capped_islands,
        font_ceiling_px=font_ceiling_px,
    )
    _save_typeset_png(result.image, typeset_dir / f"{page}.typeset.png")
    return result


def _save_typeset_png(image, target: Path) -> None:
    """原子写嵌字结果（临时文件 + 替换），并把人祸型失败说清楚。

    直接 ``image.save(target)`` 在目标文件**被别的程序打开/内存映射**时会失败，
    Windows 下报的是 ``[Errno 22] Invalid argument``（对已映射文件的写返回
    ERROR_INVALID_PARAMETER），看起来像"文件名非法"，实际是"文件被占用"。
    实测踩到：看图的程序开着 typeset PNG，重嵌字就失败（页 1/7/10）。
    先写 ``*.tmp`` 再替换：占用时替换失败，报错里直接点明去看哪个文件。
    """
    import os

    # 临时名保留 .png 结尾（PIL 按扩展名判格式，`*.png.tmp` 会 KeyError）
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, target)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ErasureError(
            f"写入 {target.name} 失败（{error}）。"
            "最常见原因：这个 PNG 正被图片查看器/预览程序打开并占用 —— "
            "关掉它再重跑该页即可。"
        ) from error


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="One-click manga translation")
    parser.add_argument(
        "--images",
        default=None,
        help="source image directory (required; book name = directory name)",
    )
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
    parser.add_argument(
        "--keep-large-boxes", action="store_true",
        help="保留大号拟音/效果字框（默认剔除：短边 > 2×同方向基准且 "
             "> 页短边 10%%）。该处理只作用于 --box-source paddle",
    )
    parser.add_argument(
        "--sfx-min-side-ratio", type=float, default=SFX_MIN_SIDE_RATIO,
        help="大拟音判定倍数：框短边 > N × 同方向基准（默认 2.0）",
    )
    parser.add_argument(
        "--sfx-min-side-page", type=float, default=SFX_MIN_SIDE_PAGE,
        help="大拟音绝对下限：框短边 > N × 页短边（默认 0.10）",
    )
    parser.add_argument(
        "--sfx-max-font-size", type=int, default=SFX_MAX_FONT_SIZE,
        help="拟声词嵌字字号上限（默认 %(default)s px）。拟声框在原图里往往"
             "又大又粗，字号跟框走会盖过对白；上限只封顶不放大（框小仍按框"
             "自适应）",
    )
    parser.add_argument(
        "--page-font-ceiling-ratio", type=float,
        default=PAGE_FONT_CEILING_RATIO,
        help="整页字号上限：译文字号 ≤ N × 该页常规字号（= 该页蓝框短边中位数，"
             "即页面主体文字尺寸）。默认 %(default)s，0 = 关闭。各岛各自撑满会"
             "让同页出现 2 倍级差（实测页 7：131px vs 基准 66px）",
    )
    parser.add_argument(
        "--short-source-max-chars", type=int, default=SHORT_SOURCE_MAX_CHARS,
        help="短原文字数门槛：原文（日文）≤ 该值时字号严格封顶到 "
             "--sfx-max-font-size（默认 %(default)s 字）。2~3 个字的台词蓝框往往"
             "很大，字号跟框走会嵌出 80~90px 的中文短句",
    )
    parser.add_argument(
        "--sfx-glossary-min-count", type=int,
        default=SFX_GLOSSARY_MIN_COUNT,
        help="拟声词自动进术语库的门槛：同一原文 + 同一译法至少出现 N 次"
             f"（默认 {SFX_GLOSSARY_MIN_COUNT}，0 = 关闭）。入库后全书统一"
             "译法；同一原文出现两种译法时自动撤出（不锁死可能错的译法）",
    )
    parser.add_argument(
        "--box-source",
        default="paddle",
        choices=["paddle", "photos"],
        help="采集期框来源：paddle = PaddleOCR 全页文本检测（默认，离线、"
             "无窗口自动化、GPU 下 32 页约 7 秒）；photos = 原来的 Photos "
             "蓝框采集（需窗口自动化）。两者产出同一 capture.json 契约，"
             "下游完全一致",
    )
    parser.add_argument(
        "--paddle-device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="PaddleOCR 设备（采集期 paddle 引擎）：auto 默认，有 CUDA 版 "
             "paddle 且检测到 GPU 就用 GPU，否则 CPU；GPU 实测快约 7.6 倍",
    )
    parser.add_argument(
        "--layout-row-spacing",
        type=int,
        default=200,
        help="vertical spacing between rows of windows (px) in "
             "--display-scale layout (default 200)",
    )
    parser.add_argument(
        "--layout-right-gap",
        type=int,
        default=100,
        help="inset of the rightmost window from the right edge (px) in "
             "--display-scale layout (default 100)",
    )
    parser.add_argument(
        "--scan-wait",
        type=float,
        default=0.5,
        help="seconds to wait after invoking the scan-text button "
             "before the first grab (default 0.5)",
    )
    parser.add_argument(
        "--scan-settle",
        type=float,
        default=1.3,
        help="deadline (s) for the scan-text edge-chrome settle "
             "re-grab (default 1.3)",
    )
    parser.add_argument(
        "--select-wait",
        type=float,
        default=0.45,
        help="base seconds to wait after Ctrl+A before grabbing "
             "(multiplied per attempt, default 0.45)",
    )
    parser.add_argument(
        "--regrab-wait",
        type=float,
        default=0.6,
        help="base seconds between overlay re-grabs while the "
             "selection is still rendering (default 0.6)",
    )
    parser.add_argument(
        "--focus-clear-pct",
        type=float,
        default=4.0,
        help="title-bar click offset right of the scan-text button "
             "centre, as percent of window width (default 4.0; raise it "
             "when the focus click still lands on the button)",
    )
    parser.add_argument("--capture-only", action="store_true",
                        help="stage 1 only (capture + blue-box check images)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--process", action="store_true",
                      help="stage 2 only (background processing)")
    mode.add_argument("--stage", choices=["ocr", "translate", "typeset"],
                      help="run only one stage of stage 2 (--pages selects "
                           "the pages for it; prerequisite artifacts must "
                           "already exist, pages without them are skipped)")
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
        "--ai-ocr-workers", type=int, default=AI_OCR_DEFAULT_WORKERS,
        help="AI OCR 并发请求数（默认 %(default)s，1 = 串行，护栏 32）。实测"
             "（deepseek-flash）：1 路 1.91s/框 → 8 路 0.20s/框（9×）→ 16 路 "
             "0.10s/框（19×）→ 24 路反而退化，故默认 12；机器/账号给力可调高，"
             "撞 429 就调小",
    )
    parser.add_argument(
        "--ai-ocr-api-key",
        default="",
        help="AI OCR API key (or AI_OCR_API_KEY env)",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "windows-ocr", "paddle"],
        default="auto",
        help="capture-stage masked OCR engine (default auto: Windows OCR then "
             "Paddle fallback; forced engines fail loudly)",
    )
    parser.add_argument(
        "--stage2-engine",
        choices=["manga-ocr", "ai-ocr"],
        default="manga-ocr",
        help="stage2 blue-box reading engine (default manga-ocr; ai-ocr needs "
             "--ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="stage2: skip pages whose typeset output already exists",
    )
    resume_group.add_argument(
        "--overwrite",
        action="store_true",
        help="replace old results: capture re-captures the selected pages "
             "(their old capture/dataset/check artifacts and cache keys are "
             "purged first), and the chosen stage redoes its work from "
             "scratch instead of reusing caches",
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
        default=3,
        help="parallel translate+typeset page workers (default 3, the cap; "
             "1 = serial). 擦字/嵌字按页并行，3 路约 3× 提速",
    )
    args = parser.parse_args(argv)
    global BLUEBOX_FILL
    try:
        parts = [int(part.strip()) for part in args.bluebox_fill.split(",")]
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            BLUEBOX_FILL = tuple(parts)
    except ValueError:
        pass

    if not args.images:
        parser.error(
            '--images is required (e.g. --images "C:\\path\\to\\原图书目录"); '
            "书名取目录名"
        )
    images = Path(args.images).resolve()
    out_root = ROOT / "outputs" / "oneclick" / images.name
    if args.confirm_no_boxes:
        confirmed = sorted(_parse_pages(args.confirm_no_boxes) or set())
        if not confirmed:
            print("no pages to confirm", file=sys.stderr)
            return 2
        _confirm_no_boxes(out_root, confirmed)
        print(f"confirmed no-blue-box pages: {', '.join(confirmed)}")
        return 0
    pages_set = _parse_pages(args.pages)
    pages = sorted(
        (path.stem for path in images.iterdir()
         if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
         and (pages_set is None or path.stem in pages_set)),
        key=lambda item: int(item) if item.isdigit() else item.lower(),
    )
    if not pages and (args.process or args.stage):
        # 只跑 stage2 时产物才是输入：用户把 --images 指到 outputs/oneclick/{书名}
        # （产物目录）也能跑——页名与原图 stem 一一对应，直接按既有 dataset 继续。
        pages = _pages_from_dataset(out_root, pages_set)
        if pages:
            print(
                f"注意：{images} 里没有图片，按既有产物继续（{len(pages)} 页）："
                f"{out_root.name}/dataset"
            )
    if not pages:
        print(f"no images found in {images}", file=sys.stderr)
        hint = _output_dir_hint(images)
        if hint:
            print(hint, file=sys.stderr)
        return 2

    if not args.process and not args.stage:
        ok = stage1_capture(images, out_root, pages_set, args.parallel,
                            args.display_scale, args.layout_row_gap,
                            args.layout_bottom_shift_up,
                            layout_region=args.layout_region,
                            box_delta=args.box_delta,
                            box_tolerance=args.box_tolerance,
                            box_area_ratio=args.box_area_ratio,
                            overwrite=args.overwrite,
                            ocr_engine=args.ocr_engine,
                            paddle_device=args.paddle_device,
                            box_source=args.box_source,
                            row_spacing_px=args.layout_row_spacing,
                            right_gap_px=args.layout_right_gap,
                            scan_wait=args.scan_wait,
                            scan_settle=args.scan_settle,
                            select_wait=args.select_wait,
                            regrab_wait=args.regrab_wait,
                            focus_clear_pct=args.focus_clear_pct)
        if not ok:
            print("capture failed", file=sys.stderr)
            return 2
        skipped = _skip_pages(out_root, pages)
        print(f"capture done: {len(pages)} pages ({len(skipped)} skipped: "
              f"black/no-text/failed) -> {out_root / 'check'}")
        review = _review_required_pages(out_root, pages)
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
    review = _review_required_pages(out_root, pages)
    if review:
        print("!! " + "=" * 60)
        print(f"!! 待人工审核页（未确认，已跳过处理）: {', '.join(review)}")
        print("!! 审核确认无框后运行:")
        print(f"!!   python one_click_translate.py --images \"{images}\" "
              f"--confirm-no-boxes {','.join(review)}")
        print("!! " + "=" * 60)
    active_pages = [p for p in pages if p not in _skip_pages(out_root, pages)]
    if not active_pages:
        print("no capturable pages (all black)", file=sys.stderr)
        return 2
    _config = _load_api_config()
    kwargs = dict(
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
        ai_ocr_workers=args.ai_ocr_workers,
        stage2_engine=args.stage2_engine,
        drop_large_boxes=not args.keep_large_boxes,
        sfx_min_side_ratio=args.sfx_min_side_ratio,
        sfx_min_side_page=args.sfx_min_side_page,
        sfx_glossary_min_count=args.sfx_glossary_min_count,
        sfx_max_font_size=args.sfx_max_font_size,
        short_source_max_chars=args.short_source_max_chars,
        page_font_ceiling_ratio=args.page_font_ceiling_ratio,
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
    )
    if args.stage:
        return 0 if _run_stage2_single(
            args.stage, images, out_root, active_pages, kwargs,
            overwrite=args.overwrite,
        ) else 2
    return 0 if stage2_process(
        images, out_root, active_pages, overwrite=args.overwrite, **kwargs,
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
