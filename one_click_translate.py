"""One-click manga translation.

Stage 1 (capture-only, default): PaddleOCR full-page text detection
(``--box-source paddle``); ``--box-source photos`` switches back to the
legacy parallel Photos frames + blue-box capture with fixed tiled windows.
Every page gets a ``{page}.bluebox_check.png``
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
import re
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
    ART_TEXT_BALLOON_MIN_CHARS,
    ART_TEXT_BALLOON_MIN_P25,
    ART_TEXT_BALLOON_RATIO,
    ART_TEXT_FLAT_RATIO,
    ART_TEXT_FLAT_RANGE,
    ART_TEXT_MAX_CHARS,
    ART_TEXT_GATE_MAX,
    ART_TEXT_GATE_MIN,
    ART_TEXT_PAD,
    ART_TEXT_PAGE_DEMOTE_MARGIN,
    ART_TEXT_PAGE_DEMOTE_MIN_SAMPLES,
    ART_TEXT_PAGE_MIN_SAMPLES,
    ART_TEXT_PAGE_PROMOTE,
    ART_TEXT_PAGE_PROMOTE_MIN_SAMPLES,
    ART_TEXT_PAGE_PROMOTE_P25,
    ART_TEXT_GUARD_RATIO,
    ART_TEXT_DIALOGUE_GUARD,
    ART_TEXT_DECISION,
    ART_TEXT_CLASSIFIER_THRESHOLD,
    ART_TEXT_REVIEW_BAND,
    ART_TEXT_SOFT_FLAG_PAGE,
    ART_TEXT_SOFT_FLAG_RATIO,
    ART_TEXT_SOFT_REVIEW_MIN_CHARS,
    ART_TEXT_STRICT_MAX_CHARS,
    ART_TEXT_VERTICAL_RATIO,
    ART_TEXT_VISION_CONTEXT,
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

# 擦字修复后端默认值。``auto``：有 LaMa 模型就用它（实测更准——FSR 会留黑残块 /
# 糊掉"白带+黑盒"结构 / 把网点抹成一片白灰，LaMa 在三个病态案例上全胜；见
# 开发记录 §13.4）。代价是**CPU 上慢 4~6 倍**（本机 onnxruntime 只有 CPU
# provider；实测页7 38.2s vs 5.8s、页60 20.9s vs 6.0s）。想要速度改 ``"fsr"``。
# 这个值进擦字缓存键，所以来回切换会自动重擦。
ERASE_INPAINTER = "auto"

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


def soft_flag_large_sfx(
    bboxes: list[list[int]], image_w: int, image_h: int, *,
    ratio: float = ART_TEXT_SOFT_FLAG_RATIO,
    page_floor: float = ART_TEXT_SOFT_FLAG_PAGE, min_group: int = 3,
) -> list[list[int]]:
    """框级软标记：大拟音硬过滤（2.0×/10%）的**放宽档，只标记不丢框**。

    严格联合版（开发记录 六之前四）的框级票：放宽到 1.2×同方向基准 且 > 页短边
    5% 的框打上"疑似效果字"标记，**框和文本都留在数据集里**——成岛后由岛级
    裁决（拿到 OCR 全文与分类器分数那一刻），被标记的岛 ≥ART_TEXT_STRICT_MAX_CHARS
    字就整岛救回、全文翻译。不丢框是救援的前提：丢了框，救回的岛就缺一段墨、
    断一句文。实测（CV）软标记贡献了严格版 453 个拦截里的 424 个。
    判据形态与 ``_filter_large_sfx`` 同族（短边/同方向基准 + 页短边绝对下限），
    基准算法共用（某方向样本 < min_group 回落全页中位）。
    """
    if len(bboxes) < 3:
        return []
    short = [min(box[2] - box[0], box[3] - box[1]) for box in bboxes]
    page_median = float(np.median(short))
    floor_px = page_floor * min(image_w, image_h)
    flagged: list[list[int]] = []
    for key, indices in _direction_groups(bboxes).items():
        baseline = _direction_baseline(bboxes, key, min_group=min_group,
                                       page_median=page_median)
        for index in indices:
            if short[index] > ratio * baseline and short[index] > floor_px:
                flagged.append(bboxes[index])
    return flagged


def flat_background_ratio(
    gray, box, *, flat_range: int = ART_TEXT_FLAT_RANGE, inset: int = 0,
) -> float | None:
    """矩形内"局部平坦"像素占比（3×3 亮度极差 < ``flat_range``）。

    气泡（白）、纯色色块、白底都平坦 → 接近 1；网点、集中线、渐变画面不平坦
    → 接近 0。用 3×3 邻域而不是逐像素：网点本身就是黑白点阵，单像素看再密的
    网点也有一半点是白的。返回 ``None`` = 矩形太小、判据无法自证，调用方应当
    保守地当作"不是手绘字"。

    ``inset`` 默认 0：调用方给的应当就是**已经收好的取样矩形**（见
    ``ART_TEXT_PAD``），这里不再二次内缩。
    """
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, y1 = max(0, x1 + inset), max(0, y1 + inset)
    x2, y2 = min(gray.shape[1], x2 - inset), min(gray.shape[0], y2 - inset)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = np.asarray(gray[y1:y2, x1:x2], dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(crop, 1, mode="edge"), (3, 3),
    )
    spread = windows.max(axis=(2, 3)) - windows.min(axis=(2, 3))
    return float((spread < flat_range).mean())


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


def _load_readings_engines(out_root: Path) -> dict[str, str]:
    """读数由哪个引擎产出（``{crop_id: "manga-ocr" | "ai-ocr"}``）。

    和 ``ocr_readings.json`` 分开存，是为了**不动那份缓存的结构**（多处读它、
    测试也锁着）。缺这个文件 = 旧格式缓存（出处未知）；那时按"出处不符"处理，
    整本重读一次把出处补齐——否则"缓存里混着两个引擎的读数"会一直查不出来
    （2026-09 实测：乙书 是 manga-ocr 读的、甲书 是视觉模型读的，半个月
    无人察觉，正是因为没有这个字段）。
    """
    path = out_root / "ocr_readings.engines.json"
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


def _store_readings_cache(out_root: Path, readings: dict,
                          engines: dict[str, str] | None = None) -> None:
    """Merge ``readings`` into the persisted OCR cache (atomic replace).

    The write merges with the on-disk state, so parallel segment
    processes do not lose each other's entries (the last writer wins only
    for entries written concurrently between load and store).

    ``engines`` records the producing engine per crop id（同一轮里也可能有两种：
    强制 ai-ocr 时读空的框会回退 manga-ocr）。只给真正写进缓存的键记出处。
    """
    stored = {key: text for key, text in readings.items() if str(text or "").strip()}
    merged = _load_readings_cache(out_root)
    merged.update(stored)
    path = out_root / "ocr_readings.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

    if engines:
        merged_engines = _load_readings_engines(out_root)
        merged_engines.update(
            {key: engines[key] for key in stored if engines.get(key)}
        )
        engines_path = out_root / "ocr_readings.engines.json"
        engines_tmp = engines_path.with_suffix(".json.tmp")
        engines_tmp.write_text(
            json.dumps(merged_engines, ensure_ascii=False), encoding="utf-8")
        engines_tmp.replace(engines_path)


def _select_crop_misses(crops: list[dict], cached: dict,
                        engines: dict[str, str] | None = None,
                        engine: str = "") -> list[dict]:
    """Crops without a cached non-empty reading (these need OCR).

    ``engine`` 给了就顺带把"**缓存是别的引擎读的**"也算成 miss：换引擎时读数
    不该被静默复用（甲书 的视觉模型读数就是这样一路留下来的）。出处**缺失**
    （旧格式缓存）同样算 miss，重读一次补上出处；代价是换引擎那一次要整本重读
    （manga-ocr 免费），换来的是"缓存里混着两个引擎"从此可查。
    """
    engines = engines or {}
    out = []
    for crop in crops:
        text = str(cached.get(crop["id"]) or "").strip()
        if not text:
            out.append(crop)
            continue
        if engine and engines.get(crop["id"]) != engine:
            out.append(crop)
    return out


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
        # 出处侧车跟着一起清，否则重采后旧出处会盖在新的读数上
        engines_path = out_root / "ocr_readings.engines.json"
        if engines_path.is_file():
            engines_payload = _load_readings_engines(out_root)
            stale_engines = [
                key for key in engines_payload
                if key.split(":", 1)[0] in selected
            ]
            if stale_engines:
                for key in stale_engines:
                    del engines_payload[key]
                _write_cache_json(engines_path, engines_payload)
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
        # 页尺寸（分类器的 area_ratio 特征要用真实值，不能用岛 bbox 估算）
        "image_size": list(islands.get("image_size") or []),
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
    texts, directions = build_page_texts(
        units, readings, dropped, vision_judge=make_vision_judge(out_root),
        keep=art_text_keep_list(out_root),
        force_drop=art_text_force_drop_list(out_root),
        dump_path=Path(out_root) / "art_text_dropped.csv")
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
    """OCR every crop with the **local manga-ocr** model.

    The AI vision path (OpenAI-compatible, e.g. GLM-4V / Qwen-VL) is
    **opt-in only** —— 它在 ``stage2_engine == "ai-ocr"`` 时才跑，**不再**
    因为 ``ai_ocr_*`` 配着就自动接管。以前不是这样：第二个分支是
    "只要 url/model/key 配全就用视觉模型读全部"，于是 ``stage2_engine =
    "manga-ocr"`` 名存实亡、每本新书白花"框数"次付费调用（实测 乙书 829 /
    甲书 586）。

    为什么默认改回 manga-ocr（2026-09，甲书 同一批裁剪上两臂对照）：随机抽 32
    个两臂不一致的框逐张看图判定，**manga-ocr 对 23 / 视觉模型对 3**（赢的三例
    都是印刷体拉丁：`[出版社]` 标志、`[印刷体日文]`、`発行：[出版社]`）。视觉
    模型输在手绘竖排——画面一整列 7 个字只读出 `素敵`、`そっ！！` 读成 `毛`、
    `す`+`で` 读成 `です`；另有三种格式性故障：返回 markdown 代码围栏、逐字加
    换行、回中文元评论（`图片中没有任何日文文本。它看起来是一个图形设计或标志…`
    / `图片中没有文字。` / `（无文字）`）。后者占全书 9/590 = 1.5%，会被
    ``is_ocr_commentary`` 过滤成空读数 → **该框台词直接丢失**。详见
    ``开发记录`` §五之后.1。

    OCR readings are persisted to ``out_root/ocr_readings.json`` (non-empty
    only): crops with a cached reading skip the OCR pass entirely, so
    re-running stage 2 (re-grouping, re-typesetting, PDF merge) costs no
    re-recognition.  Boxes that read empty are never cached and are OCRed
    again on every run (the two-pass false-selection check).  读数由哪个引擎
    产出记在 ``ocr_readings.engines.json``：旧版没有这个字段，"缓存里混着两个
    引擎的读数"因此查不出来（乙书 是 manga-ocr 读的、甲书 是视觉模型读的，
    半个月无人察觉）。换引擎时按出处**整本重读**。
    """
    cached = _load_readings_cache(out_root)
    engines_cached = _load_readings_engines(out_root)
    misses = _select_crop_misses(crops, cached, engines_cached, stage2_engine)
    stale_engine = [
        crop for crop in misses if str((cached.get(crop["id"]) or "")).strip()
    ]
    if stale_engine and progress is not None:
        recorded = sorted({
            engines_cached.get(crop["id"]) or "出处未知"
            for crop in stale_engine
        })
        progress.log(
            f"读数引擎与缓存不符（缓存是 {'/'.join(recorded)}，本次 "
            f"{stage2_engine}）→ {len(stale_engine)} 框重读一遍"
        )
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
    _store_readings_cache(
        out_root, readings,
        {crop["id"]: ("manga-ocr" if crop["id"] in fallback_ids else stage2_engine)
         for crop in misses})
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
    gap_ratio: float = 0.20,
    top_align_px: int = 6,
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

    * two VERTICAL boxes merge when their **top edges align within
      ``top_align_px``** AND the raw horizontal gap is below 20% of the
      smaller box's width;
    * two HORIZONTAL boxes merge when the raw vertical gap is below 20%
      of the smaller box's height AND their horizontal ranges overlap by
      more than 40% of the LARGER box's width;
    * mixed-orientation pairs never merge.

    **竖排用"顶端对齐"而不是"纵向重叠比例"**（用户实书反馈后改，2026-09）：
    一句话被拆成几列时，各列**顶端对齐**，但**长度可以差很多**——3 行的短列
    挨着 10 行的长列时，重叠只有较长列的 30%，过不了"重叠 > 较大框 40%"那条
    旧判据，于是同一句话被拆成两个岛。顶端对齐（而且长列短列的顶端几乎必然
    对齐）正是竖排的排版事实。横排暂未同步（横排要对齐的是**左端**）。

    (20% on raw geometry replaces the earlier "20% on padded geometry":
    padding used to add ~2*pad px to every gap -- a large constant
    relative to small boxes.  The raw form is scale-invariant.  Measured
    on 图源样张 (532 boxes): 0.6/0.5/0.4/0.3/0.2 -> 261/265/266/271/274
    groups.)

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
                # 竖排：顶端对齐 + 横向够近 → 同一句话的相邻列
                if abs(pa[1] - pb[1]) <= top_align_px and x_gap < gap_ratio * min_pw:
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


def attach_bg_flatness(
    islands: Sequence[dict], gray, *, pad: int = ART_TEXT_PAD,
) -> None:
    """给每个岛记下 ``bg_flat_ratio``（取样矩形内"局部平坦"像素占比）。

    取样矩形 = **成员框原样并集**（``members_bbox``）± ``pad``。**没有成员并集
    时退回 ``image_bbox``**（老数据集 / 兜底数据集）：那个带着编组扩边 6px，会
    把邻居的墨迹吃进取样范围（见 ``ART_TEXT_PAD`` 的说明）。

    **判据要的是整岛，不是单个框**：同一句对白的几个框各读一截
    （"あの♡" / "でもももも…"），逐框判会把一句完好印在气泡里的对白判成手绘
    字。岛矩形 = 这句话在版面上的占位，气泡上 → 平坦，画面上 → 不平坦。

    存**比例**而不是布尔：以后调阈值不必重建 dataset（重建要重跑整条链）。
    """
    for island in islands:
        bbox = list(island.get("members_bbox") or island.get("image_bbox") or [])
        if len(bbox) != 4:
            continue
        # 框本身太小就不判（判据无法自证）——按**原始框**判，不按扩边后的
        # 取样矩形判，否则扩边会把"太小"掩盖成"刚好够判"。
        if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 4:
            continue
        rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
        ratio = flat_background_ratio(gray, rect)
        if ratio is not None:
            island["bg_flat_ratio"] = round(ratio, 4)


def art_text_island(source_text: str, bg_flat_ratio, *, vertical: bool = False,
                    flat_ratio: float = ART_TEXT_FLAT_RATIO,
                    max_chars: int = ART_TEXT_MAX_CHARS,
                    vertical_ratio: float = ART_TEXT_VERTICAL_RATIO) -> bool:
    """这个岛是不是"画在画面上的手绘效果字/拟声词"（该保留原图、不译不嵌）。

    **岛底不像气泡** + **够短（或竖排且更不像气泡）**：

    * ``bg_flat_ratio`` < ``flat_ratio``(0.70) —— 取样矩形里"局部平坦"像素占比
      达不到气泡的水平。气泡对白实测 ≥ 0.74，而"底色没那么深"的手绘拟声落在
      0.40~0.66（实测 どきっ 0.40 / キェツ♡ 0.52 / ゴゴゴ 0.63），两者有空档，
      所以上限能放到 0.70（见 thresholds 的标定表）。
    * 原文 ≤ ``max_chars``(10) 字 —— 挡"压在画面上的印刷正文"（封面警示语、
      后记段落）。
    * **竖排且占比 < ``vertical_ratio``(0.40) 时无视字数护栏**：长拟声确实存在
      （`びくっぷるぷるふるふるえる` / `ぬるっと♡じゅっ必ずい`），而长印刷正文
      （封面警示语、后记段落）是**横排长行**——方向把它们分开。

    只看几何、字数与排版方向，**不看语义**：这些框读出来往往是一句像模像样的话
    （实测「よろしく」「だいすき」「だめ」「ごめんなさい」全是手绘字），语义判据
    分不开（字符级规则、字形特征都测过，见 开发记录）。
    """
    if not source_text or bg_flat_ratio is None:
        return False
    ratio = float(bg_flat_ratio)
    if ratio >= flat_ratio:
        return False
    if short_source(source_text, max_chars):
        return True
    return bool(vertical and ratio < vertical_ratio)


def build_bluebox_dataset(
    manifest_path, page_dir, source_path, min_island_area: float = 800.0, pad: int = 0,
    drop_large_boxes: bool = True,
    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE,
    drop_art_boxes: bool = False,
) -> list[dict]:
    """islands.json purely from the Photos blue boxes (original image space).

    No highlight-island detection: the atomic blue boxes are re-grouped by
    the two-axis gap/overlap rule (``_group_boxes_by_overlap``) -- every
    group is one island, padded ``pad`` px, and its own independent bbox.

    **两条框来源共用同一套过滤**（`manga_localizer.thresholds`）：整页大框
    （面积 > 20% 页）在采集期已丢；这里再丢**大号拟音/效果字**（短边 >
    2.0×同方向基准 且 > 页短边 10%），并给每个岛记下**岛底平坦度**
    （``bg_flat_ratio``，手绘字判据的几何侧，见 ``attach_bg_flatness``）。
    放在共用阶段而非采集期，Photos 与 Paddle 两条路径才会得到同一结果。
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
    box_dropped: list[dict] = []
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
                page_short = float(min(image_w, image_h))
                for item in atomic_boxes:
                    if item["id"] not in removed:
                        continue
                    box = list(item["image_bbox"])
                    short = min(box[2] - box[0], box[3] - box[1])
                    box_dropped.append({
                        "页": Path(page_dir).name, "框": item["id"], "原因": "大拟音/效果字框",
                        "box": box, "短边": int(short),
                        "页短边比": round(short / max(1.0, page_short), 3),
                        "说明": f"短边 > {sfx_min_side_ratio}×同方向基准 且 > "
                                f"{sfx_min_side_page:.0%} 页短边",
                    })
                blocks = [
                    b for b in blocks
                    if b.get("page_scale") or b["id"] not in removed
                ]
                print(
                    f"page {Path(page_dir).name}: 过滤 {len(dropped_large)} 个大拟音/"
                    f"效果字框（短边 > {sfx_min_side_ratio}×同方向基准 且 > "
                    f"{sfx_min_side_page:.0%} 页短边）"
                )

    # 框级软标记（严格联合版第一段）：放宽档只标记不丢框——框和文本都留下，
    # 成岛后由岛级裁决（见 thresholds.ART_TEXT_SOFT_FLAG_* 的标定说明）。
    # **与 bg_flat_ratio 同一门控**（drop_art_boxes）：功能没开时绝不写标记，
    # 否则功能关着的书也会被岛级裁决丢岛。
    soft_ids: set[str] = set()
    if drop_art_boxes and blocks:
        surviving = [b for b in blocks if not b.get("page_scale")]
        if len(surviving) >= 3:
            soft_flagged = soft_flag_large_sfx(
                [list(b["image_bbox"]) for b in surviving], image_w, image_h,
            )
            soft_set = {tuple(box) for box in soft_flagged}
            soft_ids = {
                b["id"] for b in surviving
                if tuple(b["image_bbox"]) in soft_set
            }
            if soft_ids:
                print(
                    f"page {Path(page_dir).name}: {len(soft_ids)} 个框软标记"
                    f"（短边 > {ART_TEXT_SOFT_FLAG_RATIO}×同方向基准 且 > "
                    f"{ART_TEXT_SOFT_FLAG_PAGE:.0%} 页短边，疑似效果字 → 岛级裁决）"
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
            box_dropped.append({
                "页": Path(page_dir).name, "框": "/".join(members), "原因": "微框（面积不达标）",
                "box": [x1, y1, x2, y2], "短边": int(min(x2 - x1, y2 - y1)),
                "页短边比": round(min(x2 - x1, y2 - y1) / max(1.0, min(image_w, image_h)), 3),
                "说明": f"面积 {int(area)} < min_island_area {int(min_island_area)}",
            })
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
        # 成员框**原样**并集：手绘字判据用它取样（见 ART_TEXT_PAD）。`image_bbox`
        # 带着编组扩边 6px，是给填黑/擦字/嵌字的版面余量，拿它判文字会把邻居的
        # 墨迹吃进来。
        member_boxes = [boxes_by_id[m]["image_bbox"] for m in members
                        if m in boxes_by_id]
        if member_boxes:
            island_record["members_bbox"] = [
                min(b[0] for b in member_boxes), min(b[1] for b in member_boxes),
                max(b[2] for b in member_boxes), max(b[3] for b in member_boxes),
            ]
        if any(m in soft_ids for m in members):
            # 框级软标记（疑似效果字）：岛级裁决用它投"像拟声"一票（与分类器
            # P<阈值 同权），字数护栏 ≥ART_TEXT_STRICT_MAX_CHARS 的岛照常翻译。
            island_record["sfx_soft_flag"] = True
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
    if drop_art_boxes:
        # 手绘字判据的几何侧：岛底平坦度。读数侧（字数护栏 + 判定）在
        # build_page_texts —— 这里只采集，不丢岛。
        try:
            with Image.open(source_path) as handle:
                gray = np.asarray(handle.convert("L"))
        except (OSError, ValueError):
            gray = None
        if gray is not None:
            attach_bg_flatness(islands, gray)
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
                "soft_flag_version": 1,
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
    # 框级"被丢清单"：**框级过滤从来没有人工真值覆盖**（被丢的框在编组前就消失、
    # 从不进核对表），所以"筛框到底杀了多少台词"此前无人知道。这里把它落表，
    # 便于将来核对（与岛级的 art_text_dropped.csv 同一思路）。
    if box_dropped:
        import csv as _csv

        drop_path = page_dir / "dropped_boxes.csv"
        fields = ["页", "框", "原因", "短边", "页短边比", "说明", "box"]
        with drop_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in box_dropped:
                row = dict(row)
                row["box"] = ",".join(str(int(v)) for v in row["box"])
                writer.writerow(row)
    shutil.copyfile(source_path, page_dir / "original.jpg")
    return islands


# 手绘字分组用的装饰符号：读数里 ♡♥…～ 这类只是装饰，不参与"同一处字"的比较
_ART_TEXT_DECOR = re.compile(r"[\s♡♥★☆♪…、。．.！？!?～〜→←↑↓・「」『』（）()\[\]{}]+")


def _art_text_key(text: str) -> str:
    """比较"同一处字"用的归一化读数（去掉装饰符号与空白）。"""
    return _ART_TEXT_DECOR.sub("", text or "")


def _same_lettering(a: str, b: str) -> bool:
    """两段读数是不是"同一处字"：归一化后一短一长、短的落在长的开头。

    同一团手绘字常被检出成两个**重叠**框（横竖不同向时编组规则不合并），两个
    框读出来的是同一条字的不同截法（实测「ぬるっと♡じゅっ」/「ぬるっと♡じゅっ
    必ずい」、「かわいい♥」/「かわい♡」）。不把它们当一条字，同一个字就会拿到
    两个相反的判定。
    """
    left, right = _art_text_key(a), _art_text_key(b)
    if len(left) < 2 or len(right) < 2:
        return False
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return long.startswith(short)


def _island_rect(island: dict) -> tuple[int, int, int, int] | None:
    """判据的取样矩形（成员框原样并集 ± ART_TEXT_PAD），拿不到就是 None。"""
    bbox = list(island.get("members_bbox") or island.get("image_bbox") or [])
    if len(bbox) != 4:
        return None
    pad = ART_TEXT_PAD
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def _rects_overlap(a, b) -> bool:
    return (min(a[2], b[2]) - max(a[0], b[0]) > 0
            and min(a[3], b[3]) - max(a[1], b[1]) > 0)


def _balloon_ratios(unit, texts) -> list[float]:
    """一页（或一本书）的"气泡样本"：长文本岛（≥12 字，几乎必然是对白）的平坦占比。"""
    values = []
    for iid, island in unit["islands"].items():
        ratio = island.get("bg_flat_ratio")
        if ratio is None:
            continue
        if len((texts or {}).get(iid, "")) >= ART_TEXT_BALLOON_MIN_CHARS:
            values.append(float(ratio))
    return values


def _balloon_stats(values) -> tuple[float, float, float]:
    """``(中位, p25, 闸门)``——书级与页级共用同一套统计，免得两边口径漂移。"""
    import statistics as _st

    ordered = sorted(values)
    median = float(_st.median(ordered))
    p25 = float(ordered[max(0, int(len(ordered) * 0.25))])
    gate = min(ART_TEXT_GATE_MAX,
               max(ART_TEXT_GATE_MIN, median * ART_TEXT_BALLOON_RATIO))
    return median, p25, gate


def art_text_book_baseline(units, texts_by_page) -> dict:
    """整本书的气泡基准：``{gate, median, p25, n, basis}``（``gate=None`` = 自证不住）。

    基准 = 这本书自己**长文本岛**（≥12 字，几乎必然是对白）的平坦占比，量的是
    **p25**——长文本岛里必然混着少数不该算气泡基准的岛（横排印刷长文、偶然被
    读出 ≥12 字的手绘字），p10 那个位置由它们决定，同一本书换个编组就会从 0.67
    掉到 0.37（见 thresholds 的实测）。
    """
    values = []
    for page, unit in units.items():
        values.extend(_balloon_ratios(unit, (texts_by_page or {}).get(page)))
    if len(values) < 8:
        return {"gate": None, "median": None, "p25": None, "n": len(values),
                "basis": f"长文本岛只有 {len(values)} 个，样本不足"}
    median, p25, gate = _balloon_stats(values)
    if p25 < ART_TEXT_BALLOON_MIN_P25:
        return {"gate": None, "median": median, "p25": p25, "n": len(values),
                "basis": (f"长文本岛占比中位 {median:.2f} 但 p25 只有 {p25:.2f}"
                          f"（< {ART_TEXT_BALLOON_MIN_P25}）→ 自证不住气泡基准")}
    return {"gate": round(gate, 4), "median": median, "p25": p25, "n": len(values),
            "basis": f"长文本岛占比中位 {median:.2f} / p25 {p25:.2f}"}


def art_text_gate(units, texts_by_page) -> tuple[float | None, str]:
    """按书自标定闸门。返回 ``(闸门 或 None, 说明)``。

    ``None`` = 这本书自证不住气泡基准 → **书级**禁用几何判据（宁可全部保留原图，
    也不要在无信号的页面上丢一半对白）。逐页路由见 ``art_text_page_gates``。
    """
    baseline = art_text_book_baseline(units, texts_by_page)
    return baseline["gate"], baseline["basis"]


def art_text_page_gates(units, texts_by_page, baseline,
                        *, promote: bool | None = None,
                        ) -> dict[str, float | None]:
    """逐页定闸门：``{页: 闸门 或 None}``，``None`` = 该页交给视觉判据（或没有
    视觉接口时不丢任何岛）。

    ``promote=None`` 时读 ``ART_TEXT_PAGE_PROMOTE``（默认关，理由见 thresholds）。

    书级结论（``baseline``）只当**先验**，页级规则刻意不对称：样本不足就继承
    书级结论（不判断、也就不花钱），只有明确的反向证据才改判。两个方向：

    * 书级自证得住 → 该页本地中位比书级低 ``ART_TEXT_PAGE_DEMOTE_MARGIN``
      （或样本够多且本地 p25 比书级 p25 低同样多）→ 这页的"气泡基准"不成立，
      转视觉；其余页仍用**书级闸门**（页级闸门在实测里只差 0.03 上下，用书级
      更稳）。两条判据都拿**书级同统计量**做比较、都带同一个边距：直接与绝对
      门槛比会被边界噪声推走（实测甲书 页 2 的 p25 = 0.649，与 0.65 只差
      0.001，却会把一页正常的页整页转走）。
    * 书级自证不住 → 只有 ``promote`` 且页级证据**更强**（更高 p25、更多样本）
      才升回几何，且用该页自己的闸门。
    """
    if promote is None:
        promote = ART_TEXT_PAGE_PROMOTE
    book_gate = baseline.get("gate")
    book_median = baseline.get("median")
    book_p25 = baseline.get("p25")
    gates: dict[str, float | None] = {}
    for page, unit in units.items():
        values = _balloon_ratios(unit, (texts_by_page or {}).get(page))
        if len(values) < ART_TEXT_PAGE_MIN_SAMPLES:
            gates[page] = book_gate               # 样本不足 → 继承书级
            continue
        median, p25, page_gate = _balloon_stats(values)
        if book_gate is not None:
            demote = median < float(book_median) - ART_TEXT_PAGE_DEMOTE_MARGIN
            if not demote and len(values) >= ART_TEXT_PAGE_DEMOTE_MIN_SAMPLES:
                demote = p25 < float(book_p25) - ART_TEXT_PAGE_DEMOTE_MARGIN
            gates[page] = None if demote else book_gate
            continue
        promote_ok = (promote
                      and len(values) >= ART_TEXT_PAGE_PROMOTE_MIN_SAMPLES
                      and p25 >= ART_TEXT_PAGE_PROMOTE_P25)
        gates[page] = page_gate if promote_ok else None
    return gates


def make_vision_judge(out_root: Path, *, progress=None):
    """自证不住的书（或页）用的视觉判据回调；没配视觉接口时返回 None。

    返回的 ``judge(units, page_texts) -> set[str]`` 给出**该丢弃的岛**（``页:岛``）。
    逐页路由只把"几何不可信"的页放进 ``units`` 送过来，所以判据只问这些页；模型答
    ``S``（拟声/效果字）或 ``U``（图里没有清晰可读的日文）都丢，**答 D 才照常翻译**。

    **判不出来（重试后仍失败、或回复里没有 S/D/U）按"保留原图、不翻译"处理**
    （fail-closed）：这类岛的读数本来就不可信，照译等于把拟声/读废的字盖成中文；
    代价是接口大面积故障时会整片不译，**所以失败要显式可见**——日志报数量，某页
    失败过半且 ≥3 个时把该页标进审核页。

    跳过两类岛（实测"跳过长文本是免费的"：调用数 431→264，F1 只差 0.004）：

    * **无读数的岛**――管线本来就不翻译它（文本为空 → 保留原图），不必问；
    * **长文本岛**（≥ ``ART_TEXT_BALLOON_MIN_CHARS`` 字）――几乎必然是对白。

    判定结果落盘 ``art_text_judge.json`` 缓存（键含岛框，编组一变自动失效），
    重跑（含 ``--stage typeset``）不会重复花钱。
    """
    config = _load_api_config()
    url = str(config.get("ai_ocr_url") or "")
    model = str(config.get("ai_ocr_model") or "")
    key = str(config.get("ai_ocr_api_key") or config.get("deepseek_api_key") or "")
    if not (url and model and key):
        return None

    from manga_localizer.sfx_judge import DEFAULT_WORKERS, judge_crops

    def judge(units, page_texts) -> set[str]:
        cache_path = out_root / "art_text_judge.json"
        cached: dict[str, str] = {}
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    cached = {str(k): str(v) for k, v in payload.items()}
            except (OSError, UnicodeError, json.JSONDecodeError):
                cached = {}
        pending, keyed = [], {}
        for page, unit in units.items():
            texts = page_texts.get(page) or {}
            source = out_root / "dataset" / page / "original.jpg"
            for iid, island in unit["islands"].items():
                text = texts.get(iid, "")
                if not text.strip() or len(text) >= ART_TEXT_BALLOON_MIN_CHARS:
                    continue
                box = list(island.get("members_bbox") or island["image_bbox"])
                cache_key = f"{page}:{iid}:" + ",".join(str(int(v)) for v in box)
                keyed[cache_key] = f"{page}:{iid}"
                if cache_key not in cached:
                    pending.append({"id": cache_key, "source": str(source),
                                    "bbox": box, "pad": ART_TEXT_VISION_CONTEXT})
        if pending:
            if progress is not None:
                progress.log(f"视觉判据：送 {len(pending)} 个岛（几何不可信，改用看图判定）")
            cached.update(judge_crops(
                pending, url, model, key, progress=progress,
                max_workers=DEFAULT_WORKERS))
            _write_cache_json(cache_path, cached)
        drop = set()
        failed_by_page: dict[str, int] = {}
        asked_by_page: dict[str, int] = {}
        for cache_key, island_key in keyed.items():
            answer = str(cached.get(cache_key) or "").upper()[:1]
            page = island_key.partition(":")[0]
            asked_by_page[page] = asked_by_page.get(page, 0) + 1
            if answer == "D":
                continue                    # 明确判成台词 → 照常翻译
            if answer not in ("S", "U"):
                # 判不出来：当"保留原图、不翻译"（fail-closed）。照译才是错的——
                # 读数本来就不可信，盖成中文就是往画面上写错字。
                failed_by_page[page] = failed_by_page.get(page, 0) + 1
            drop.add(island_key)
        failed = sum(failed_by_page.values())
        if failed:
            message = (f"视觉判据：{failed}/{len(keyed)} 个岛没得到有效回答"
                       f"（接口失败或回复不合格式）→ 按保留原图处理，不翻译")
            if progress is not None:
                progress.log(message)
            else:
                print(message)
            for page, count in sorted(failed_by_page.items(),
                                      key=lambda item: -item[1]):
                if count >= 3 and count * 2 >= asked_by_page.get(page, count):
                    _mark_review_required(
                        out_root, page,
                        f"视觉判据 {count}/{asked_by_page.get(page, count)} 个岛判"
                        f"不出（已按保留原图处理，请复核该页）",
                    )
        return drop

    return judge


def _art_text_verdicts(by_island: dict, texts: dict[str, str],
                       directions: dict[str, str],
                       flat_gate: float | None = None) -> set[str]:
    """该按"手绘字"处理的岛（判据 + 重叠同判）。

    先按"同一处字 + 矩形相交"把岛并成组（``_same_lettering``），组内**同判**：
    占比取组内最小（任一段明显落在画面上，整团就算画在画面上）、原文取最长
    的一段（取最短会把印刷长句的碎片误杀）。单岛自成一组，行为与逐岛判一致。
    """
    ids = list(by_island)
    parent = {iid: iid for iid in ids}

    def find(iid):
        while parent[iid] != iid:
            parent[iid] = parent[parent[iid]]
            iid = parent[iid]
        return iid

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for index, left in enumerate(ids):
        rect_left = _island_rect(by_island[left])
        if rect_left is None:
            continue
        for right in ids[index + 1:]:
            if not _same_lettering(texts.get(left, ""), texts.get(right, "")):
                continue
            rect_right = _island_rect(by_island[right])
            if rect_right is not None and _rects_overlap(rect_left, rect_right):
                union(left, right)

    groups: dict[str, list[str]] = {}
    for iid in ids:
        groups.setdefault(find(iid), []).append(iid)

    verdicts: set[str] = set()
    for members in groups.values():
        ratios = [
            by_island[iid].get("bg_flat_ratio") for iid in members
            if by_island[iid].get("bg_flat_ratio") is not None
        ]
        if not ratios:
            continue
        longest = max((texts.get(iid, "") for iid in members), key=len, default="")
        # 组内方向取任一成员（同一条字被切开的碎片方向一致；不一致时保守当横排）
        vertical = all(directions.get(iid) == "vertical" for iid in members)
        if art_text_island(longest, min(ratios), vertical=vertical,
                           flat_ratio=flat_gate):
            verdicts.update(members)
    return verdicts


def art_text_keep_list(out_root: Path) -> set[str]:
    """读 ``out_root/art_text_keep.txt``：**强制保留**清单（``页:岛``，``#`` 注释）。

    为什么需要它：实测**没有任何信号**能把"手绘台词"与"手绘拟声"分开——几何占比
    （最低那一档也只有约 70% 是"不翻译"）、看图模型（在这批上前者判 S 的仍有 28 个
    是台词）、日文词性/词典（只是长度的代理，误杀降 9 个的同时漏拟声涨到 147/173）、
    中文侧判定（F1 0.194）。所以"保证台词不丢"不做成判据，做成**机制**：

    1. 每次跑都把被丢的岛写成 ``art_text_dropped.csv`` + 同名 ``.html``（带裁剪图，
       见 ``_write_art_text_dropped``）；
    2. 人工核对后把误杀的写进 ``art_text_keep.txt``（一行一个 ``页:岛``）；
    3. 再跑——这些岛**不再丢**，且不用重跑 OCR/翻译（读数缓存还在，只重排受影响页）。

    两个调用点（``_stage2_ocr`` 与 ``_load_page_texts``）都读这个文件，保证
    "重跑 typeset" 与首次跑的口径一致（否则救回的岛会被重新丢掉）。
    """
    path = Path(out_root) / "art_text_keep.txt"
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return set()
    return {line.split("#", 1)[0].strip() for line in lines
            if line.split("#", 1)[0].strip()}


def art_text_force_drop_list(out_root: Path) -> set[str]:
    """`out_root/art_text_force_drop.txt`：**强制丢弃**清单（``页:岛``）——人工复核后
    认为"该岛确实是拟声/不该翻译"时写进去（与 `art_text_keep.txt` 对称）。"""
    path = Path(out_root) / "art_text_force_drop.txt"
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return set()
    return {line.split("#", 1)[0].strip() for line in lines
            if line.split("#", 1)[0].strip()}


def _write_art_text_review(path: Path, rows: list[dict]) -> None:
    """复核清单：概率落在阈值附近、判据没把握的岛（用户要求"可疑的交给我判"）。"""
    import csv

    fields = ["岛", "概率", "决定", "原文"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record_art_text_drop(drops: list[dict], keep: set[str], island_key: str,
                          source: str, answer: str, island: dict,
                          direction: str, text: str) -> None:
    """记一条"被丢"（给清单用）。``keep`` 里的岛仍记账，便于统计救回数。"""
    drops.append({
        "岛": island_key,
        "来源": source,
        "模型答案": answer,
        "占比": island.get("bg_flat_ratio"),
        "方向": "竖" if direction == "vertical" else "横",
        "字数": len([c for c in (text or "").strip() if not c.isspace()]),
        "原文": " ".join((text or "").split()),
        "保留": "是" if island_key in keep else "",
    })


def _write_art_text_dropped(path: Path, drops: list[dict], units) -> None:
    """被丢清单：CSV（机器读）+ HTML（带裁剪图，人工核）。

    这张表就是"台词不丢"的兜底：每次丢弃都在表里，人工扫一眼把误杀的岛写进
    ``art_text_keep.txt`` 即可救回。裁剪图直接内嵌（data URL），不依赖原图路径。
    """
    import base64
    import csv
    import html as _html
    import io as _io

    fields = ["岛", "来源", "模型答案", "占比", "方向", "字数", "原文", "保留"]
    csv_path = Path(path)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(drops)

    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>被丢的手绘字岛</title>",
             "<style>body{font-family:sans-serif;font-size:13px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;"
             "vertical-align:middle}img{max-height:110px}</style>",
             f"<h3>被丢的手绘字岛（{len(drops)} 个）</h3>",
             "<p>核对：误杀的岛写进同目录 <code>art_text_keep.txt</code>"
             "（一行一个 <code>页:岛</code>）后重跑，即不再丢。</p><table>",
             "<tr><th>岛<th>来源<th>模型<th>占比<th>方向<th>字数<th>原文"
             "<th>裁剪<th>已在保留清单</tr>"]
    for row in drops:
        page, _, island_id = row["岛"].partition(":")
        image_tag = ""
        unit = units.get(page)
        island = (unit or {}).get("islands", {}).get(island_id)
        source = Path("dataset") / page / "original.jpg"
        if island is not None:
            try:
                with Image.open(Path(path).parent / source) as handle:
                    x1, y1, x2, y2 = (island.get("members_bbox")
                                      or island["image_bbox"])
                    pad = 8
                    crop = handle.convert("RGB").crop(
                        (max(0, x1 - pad), max(0, y1 - pad),
                         min(handle.width, x2 + pad), min(handle.height, y2 + pad)))
                    scale = min(2.5, 220 / max(1, crop.width), 220 / max(1, crop.height))
                    crop = crop.resize((max(1, int(crop.width * scale)),
                                        max(1, int(crop.height * scale))),
                                       Image.LANCZOS)
                    buffer = _io.BytesIO()
                    crop.save(buffer, format="JPEG", quality=80)
                image_tag = ("<img src='data:image/jpeg;base64,"
                             + base64.b64encode(buffer.getvalue()).decode("ascii")
                             + "'>")
            except (OSError, ValueError):
                image_tag = "(裁剪失败)"
        cells = [_html.escape(str(row.get(key) or "")) for key in
                 ("岛", "来源", "模型答案", "占比", "方向", "字数", "原文")]
        cells += [image_tag, _html.escape(str(row.get("保留") or ""))]
        parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    parts.append("</table>")
    Path(path).with_suffix(".html").write_text("".join(parts), encoding="utf-8")


def _typeset_page_sort_key(path) -> tuple:
    """嵌字图的页序键：**按页号数字排**（1, 2, 10, 11…），不是字典序。

    踩过的坑：`Path("10.typeset.png").stem` 得到 `"10.typeset"`（只剥最后一个
    后缀），永远不是纯数字 → 排序键落进"按字符串"分支 → PDF 页序变成
    1, 10, 100, 101…（实测 160 页只有第 1 页对）。所以这里显式剥掉
    `.typeset.png` 再判数字。
    """
    name = str(getattr(path, "name", path)).replace(".typeset.png", "")
    if name.isdigit():
        return (0, int(name), "")
    return (1, 0, name.lower())


def _typeset_fingerprint(page: str, translations_page, directions, dropped,
                         config_slice) -> str:
    """这一页"要不要重嵌"的输入指纹（任一输入变 → 指纹变）。"""
    import hashlib

    payload = json.dumps({
        "page": page,
        "texts": {k: str(v) for k, v in sorted((translations_page or {}).items())},
        "directions": {k: str(v) for k, v in sorted((directions or {}).items())},
        "dropped": sorted(str(v) for v in (dropped or set())),
        "config": {k: str(v) for k, v in sorted(config_slice.items())},
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_page_texts(units, readings, dropped_boxes,
                     flat_gate: float | None = None, vision_judge=None,
                     keep: set[str] | None = None, dump_path: Path | None = None,
                     force_drop: set[str] | None = None):
    """Join blue-box manga readings into per-island sentence texts.

    Dropped boxes (false selections) never join a sentence and never set a
    direction.  Islands whose members all read empty get an empty text so
    the caller keeps them untouched (no erasure, no typesetting).
    Returns ``(page_texts, page_directions)``.

    标了 ``bg_flat_ratio`` 且**原文很短**的岛整岛不参与：岛底不平坦说明这些
    "字"是画在画面上的手绘效果字/拟声词，读出来往往是"像模像样的一句话"
    （实测「よろしく」「だいすき」），照译照嵌只是往画面上盖一行中文。原文长的
    （压在画面上的封面警示语、后记段落）照旧参与——这才是
    ``ART_TEXT_MAX_CHARS`` 这条字数护栏的作用。判空文本 = 保留原图（不擦、
    不译、不嵌），与"读不出字的假选框"同一出口。判据见 ``art_text_island``。

    两条护栏在这里生效，都是被实测逼出来的：

    * **AI 说明性输出按空读数处理**（``is_ocr_commentary``）。视觉模型读不到字时
      会回一句"图片中没有任何日文文本…"，它不算读数，但**字数很多**——当成原文
      就会一路翻成中文塞进原框，而且会把字数护栏骗过去（实测页 4 island_014：
      占比 0.23 本该判成手绘字，就因为这 37 字的说明文字被放行了）。
    * **同一处字被切成几个岛时同判**（``_same_lettering`` + 矩形相交）。两个框
      盖着同一团手绘字时，矩形只差一点、占比却可能分居阈值两侧（实测页 17
      「かわいい♥」0.46 留 /「かわい♡」0.37 丢；另一处「ぬるっと♡じゅっ」8 字
      丢 /「ぬるっと♡じゅっ必ずい」11 字靠字数护栏漏过）。同判取**最小占比**
      （任一段明显落在画面上就算整团都在画面上）+ **最长的一段原文**（用最短的
      会把印刷长句的碎片误杀）。
    """
    page_texts: dict[str, dict[str, str]] = {}
    page_directions: dict[str, dict[str, str]] = {}
    art_text_islands: set[str] = set()
    keep = keep or set()
    force_drop = force_drop or set()
    review_rows: list[dict] = []
    drops: list[dict] = []
    dropped_by_keep = 0
    commentary_boxes = 0
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
                        reading = readings.get(f"{page}:box:{mid}", "")
                        if is_ocr_commentary(reading):
                            commentary_boxes += 1
                            reading = ""
                        members.append((box, reading))
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
    # 手绘字（画在画面上的效果字/拟声词）：**整岛判**——同一句对白的几个框各读
    # 一截，逐框判会把印在气泡里的对白判成手绘字。判空岛文本 = 该岛保留原图
    # （不擦、不译、不嵌），与"读不出字的假选框"同一出口。
    # 路由**按书自证 + 逐页改判**：书级结论当先验，页级只做"反向证据"的改判
    # （见 art_text_page_gates / thresholds）——混合的书不再被整本二选一。
    if flat_gate is not None:
        gate, basis = float(flat_gate), "调用方指定（跳过按书自标定）"
        page_gates = {page: gate for page in units}
    else:
        baseline = art_text_book_baseline(units, page_texts)
        gate, basis = baseline["gate"], baseline["basis"]
        page_gates = art_text_page_gates(units, page_texts, baseline)
    measured = any(
        island.get("bg_flat_ratio") is not None
        for unit in units.values() for island in unit["islands"].values()
    )
    geometry_pages = {page: value for page, value in page_gates.items()
                      if value is not None}
    vision_pages = [page for page, value in page_gates.items() if value is None]

    if gate is None:
        detail = (f"；其中 {len(geometry_pages)} 页本地自证得住 → 该页仍用几何"
                  if geometry_pages else "")
        print(f"手绘字判据：本书自证不住几何基准（{basis}）{detail}")
    else:
        detail = (f"；{len(vision_pages)} 页本地自证不住 → 该页转视觉判据"
                  if vision_pages else "")
        print(f"手绘字判据：本书闸门 {gate:.2f}（{basis}）{detail}")

    # 保台词档位（ART_TEXT_DIALOGUE_GUARD）：几何候选先交模型表态，只丢"两路都同意"的。
    # 没配视觉接口时自动退回 off（现状），并在日志里说明——绝不因为配不上接口就改变行为。
    guard_mode = str(ART_TEXT_DIALOGUE_GUARD)
    guard_drops: set[str] = set()
    guard_active = guard_mode in ("agree", "strict") and vision_judge and measured
    if guard_mode in ("agree", "strict") and not guard_active:
        print(f"手绘字判据：保台词档位 {guard_mode} 需要视觉判据（没配接口）→ 退回现状")
    if guard_active:
        guard_drops = set(vision_judge(units, page_texts) or set())
        print(f"手绘字判据：保台词档位 {guard_mode}——几何候选只丢"
              f"「模型也判拟声/无字」的 {len(guard_drops)} 个"
              f"（模型判台词的一律照常翻译）")

    def _guard_allows(page: str, iid: str) -> bool:
        """这个候选是否允许丢（True = 丢）。off 档恒 True；其余要求模型同意。"""
        if not guard_active:
            return True
        if f"{page}:{iid}" not in guard_drops:
            return False
        if guard_mode == "strict":
            ratio = units[page]["islands"][iid].get("bg_flat_ratio")
            return ratio is not None and float(ratio) < ART_TEXT_GUARD_RATIO
        return True

    # 决策侧：默认用**小分类器**（用我们自己的真值训的，阈值即"保台词↔少混入"旋钮，
    # 见 thresholds.ART_TEXT_DECISION）；权重缺失/维度不符时自动退回几何规则。
    classifier = None
    # 调用方**显式给了 flat_gate** = "按我指定的几何闸门走"（单元测试与特殊调用都这样用），
    # 只有自标定那条路才交给分类器。
    if str(ART_TEXT_DECISION) == "classifier" and flat_gate is None:
        from manga_localizer.sfx_classifier import (
            extract_features, load_weights, probability,
        )

        classifier = load_weights()
        if classifier is None:
            print("手绘字判据：分类器权重缺失 → 退回几何规则")
    if classifier is not None:
        page_sizes = {}
        for page, unit in units.items():
            islands = unit["islands"]
            first = next(iter(islands.values()), None)
            if first and "_page_size" not in first:
                pass
            page_sizes[page] = None
        dropped_by_classifier = 0
        dropped_by_flag = 0
        dropped_total = 0
        # **分类器是书无关的**（它量的是岛自身的特征，不依赖"这本书自证得住"），
        # 所以对所有页生效——这正是它相对几何路的价值：自证不住的书不必再逐岛调接口。
        # 视觉路由只在分类器不可用时兜底（见下面的 vision_pages 分支）。
        for page in units:
            unit = units[page]
            size = unit.get("image_size") or []
            if len(size) == 2:
                page_size = (float(size[0]), float(size[1]))
            else:                      # 老数据集没有 image_size → 用岛 bbox 兜底
                xs = [i["image_bbox"][2] for i in unit["islands"].values()] or [1]
                ys = [i["image_bbox"][3] for i in unit["islands"].values()] or [1]
                page_size = (max(xs) * 1.05, max(ys) * 1.05)
            threshold = float(ART_TEXT_CLASSIFIER_THRESHOLD)
            for iid, island in unit["islands"].items():
                text = page_texts[page][iid].strip()
                if not text:
                    continue
                island_key = f"{page}:{iid}"
                flagged = bool(island.get("sfx_soft_flag"))
                chars_n = len([c for c in text if not c.isspace()])
                score = None
                if island.get("bg_flat_ratio") is not None:
                    bbox = list(island.get("members_bbox") or island["image_bbox"])
                    features = extract_features(
                        text=text, ratio=island.get("bg_flat_ratio"),
                        direction=page_directions[page][iid], bbox=bbox,
                        page_size=page_size)
                    score = probability(features, classifier)
                # 严格联合版裁决（用户 2026-09-30 定，开发记录 六之前四）：
                #   留日文 = (P < 阈值 ∨ 框级软标记) ∧ 字数 < 护栏
                # 两路是**同一裁决的两个来源**，不是两道独立筛——框级标记来自
                # 数据集（几何票），分类器来自读数（文本+几何票），在成岛后
                # 一起裁决；救回（≥护栏字数）的岛全文翻译、擦字覆盖全部成员框。
                vote = flagged or (score is not None and score < threshold)
                if score is not None and abs(score - threshold) \
                        < float(ART_TEXT_REVIEW_BAND):
                    # 判据没把握 → 照裁决处理，但列进复核清单（用户要求"可疑的交给我判"）
                    review_rows.append({
                        "岛": island_key, "概率": round(score, 3),
                        "决定": "丢" if vote else "翻译",
                        "原文": " ".join(text.split())[:60],
                    })
                if island_key in force_drop:
                    pass              # 人工判定优先：护栏与投票都不拦人工裁决
                elif not vote:
                    continue          # 两路都没说像拟声 → 照常翻译
                elif chars_n >= ART_TEXT_STRICT_MAX_CHARS:
                    continue          # 字数护栏：≥护栏字数的岛结构上不因拟声留日文
                source = ("框级标记+分类器"
                          if flagged and score is not None and score < threshold
                          else ("框级标记" if flagged else "分类器"))
                if flagged and ART_TEXT_SOFT_REVIEW_MIN_CHARS <= chars_n \
                        < ART_TEXT_STRICT_MAX_CHARS:
                    # 复核加厚带（开发记录 六之前四 §5）：字数落在 [5, 护栏) 的被
                    # 标记岛= k=5 若启用就会救回的那批，台词与拟声在所有本地
                    # 信号上同形（交换率 2.6:1 即墙）→ 自动裁决按 k=7 丢，但
                    # 全部列进复核清单交人工定夺（判错写 keep 即救回）。
                    review_rows.append({
                        "岛": island_key,
                        "概率": round(score, 3) if score is not None else "",
                        "决定": "丢",
                        "原文": " ".join(text.split())[:60],
                    })
                _record_art_text_drop(
                    drops, keep, island_key, source, "", island,
                    page_directions[page][iid], text)
                if island_key in keep:
                    dropped_by_keep += 1
                    continue
                if not _guard_allows(page, iid):
                    continue
                art_text_islands.add(island_key)
                page_texts[page][iid] = ""
                dropped_total += 1
                if flagged:
                    dropped_by_flag += 1
                if score is not None and score < threshold:
                    dropped_by_classifier += 1
        print(f"手绘字判据：严格联合版（P<{ART_TEXT_CLASSIFIER_THRESHOLD} ∨ "
              f"框级软标记，护栏 <{ART_TEXT_STRICT_MAX_CHARS} 字）"
              f"留日文 {dropped_total} 个岛（分类器票 {dropped_by_classifier} / "
              f"框级标记票 {dropped_by_flag}，可重叠；权重训自 "
              f"{classifier.get('trained_on', {}).get('islands', '?')} 个岛）")
        if review_rows and dump_path is not None:
            review_path = Path(dump_path).with_name("art_text_review.csv")
            _write_art_text_review(review_path, review_rows)
            print(f"判据没把握、需要人工复核的岛 {len(review_rows)} 个 → "
                  f"{review_path.name}（照阈值先处理，判错就写进 art_text_keep.txt / "
                  f"art_text_force_drop.txt）")

    # 几何侧：逐页用自己的闸门（未改判的页用书级闸门，实测与页级只差 0.03 上下）
    for page, page_gate in ({} if classifier is not None else geometry_pages).items():
        for iid in _art_text_verdicts(
                units[page]["islands"], page_texts[page], page_directions[page],
                flat_gate=page_gate):
            island_key = f"{page}:{iid}"
            _record_art_text_drop(
                drops, keep, island_key, "几何", "",
                units[page]["islands"][iid], page_directions[page][iid],
                page_texts[page][iid])
            if island_key in keep:
                dropped_by_keep += 1
                continue
            if not _guard_allows(page, iid):
                continue                      # 保台词档位：模型说这是台词 → 照常翻译
            art_text_islands.add(island_key)
            page_texts[page][iid] = ""

    # 视觉侧：只把被改判的页送过去（判据只问它拿到的那几页）；
    # 分类器已经拍板时不再调接口。
    if vision_pages and classifier is None:
        vision_units = {page: units[page] for page in vision_pages}
        vision_drop = (vision_judge(vision_units, page_texts)
                       if (vision_judge and measured) else set())
        if vision_drop:
            print(f"手绘字判据：{len(vision_pages)} 页改用视觉判据，"
                  f"丢 {len(vision_drop)} 个岛（拟声/无字）")
        else:
            reason = ("没配视觉接口" if not (vision_judge and measured)
                      else "模型都判成台词")
            print(f"手绘字判据：{len(vision_pages)} 页需要视觉判据但没丢任何岛"
                  f"（{reason}）→ 这些页照常翻译")
        for island_key in vision_drop:
            page, _, iid = island_key.partition(":")
            if not (iid and page in page_texts and iid in page_texts[page]):
                continue
            _record_art_text_drop(
                drops, keep, island_key, "视觉（S/U）", "",
                units[page]["islands"][iid], page_directions[page][iid],
                page_texts[page][iid])
            if island_key in keep:
                dropped_by_keep += 1
                continue
            page_texts[page][iid] = ""
            art_text_islands.add(island_key)
    if art_text_islands:
        print(
            f"丢弃 {len(art_text_islands)} 个手绘字岛（画在画面上、原文 ≤ "
            f"{ART_TEXT_MAX_CHARS} 字）→ 保留原图，不译不嵌"
        )
    if dropped_by_keep:
        print(f"按 art_text_keep.txt 强制保留 {dropped_by_keep} 个岛（不丢、照常翻译）")
    if drops and dump_path is not None:
        _write_art_text_dropped(dump_path, drops, units)
        print(f"被丢清单已写出：{dump_path.name}（同目录 .html 带裁剪图，可核对后加进 "
              f"art_text_keep.txt 救回）")
    if commentary_boxes:
        print(f"按空读数处理 {commentary_boxes} 个 AI 说明性输出框（不是文字）")
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
    drop_art_boxes: bool = False,
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
    erase_inpainter: str = ERASE_INPAINTER,
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
        drop_art_boxes=drop_art_boxes,
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
        erase_inpainter=erase_inpainter,
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


def _dataset_has_bg_flatness(islands_path: Path) -> bool:
    """数据集里有没有手绘字判据要的两个字段（见 ``attach_bg_flatness``）。

    判据的取样矩形来自 ``members_bbox``（成员框原样并集），占比存在
    ``bg_flat_ratio``。缺任一个都说明这份 dataset 是旧规则算出来的。

    读不了（损坏/半截）也当"没有" —— 重建一次比复用坏产物便宜。
    """
    try:
        payload = json.loads(islands_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    islands = payload.get("islands") or []
    return bool(islands) and all(
        isinstance(item, dict)
        and item.get("bg_flat_ratio") is not None
        and item.get("members_bbox")
        for item in islands
    )


def _dataset_has_soft_flags(islands_path: Path) -> bool:
    """数据集是否由"严格联合版"规则构建（islands.json 带 soft_flag_version）。

    软标记（``sfx_soft_flag``）是按需写的——没有框触发放宽判据时全页都没有，
    不能拿"岛里没有这个键"当旧版判据，所以用数据集级标记。
    """
    try:
        payload = json.loads(islands_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return "soft_flag_version" in payload


def _stage2_dataset(images: Path, out_root: Path, pages: list[str],
                    progress: "ProgressFile", *,
                    overwrite: bool = False,
                    box_source: str = "paddle",
                    drop_large_boxes: bool = True,
                    sfx_min_side_ratio: float = SFX_MIN_SIDE_RATIO,
                    sfx_min_side_page: float = SFX_MIN_SIDE_PAGE,
                    drop_art_boxes: bool = False) -> list[str]:
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
            # 规则也参与"新不新"：数据集是规则算出来的产物，规则变了（这里：
            # 多算一个 bg_flat_ratio、或缺严格联合版的框级软标记）旧 dataset 就得
            # 重建——否则改了规则却静默复用旧产物，看起来"改了没用"。
            stale_rules = (
                drop_art_boxes
                and not (_dataset_has_bg_flatness(islands_path)
                         and _dataset_has_soft_flags(islands_path))
            )
            if islands_mtime >= manifest_mtime and not stale_rules:
                continue
            import shutil

            shutil.rmtree(page_dir, ignore_errors=True)
            print(
                f"{page}: dataset 缺手绘字判据字段（bg_flat_ratio/框级软标记），"
                "重建 islands"
                if stale_rules else
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
                drop_art_boxes=drop_art_boxes,
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
    page_texts, page_directions = build_page_texts(
        units, readings, dropped_boxes,
        vision_judge=make_vision_judge(out_root, progress=progress),
        keep=art_text_keep_list(out_root),
        force_drop=art_text_force_drop_list(out_root),
        dump_path=Path(out_root) / "art_text_dropped.csv")
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
                        ai_api_key: str,
                        previous_lines: tuple[str, ...] = ()) -> dict[str, str]:
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
                previous_lines=previous_lines,
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

    # 跨页上下文：上一页的**原文**（不是译文）。只给原文的好处是不依赖
    # "上一页已经翻完"——页仍然可以并行翻译，缓存键也完全确定。
    ordered_pages = sorted(texts, key=lambda name: int(name) if name.isdigit() else 0)
    previous_page_of = {
        name: (ordered_pages[index - 1] if index else None)
        for index, name in enumerate(ordered_pages)
    }
    # 上限：最多 12 条、合计 400 字，避免把 token 预算花在上下文上
    PREVIOUS_MAX_LINES = 12
    PREVIOUS_MAX_CHARS = 400

    def _previous_lines(page: str) -> tuple[str, ...]:
        previous = previous_page_of.get(page)
        lines: list[str] = []
        budget = PREVIOUS_MAX_CHARS
        for text in (texts.get(previous) or {}).values():
            cleaned = " ".join(str(text or "").split())
            if not cleaned:
                continue
            if len(lines) >= PREVIOUS_MAX_LINES or budget - len(cleaned) < 0:
                break
            budget -= len(cleaned)
            lines.append(cleaned)
        return tuple(lines)

    def _translate(page: str) -> tuple[str, dict[str, str] | None]:
        try:
            return page, _translate_one_page(
                page, texts[page], dataset_dir=dataset_dir,
                glossary=glossary, cache=cache, batch_cache=batch_cache,
                ai_url=ai_url, ai_model=ai_model, ai_api_key=ai_api_key,
                previous_lines=_previous_lines(page),
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
                    erase_inpainter: str = ERASE_INPAINTER,
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
        # **按输入指纹判断要不要重嵌**（用户要求"只重新嵌受影响的岛/页"）。
        # 只看"文件在不在"是不够的：编辑 art_text_keep.txt / art_text_force_drop.txt
        # 之后，被改的页输入变了但输出文件还在，旧逻辑会把它一起跳过（漏嵌），
        # 而不带 --resume 又会把 160 页全重嵌。指纹 = 这一页真正参与嵌字的东西：
        # 岛→译文、方向、被丢的框集合、以及影响排版的几个配置。任一变化 → 该页重嵌。
        key_path = typeset_dir / f"{page}.typeset.key"
        fingerprint = _typeset_fingerprint(
            page, translations_page, page_directions.get(page),
            dropped_boxes.get(page, set()),
            {"min_font_size": min_font_size, "font_name": font_name,
             "stroke_divisor": stroke_divisor, "row_gap_divisor": row_gap_divisor,
             "column_gap_divisor": column_gap_divisor,
             "tracking_divisor": tracking_divisor, "text_color": text_color,
             "stroke_color": stroke_color,
             "background_color": background_color},
        )
        if out_png.is_file() and key_path.is_file():
            try:
                if key_path.read_text(encoding="utf-8").strip() == fingerprint:
                    progress.log(f"skip {page}: 输入未变（按指纹跳过重嵌）")
                    return False
            except (OSError, UnicodeError):
                pass
        elif resume and out_png.is_file():
            # 老产物没有指纹 → 保持 --resume 的旧语义（文件在就跳过）
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
        key=_typeset_page_sort_key,
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
            drop_art_boxes=kwargs.get("drop_art_boxes", False),
            sfx_min_side_ratio=kwargs.get("sfx_min_side_ratio", SFX_MIN_SIDE_RATIO),
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
            # 修复后端进键：换 LaMa/FSR 会自动重擦（否则会串用另一种后端的旧结果）
            "inpainter": str(getattr(config, "inpainter", "auto")),
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
    # 擦字修复后端：auto = 有 LaMa 就用它（更准但 CPU 上慢 4~6 倍）/ fsr = 快
    erase_inpainter: str = ERASE_INPAINTER,
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
    # **还要"这一轮确实读出了原文"**：`translations.json` 是跨轮持久化的，判据变了
    # （或某岛被判成手绘字）之后只跑 `--stage typeset` 时旧译文仍在文件里；只看
    # 译文就会照旧擦字嵌字，让人以为开关没生效。`sources` 为 None 表示调用方没给
    # 原文，此时保持原行为（不知道原文就不拦）。
    erase_ids = {
        island_id for island_id in boxed_ids
        if translations.get(island_id)
        and (sources is None or sources.get(island_id))
    }
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
    # 预排一遍布局（只算不画）：连 12px 绝对下限也放不下、只能被裁的岛
    # **直接跳过**——擦掉原文却写不出看得见的字，比保留原文更差（实测 丙书 有 17 个
    # 这种岛，框只有 8~41px）。判定输入与真正嵌字完全一致，所以不会"预排说放不下、
    # 实排又放下了"。
    unfittable = _unfittable_island_ids(
        canonical_dataset, translations, config,
        directions=directions, union_overrides=union_overrides,
        capped_islands=capped_islands, font_ceiling_px=font_ceiling_px,
    )
    if unfittable:
        print(f"  {page}: {len(unfittable)} 个岛放不下 → 保留原图（不擦不写）")
        erase_ids -= unfittable
        translations = {
            island_id: text for island_id, text in translations.items()
            if island_id not in unfittable
        }
        active_dataset = dataclasses.replace(
            canonical_dataset,
            islands=tuple(i for i in canonical_dataset.islands if i.id in erase_ids),
        )
        if not erase_ids:
            with Image.open(dataset_dir / page / "original.jpg") as image:
                image.convert("RGB").save(typeset_dir / f"{page}.typeset.png")
            return None
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
        ErasureConfig(inpainter=erase_inpainter), 800.0,
    )
    result = typeset_island_page(
        erased, canonical_dataset, translations, blue_lines, config=config,
        direction_overrides=directions,
        union_overrides=union_overrides,
        capped_islands=capped_islands,
        font_ceiling_px=font_ceiling_px,
    )
    _save_typeset_png(result.image, typeset_dir / f"{page}.typeset.png")
    return result


def _unfittable_island_ids(
    dataset, translations, config, *, directions=None,
    union_overrides=None, capped_islands=None, font_ceiling_px=None,
) -> set[str]:
    """布局预排：找出"连 12px 绝对下限都放不下、只能被裁"的岛。

    这类岛擦掉原文却写不下字（实测 丙书 有 17 个，框只有 8~41px，4~7 个中文字
    都塞不进），成品里是一个看不见的小点 —— 保留原图反而更好。判定输入与真正
    嵌字完全一致，所以不会出现"预排说放不下、实排又放下了"。
    """
    from PIL import Image as _Image

    from manga_localizer.island_typeset import typeset_island_page

    if not translations:
        return set()
    blank = _Image.new("RGB", dataset.image_size, (255, 255, 255))
    probe = typeset_island_page(
        blank, dataset, translations, [], config=config,
        direction_overrides=directions, union_overrides=union_overrides,
        capped_islands=capped_islands, font_ceiling_px=font_ceiling_px,
    )
    return {
        audit.island_id for audit in probe.audits
        if any("rendered and clipped" in reason for reason in audit.reasons)
    }


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
        "--drop-art-boxes", action="store_true",
        help="把画在画面上的手绘字当非台词丢掉（岛底不平坦且原文 ≤ %d 字的岛"
             "不译、不擦、不嵌，保留原图）。默认关——这条判据只看几何，分不开"
             "拟声词与「没气泡、直接画在画面上的对白」（实测 159 个命中岛里 41 个"
             "是真对白），要开请先确认这本书的手绘字以拟声为主" % ART_TEXT_MAX_CHARS,
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
    parser.add_argument(
        "--erase-inpainter", choices=["auto", "lama", "fsr"], default="",
        help="擦字修复后端：auto=有 LaMa 就用（更准、CPU 上慢 4~6 倍），fsr=快")
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
        drop_art_boxes=args.drop_art_boxes,
        sfx_min_side_ratio=args.sfx_min_side_ratio,
        sfx_min_side_page=args.sfx_min_side_page,
        sfx_glossary_min_count=args.sfx_glossary_min_count,
        sfx_max_font_size=args.sfx_max_font_size,
        short_source_max_chars=args.short_source_max_chars,
        page_font_ceiling_ratio=args.page_font_ceiling_ratio,
        erase_inpainter=args.erase_inpainter or _config.get("erase_inpainter", ERASE_INPAINTER),
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
