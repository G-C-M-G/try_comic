"""OCR 引擎横向对比：PaddleOCR / Tesseract 全页自动检测 vs Photos 蓝框。

用法（root venv）：
    venv/Scripts/python.exe ocr_box_compare.py --pages 1          # 单页试跑
    venv/Scripts/python.exe ocr_box_compare.py                    # 全部有产物的页

对每一页生成三联对比图（蓝框 check / PaddleOCR 框 / Tesseract 框），
输出到 --out 目录（默认桌面 图源样张_ocr对比）。
PaddleOCR 部分经 subprocess 用 paddle_venv 跑（lang=japan, CPU）；
Tesseract 部分直接调 D:/tools 的 CLI（jpn + jpn_vert, psm 11, conf≥30）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
TESSERACT = Path(r"D:\tools\Tesseract-OCR\tesseract.exe")
PADDLE_PYTHON = ROOT / "paddle_venv" / "Scripts" / "python.exe"
DEFAULT_OUT = Path(r"C:\Users\YLP\Desktop\图源样张_ocr对比")
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

PADDLE_SCRIPT = r"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[3])
PADDLE_CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(PADDLE_CACHE))
os.environ.setdefault("USERPROFILE", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

paths = json.load(open(sys.argv[1], encoding="utf-8"))
out, err = {}, None
try:
    from paddleocr import PaddleOCR
    # 与 paddle_ocr_worker._build_engine 同配置：本地 PP-OCRv6 模型 +
    # enable_mkldnn=False（onednn/PIR 在本机会 NotImplementedError）
    model_paths = {}
    det_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_det"
    rec_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_rec"
    if (det_dir / "inference.json").is_file() and (rec_dir / "inference.json").is_file():
        model_paths = {
            "text_detection_model_dir": str(det_dir),
            "text_recognition_model_dir": str(rec_dir),
        }
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1536,
        text_det_limit_type="max",
        text_det_thresh=0.20,
        text_det_box_thresh=0.40,
        text_det_unclip_ratio=1.25,
        device="cpu",
        enable_mkldnn=False,
        **model_paths,
    )
    for i, p in enumerate(paths, 1):
        try:
            items = []
            for result in ocr.predict(p):
                payload = result.json["res"]
                polys = payload.get("rec_polys", [])
                scores = payload.get("rec_scores", [])
                texts = payload.get("rec_texts", [])
                for poly, score, text in zip(polys, scores, texts):
                    pts = np.asarray(poly).reshape(-1, 2)
                    items.append({
                        "poly": [[int(x), int(y)] for x, y in pts],
                        "score": float(score),
                        "text": str(text),
                    })
            out[p] = items
            print(f"paddle {i}/{len(paths)} {p}: {len(items)} boxes", flush=True)
        except Exception as e:  # noqa: BLE001 - 单页失败不拖垮整批
            out[p] = [{"error": repr(e)}]
            print(f"paddle {i}/{len(paths)} {p}: ERROR {e!r}", flush=True)
except Exception as e:  # noqa: BLE001
    err = repr(e)
prev = {}
if os.path.isfile(sys.argv[2]):
    try:
        prev = json.load(open(sys.argv[2], encoding="utf-8")).get("results", {})
    except Exception:  # noqa: BLE001 - 缓存损坏则重建
        prev = {}
prev.update(out)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump({"error": err, "results": prev}, f, ensure_ascii=False)
"""


def _paddle_items_broken(items) -> bool:
    return (
        isinstance(items, list) and bool(items)
        and isinstance(items[0], dict) and "error" in items[0]
    )


def run_paddle(pages: list[str], book: Path, cache: Path) -> dict:
    sources = {page: str(book / "dataset" / page / "original.jpg") for page in pages}
    existing: dict = {}
    if cache.is_file():
        existing = json.loads(cache.read_text(encoding="utf-8")).get("results", {})
    todo = [
        path for path in sources.values()
        if not existing.get(path) or _paddle_items_broken(existing[path])
    ]
    if todo:
        payload = cache.with_name("paddle_jobs.json")
        payload.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.run(
            [str(PADDLE_PYTHON), "-c", PADDLE_SCRIPT, str(payload), str(cache), str(ROOT)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3000,
        )
        if proc.returncode != 0 and not cache.is_file():
            raise RuntimeError(f"paddle subprocess failed rc={proc.returncode}\n"
                               f"{(proc.stderr or '')[-2000:]}")
    result = json.loads(cache.read_text(encoding="utf-8"))
    by_path = result.get("results", {})
    return {page: by_path.get(path, []) for page, path in sources.items()}


def natural_keys(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in __import__("re").split(r"(\d+)", text)]


def discover_pages(book: Path) -> list[str]:
    dataset = book / "dataset"
    check = book / "check"
    with_original = {
        d.name for d in dataset.iterdir() if d.is_dir() and (d / "original.jpg").is_file()
    } if dataset.is_dir() else set()
    with_check = {
        p.name[: -len(".bluebox_check.png")]  # stem 会被第二个点截断成 "1.bluebox_check"
        for p in check.glob("*.bluebox_check.png")
    } if check.is_dir() else set()
    return sorted(with_original & with_check, key=lambda name: natural_keys(name))


def tesseract_boxes(image: Path, psm: int = 11, min_conf: float = 30.0) -> list[dict]:
    boxes: list[dict] = []
    for lang in ("jpn", "jpn_vert"):
        cmd = [str(TESSERACT), str(image), "stdout", "-l", lang, "--psm", str(psm), "tsv"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=180,
            )
        except subprocess.TimeoutExpired:
            print(f"tesseract timeout: {image.name} {lang}", file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"tesseract rc={proc.returncode}: {image.name} {lang}: "
                  f"{(proc.stderr or '').strip()[:200]}", file=sys.stderr)
            continue
        for line in (proc.stdout or "").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            try:
                conf = float(parts[10])
            except ValueError:
                continue
            text = parts[11].strip()
            if conf < min_conf or not text:
                continue
            left, top, width, height = (int(parts[6]), int(parts[7]),
                                        int(parts[8]), int(parts[9]))
            if width <= 0 or height <= 0:
                continue
            boxes.append({"box": [left, top, width, height], "lang": lang,
                          "text": text, "conf": conf})
    return boxes


def annotate(original: Image.Image, boxes: list[dict], color) -> Image.Image:
    canvas = original.copy()
    draw = ImageDraw.Draw(canvas)
    for item in boxes:
        if "error" in item:
            continue
        if "poly" in item:
            pts = [tuple(p) for p in item["poly"]]
            draw.polygon(pts, outline=color, width=4)
        else:
            l, t, w, h = item["box"]
            draw.rectangle([l, t, l + w, t + h], outline=color, width=4)
    return canvas


def compose(page: str, check_path: Path, original_path: Path,
            paddle_items: list, tess_items: list, out_path: Path,
            panel_width: int) -> None:
    panels = [
        ("Photos 蓝框（人工框选，现流程基准）", Image.open(check_path).convert("RGB"), None),
        ("PaddleOCR 3.7 全页自动检测（lang=japan, CPU）",
         Image.open(original_path).convert("RGB"), (paddle_items, (255, 40, 40))),
        ("Tesseract 5 全页自动检测（jpn + jpn_vert, psm 11, conf≥30）",
         Image.open(original_path).convert("RGB"), (tess_items, (0, 150, 255))),
    ]
    font = ImageFont.truetype(FONT_PATH, 26)
    label_h = 38
    gap = 6
    columns = []
    for label, image, draw_args in panels:
        ratio = panel_width / image.width
        resized = image.resize((panel_width, max(1, round(image.height * ratio))),
                               Image.LANCZOS)
        if draw_args:
            # 在原分辨率上画框再缩放会丢细线；这里按缩放比例重画更稳
            items, color = draw_args
            scale = panel_width / image.width
            redraw = []
            for item in items:
                if "error" in item:
                    continue
                if "poly" in item:
                    redraw.append({"poly": [[x * scale, y * scale] for x, y in item["poly"]]})
                else:
                    l, t, w, h = item["box"]
                    redraw.append({"box": [l * scale, t * scale, w * scale, h * scale]})
            annotated = annotate(resized, redraw, color)
        else:
            annotated = resized
        strip = Image.new("RGB", (panel_width, label_h), (24, 24, 24))
        ImageDraw.Draw(strip).text((8, 4), label, font=font, fill=(240, 240, 240))
        column = Image.new("RGB", (panel_width, label_h + annotated.height), (10, 10, 10))
        column.paste(strip, (0, 0))
        column.paste(annotated, (0, label_h))
        columns.append(column)
    total_h = max(c.height for c in columns)
    total_w = len(columns) * panel_width + gap * (len(columns) - 1)
    board = Image.new("RGB", (total_w, total_h), (10, 10, 10))
    x = 0
    for column in columns:
        board.paste(column, (x, 0))
        x += panel_width + gap
    board.save(out_path, quality=88)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="sample-book")
    parser.add_argument("--pages", default="", help="逗号分隔页码，空=全部")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--width", type=int, default=900, help="对比图面板宽度")
    parser.add_argument("--skip-tesseract", action="store_true")
    args = parser.parse_args(argv)

    book = ROOT / "outputs" / "oneclick" / args.book
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "var" / "ocr_compare"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pages = discover_pages(book)
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        pages = [p for p in pages if p in wanted]
    if not pages:
        print("no pages with both dataset/original.jpg and check image")
        return 2
    print(f"pages: {len(pages)} -> {out_dir}")

    paddle_boxes = run_paddle(pages, book, cache_dir / "paddle_boxes.json")
    paddle_err = None
    if isinstance(paddle_boxes, dict) and paddle_boxes.get("error"):
        paddle_err = paddle_boxes["error"]
    tess_cache = cache_dir / "tesseract_boxes.json"
    tess_all = json.loads(tess_cache.read_text(encoding="utf-8")) if tess_cache.is_file() else {}
    if not args.skip_tesseract:
        for i, page in enumerate(pages, 1):
            if page in tess_all:
                continue
            tess_all[page] = tesseract_boxes(book / "dataset" / page / "original.jpg")
            tess_cache.write_text(json.dumps(tess_all, ensure_ascii=False),
                                  encoding="utf-8")
            print(f"tesseract {i}/{len(pages)} page {page}: "
                  f"{len(tess_all[page])} boxes", flush=True)

    stats = {"page": [], "blue": [], "paddle": [], "tess": []}
    for i, page in enumerate(pages, 1):
        paddle_items = paddle_boxes.get(page, []) if isinstance(paddle_boxes, dict) else []
        tess_items = tess_all.get(page, [])
        blue_count = 0
        blocks_json = book / "dataset" / page / "selection_blocks.json"
        if blocks_json.is_file():
            data = json.loads(blocks_json.read_text(encoding="utf-8"))
            blocks = data.get("selection_blocks") or []
            blue_count = sum(1 for b in blocks if not b.get("page_scale"))
        stats["page"].append(page)
        stats["blue"].append(blue_count)
        stats["paddle"].append(sum(1 for it in paddle_items if "error" not in it))
        stats["tess"].append(sum(1 for it in tess_items))
        compose(page, book / "check" / f"{page}.bluebox_check.png",
                book / "dataset" / page / "original.jpg",
                paddle_items, tess_items,
                out_dir / f"{page}_对比.png", args.width)
        print(f"compose {i}/{len(pages)} page {page}", flush=True)

    print("\n=== 每页框数（蓝框 / PaddleOCR / Tesseract）===")
    for page, b, p, t in zip(stats["page"], stats["blue"], stats["paddle"], stats["tess"]):
        print(f"  page {page:>4}: {b:>3} / {p:>3} / {t:>3}")
    print(f"total: blue={sum(stats['blue'])} paddle={sum(stats['paddle'])} "
          f"tess={sum(stats['tess'])}")
    if paddle_err:
        print(f"paddle engine error: {paddle_err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
