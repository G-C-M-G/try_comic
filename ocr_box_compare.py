"""OCR 引擎横向对比：PaddleOCR 全页自动检测 vs Photos 蓝框。

用法（root venv）：
    venv/Scripts/python.exe ocr_box_compare.py --pages 1          # 单页试跑
    venv/Scripts/python.exe ocr_box_compare.py                    # 全部有产物的页
    venv/Scripts/python.exe ocr_box_compare.py --ctd              # 追加 CTD 联

默认两联（蓝框 check / PaddleOCR），输出到 {书目录}/ocr_compare/（横排）。
`--ctd` 追加 comic-text-detector 联——实测精确率仅 39.2%（过度识别严重，
把网点/木纹/衣褶都当文字），默认关闭，仅在复查该结论时开启。结果落盘
var/ocr_compare/ 缓存，可增量重跑。

PaddleOCR 经 subprocess 用 paddle_venv 跑（与 paddle_ocr_worker 同配置，
CPU）。Tesseract 实测过检 8.5 倍（4328 框，全是字符碎片）已淘汰移除（见
开发记录 选型一节）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PADDLE_PYTHON = ROOT / "paddle_venv" / "Scripts" / "python.exe"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

PADDLE_SCRIPT = r"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[3])
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "auto"
PADDLE_CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(PADDLE_CACHE))
os.environ.setdefault("USERPROFILE", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

if DEVICE == "auto":
    # PaddleOCR 只认 cpu / gpu；auto 在此解析（与 paddle_ocr_worker._resolve_device 同语义）
    try:
        import paddle
        DEVICE = ("gpu" if paddle.device.is_compiled_with_cuda()
                  and paddle.device.cuda.device_count() > 0 else "cpu")
    except Exception:
        DEVICE = "cpu"
    print(f"device auto -> {DEVICE}", flush=True)

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
        device=DEVICE,
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
    json.dump({"device": DEVICE, "error": err, "results": prev}, f, ensure_ascii=False)
"""


_CTD = {"session": None, "mode": None}


def _ctd_session():
    if _CTD["session"] is None:
        import onnxruntime as ort

        model = ROOT / "models" / "comic-text-detector-onnx" / "comic-text-detector.onnx"
        if not model.is_file():
            raise FileNotFoundError(f"CTD 模型不存在: {model}")
        _CTD["session"] = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return _CTD["session"]


def _ctd_seg(image: Image.Image, mode: str):
    import numpy as np

    x = np.asarray(image, np.float32) / 255.0
    if mode == "imagenet":
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32)
    pixel = x.transpose(2, 0, 1)[None]
    return _ctd_session().run(["seg"], {"images": pixel})[0][0, 0]


def _ctd_pick_mode(image_path: Path) -> str:
    """官方没给推理代码，预处理模式用首页实测标定：取文字占比在合理区间的那个。"""
    if _CTD["mode"] is None:
        image = Image.open(image_path).convert("RGB").resize((1024, 1024), Image.BICUBIC)
        fractions = {}
        for mode in ("imagenet", "unit"):
            seg = _ctd_seg(image, mode)
            fractions[mode] = float((seg > 0.4).mean())
            print(f"ctd calib {mode}: fg={fractions[mode]:.3f} "
                  f"seg range [{seg.min():.3f},{seg.max():.3f}]", flush=True)
        # 漫画页文字占比通常在百分之几；网点/画面误入会显著抬高前景占比
        _CTD["mode"] = min(fractions, key=lambda m: abs(fractions[m] - 0.05))
        print(f"ctd calib -> mode={_CTD['mode']}", flush=True)
    return _CTD["mode"]


def ctd_boxes(image_path: Path, threshold: float = 0.4) -> list[dict]:
    """comic-text-detector 的 seg 掩膜 → 膨胀融合 → 连通域 → 原图坐标框。"""
    import cv2
    import numpy as np

    mode = _ctd_pick_mode(image_path)
    image = Image.open(image_path).convert("RGB").resize((1024, 1024), Image.BICUBIC)
    seg = _ctd_seg(image, mode)
    mask = ((seg > threshold).astype(np.uint8)) * 255
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    width, height = Image.open(image_path).size
    sx, sy = width / 1024.0, height / 1024.0
    boxes = []
    for i in range(1, count):
        px, py, pw, ph, area = (int(v) for v in stats[i])
        if area < 80 or pw < 8 or ph < 8:
            continue
        boxes.append({
            "poly": [[int(px * sx), int(py * sy)],
                     [int((px + pw) * sx), int(py * sy)],
                     [int((px + pw) * sx), int((py + ph) * sy)],
                     [int(px * sx), int((py + ph) * sy)]],
            "score": 1.0, "text": "",
        })
    return boxes


def _paddle_items_broken(items) -> bool:
    return (
        isinstance(items, list) and bool(items)
        and isinstance(items[0], dict) and "error" in items[0]
    )


def run_paddle(pages: list[str], book: Path, cache: Path, device: str = "auto") -> dict:
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
            [str(PADDLE_PYTHON), "-c", PADDLE_SCRIPT, str(payload), str(cache), str(ROOT), device],
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
            paddle_items: list, out_path: Path,
            panel_width: int, ctd_items: list | None = None,
            device_label: str = "auto") -> None:
    panels = [
        ("Photos 蓝框（人工框选，现流程基准）", Image.open(check_path).convert("RGB"), None),
        (f"PaddleOCR 3.7 全页自动检测（PP-OCRv6, {device_label}）",
         Image.open(original_path).convert("RGB"), (paddle_items, (255, 40, 40))),
    ]
    if ctd_items is not None:
        panels.append(
            ("comic-text-detector 全页自动检测（seg→连通域, ONNX）",
             Image.open(original_path).convert("RGB"), (ctd_items, (0, 175, 0))),
        )
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
    _save_with_retry(board, out_path)


def _save_with_retry(image, out_path: Path, attempts: int = 5) -> None:
    """原子写 + 重试：杀软扫描/预览句柄会造成 Windows 上的瞬时占用
    （OSError 22/13），单次失败不代表目标不可写。"""
    temporary = out_path.with_name(f".{out_path.name}.{__import__('os').getpid()}.tmp")
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            image.save(temporary, format="PNG")
            temporary.replace(out_path)
            return
        except OSError as error:
            last_error = error
            __import__("time").sleep(0.3 * (attempt + 1))
    temporary.unlink(missing_ok=True)
    raise OSError(f"cannot save {out_path}: {last_error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="sample-book")
    parser.add_argument("--device", default="auto",
                        help="Paddle 设备：auto（有 GPU 用 GPU）/ cpu / gpu")
    parser.add_argument("--pages", default="", help="逗号分隔页码，空=全部")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认 {书目录}/ocr_compare）")
    parser.add_argument("--width", type=int, default=900, help="对比图面板宽度")
    parser.add_argument(
        "--ctd", action="store_true",
        help="追加 comic-text-detector 联（实测精确率仅 39.2%，过度识别严重，"
             "默认关闭；仅在复查该结论时开启）",
    )
    args = parser.parse_args(argv)

    book = ROOT / "outputs" / "oneclick" / args.book
    out_dir = Path(args.out) if args.out else book / "ocr_compare"
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

    paddle_boxes = run_paddle(pages, book, cache_dir / "paddle_boxes.json", args.device)
    paddle_err = None
    if isinstance(paddle_boxes, dict) and paddle_boxes.get("error"):
        paddle_err = paddle_boxes["error"]

    device_label = "auto"
    try:
        device_label = json.loads(
            (cache_dir / "paddle_boxes.json").read_text(encoding="utf-8")
        ).get("device") or args.device
    except (OSError, ValueError):
        device_label = args.device

    ctd_all: dict = {}
    if args.ctd and (ROOT / "models" / "comic-text-detector-onnx" / "comic-text-detector.onnx").is_file():
        ctd_cache = cache_dir / "ctd_boxes.json"
        ctd_all = json.loads(ctd_cache.read_text(encoding="utf-8")) if ctd_cache.is_file() else {}
        for i, page in enumerate(pages, 1):
            if page in ctd_all:
                continue
            ctd_all[page] = ctd_boxes(book / "dataset" / page / "original.jpg")
            ctd_cache.write_text(json.dumps(ctd_all, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"ctd {i}/{len(pages)} page {page}: "
                  f"{len(ctd_all[page])} boxes", flush=True)
    elif args.ctd:
        print("ctd: 模型未下载（models/comic-text-detector-onnx），跳过该联")

    def _bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def _uncovered(blue_bbox, items):
        if not items:
            return True
        l, t, r, b = blue_bbox
        area = max(1, (r - l) * (b - t))
        best = 0.0
        for item in items:
            cl, ct, cr, cb = _bbox(item["poly"])
            inter = max(0, min(cr, r) - max(cl, l)) * max(0, min(cb, b) - max(ct, t))
            best = max(best, inter / area)
        return best < 0.3

    def _is_true_positive(poly, blues, threshold: float = 0.5) -> bool:
        """引擎框是否落在某个蓝框内（交集 / 自身面积 ≥ threshold）。"""
        cl, ct, cr, cb = _bbox(poly)
        area = max(1, (cr - cl) * (cb - ct))
        for l, t, r, b in blues:
            inter = max(0, min(cr, r) - max(cl, l)) * max(0, min(cb, b) - max(ct, t))
            if inter / area >= threshold:
                return True
        return False

    # 精确率必须和召回率一起看：只看漏检（召回）会漏掉"多出一堆假框"的引擎
    # （CTD 的教训，见 开发记录 选型一节）。
    metrics = {"paddle": {"tp": 0, "fp": 0, "fn": 0}, "ctd": {"tp": 0, "fp": 0, "fn": 0}}
    per_page = {"page": [], "blue": [], "paddle": [], "ctd": [],
                "miss_paddle": [], "miss_ctd": []}
    for i, page in enumerate(pages, 1):
        paddle_items = [it for it in (paddle_boxes.get(page, [])
                                      if isinstance(paddle_boxes, dict) else [])
                        if "error" not in it]
        ctd_items = ctd_all.get(page)
        blues = []
        blocks_json = book / "dataset" / page / "selection_blocks.json"
        if blocks_json.is_file():
            data = json.loads(blocks_json.read_text(encoding="utf-8"))
            blues = [b["image_bbox"] for b in (data.get("selection_blocks") or [])
                     if not b.get("page_scale")]
        per_page["page"].append(page)
        per_page["blue"].append(len(blues))
        per_page["paddle"].append(len(paddle_items))
        per_page["ctd"].append(len(ctd_items or []))
        per_page["miss_paddle"].append(sum(1 for bb in blues
                                           if _uncovered(bb, paddle_items)))
        per_page["miss_ctd"].append(sum(1 for bb in blues
                                        if _uncovered(bb, ctd_items or [])))
        for engine, items in (("paddle", paddle_items), ("ctd", ctd_items)):
            if items is None:
                continue
            for item in items:
                if _is_true_positive(item["poly"], blues):
                    metrics[engine]["tp"] += 1
                else:
                    metrics[engine]["fp"] += 1
            metrics[engine]["fn"] += sum(1 for bb in blues if _uncovered(bb, items))
        compose(page, book / "check" / f"{page}.bluebox_check.png",
                book / "dataset" / page / "original.jpg",
                paddle_items,
                out_dir / f"{page}_对比.png", args.width, ctd_items=ctd_items,
                device_label=device_label)
        print(f"compose {i}/{len(pages)} page {page}", flush=True)

    stats = per_page
    print("\n=== 每页框数（蓝框 / PaddleOCR | 蓝框漏检 P）===")
    for page, b, p, mp in zip(stats["page"], stats["blue"],
                              stats["paddle"], stats["miss_paddle"]):
        print(f"  page {page:>4}: {b:>3} / {p:>3} | {mp}")
    print(f"total: blue={sum(stats['blue'])} paddle={sum(stats['paddle'])}")
    print("\n=== 精确率 / 召回率（引擎框落在蓝框内=真阳性，阈值 50%）===")
    for engine, name in (("paddle", "PaddleOCR"), ("ctd", "CTD")):
        m = metrics[engine]
        if not (m["tp"] or m["fp"] or m["fn"]):
            continue
        precision = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if m["tp"] + m["fn"] else 0.0
        print(f"  {name:<10} TP={m['tp']:<4} FP={m['fp']:<4} FN={m['fn']:<3} "
              f"精确率={precision:.1%} 召回率={recall:.1%}")
    # 粒度诊断：上面的精确率把"引擎框覆盖多个蓝框"也算成假阳性。当引擎按行/
    # 整句出框、而蓝框被切得过细（如整页大字被逐字切分）时，精确率会失真，
    # 必须单独指出，否则会误判引擎"差"。
    granularity = []
    for page in pages:
        paddle_items = [it for it in (paddle_boxes.get(page, [])
                                      if isinstance(paddle_boxes, dict) else [])
                        if "error" not in it]
        blues = []
        blocks_json = book / "dataset" / page / "selection_blocks.json"
        if blocks_json.is_file():
            data_blocks = json.loads(blocks_json.read_text(encoding="utf-8"))
            blues = [b["image_bbox"] for b in (data_blocks.get("selection_blocks") or [])
                     if not b.get("page_scale")]
        if not blues or not paddle_items:
            continue
        merged = 0
        for item in paddle_items:
            cl, ct, cr, cb = _bbox(item["poly"])
            covered = 0
            for l, t, r, b in blues:
                area = max(1, (r - l) * (b - t))
                inter = max(0, min(cr, r) - max(cl, l)) * max(0, min(cb, b) - max(ct, t))
                if inter / area >= 0.5:
                    covered += 1
            if covered >= 2:
                merged += 1
        if merged:
            granularity.append((page, merged, len(paddle_items)))
    if granularity:
        print("\n=== 粒度诊断（引擎框覆盖了 ≥2 个蓝框 → 精确率会失真）===")
        for page, merged, total in granularity:
            print(f"  page {page}: {merged}/{total} 个引擎框各覆盖多个蓝框 "
                  f"→ 该页蓝框切分比引擎更细，精确率不可直接采信")
    if paddle_err:
        print(f"paddle engine error: {paddle_err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
