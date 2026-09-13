"""Paddle 检测档位对比：v6_medium（现用）vs v5_server（更准的重档）。

在同样的页面上跑两个检测模型，各自算对蓝框的精确率/召回率 + 耗时，
用数据决定要不要给采集期换更强的检测档。

用法（root venv）：
    venv/Scripts/python.exe paddle_det_tiers.py --pages 14,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PADDLE_PYTHON = ROOT / "paddle_venv" / "Scripts" / "python.exe"
PADDLE_CACHE = ROOT / "models" / "paddle"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

SCRIPT = r"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[3])
CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(CACHE))
os.environ.setdefault("USERPROFILE", str(CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE))
# 允许下载 server 档检测模型（paddle_ocr_worker 里禁用了下载，这里需要联网）

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cpu"

from paddleocr import PaddleOCR

models = {}
local = {
    "PP-OCRv6_medium_det": CACHE / "official_models" / "PP-OCRv6_medium_det",
}
for name, directory in local.items():
    if (directory / "inference.json").is_file():
        models[name] = {"text_detection_model_dir": str(directory)}

results = {}
for tier in ("PP-OCRv6_medium_det", "PP-OCRv5_server_det"):
    try:
        engine = PaddleOCR(
            text_detection_model_name=tier,
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
            **models.get(tier, {}),
        )
    except Exception as error:  # noqa: BLE001
        results[tier] = {"error": repr(error)}
        print(f"tier {tier}: BUILD FAILED {error!r}", flush=True)
        continue
    per_page = {}
    for page, path in jobs.items():
        t0 = time.monotonic()
        try:
            items = []
            for r in engine.predict(path):
                payload = r.json["res"]
                polys = payload.get("rec_polys") or payload.get("dt_polys") or []
                for poly in polys:
                    pts = np.asarray(poly).reshape(-1, 2)
                    items.append([[int(x), int(y)] for x, y in pts])
            elapsed = time.monotonic() - t0
            per_page[page] = {"items": items, "seconds": round(elapsed, 2)}
            print(f"tier {tier} page {page}: {len(items)} boxes in {elapsed:.1f}s",
                  flush=True)
        except Exception as error:  # noqa: BLE001
            per_page[page] = {"error": repr(error)}
            print(f"tier {tier} page {page}: FAILED {error!r}", flush=True)
    results[tier] = {"pages": per_page}

out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
"""


def bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def score(items, blues, tp_threshold=0.5, cover_threshold=0.3):
    tp = fp = 0
    for poly in items:
        cl, ct, cr, cb = bbox(poly)
        area = max(1, (cr - cl) * (cb - ct))
        if any(
            max(0, min(cr, r) - max(cl, l)) * max(0, min(cb, b) - max(ct, t)) / area
            >= tp_threshold
            for l, t, r, b in blues
        ):
            tp += 1
        else:
            fp += 1
    fn = 0
    for l, t, r, b in blues:
        area = max(1, (r - l) * (b - t))
        if not any(
            max(0, min(cr, r) - max(cl, l)) * max(0, min(cb, b) - max(ct, t)) / area
            >= cover_threshold
            for cl, ct, cr, cb in (bbox(p) for p in items)
        ):
            fn += 1
    return tp, fp, fn


def discover_pages(book: Path) -> list[str]:
    dataset = book / "dataset"
    if not dataset.is_dir():
        return []
    pages = [
        d.name for d in dataset.iterdir()
        if d.is_dir()
        and (d / "original.jpg").is_file()
        and (d / "selection_blocks.json").is_file()
    ]
    return sorted(pages, key=lambda name: (int(name) if name.isdigit() else 10**9, name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="sample-book")
    parser.add_argument("--pages", default="14,1,2,3,5,6",
                        help="逗号分隔页码，或 all = 全部有产物的页")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-images", action="store_true",
                        help="只出统计与每页对照表，不生成对比图（全量时省时间）")
    parser.add_argument("--device", default="gpu",
                        help="paddle 设备：gpu / cpu（默认 gpu）")
    parser.add_argument("--cache", default=None,
                        help="结果缓存路径（默认 var/ocr_compare/det_tiers_{device}.json）")
    args = parser.parse_args(argv)

    book = ROOT / "outputs" / "oneclick" / args.book
    out_dir = Path(args.out) if args.out else book / "ocr_compare" / "det_tiers"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else (
        ROOT / "var" / "ocr_compare" / f"det_tiers_{args.device}.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)

    if args.pages.strip().lower() == "all":
        pages = discover_pages(book)
    else:
        pages = [p.strip() for p in args.pages.split(",") if p.strip()]
    if not pages:
        print("no pages found")
        return 2
    jobs = {p: str(book / "dataset" / p / "original.jpg") for p in pages}

    # 增量：缓存里已有的页不重跑，只补缺的（首轮 4 页的缓存可复用）
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    todo = {
        page: path for page, path in jobs.items()
        if any(
            page not in payload.get("pages", {})
            for payload in data.values() if "error" not in payload
        ) or not data
    }
    if todo:
        payload_path = cache.with_name("det_tiers_jobs.json")
        payload_path.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.run(
            [str(PADDLE_PYTHON), "-c", SCRIPT, str(payload_path), str(cache),
             str(ROOT), args.device],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=7200,
        )
        print(proc.stdout[-4000:], flush=True)
        if not cache.is_file():
            print(proc.stderr[-3000:], file=sys.stderr)
            return 2
        data = json.loads(cache.read_text(encoding="utf-8"))

    tiers = [t for t in ("PP-OCRv6_medium_det", "PP-OCRv5_server_det") if t in data]

    def tier_score(payload, page: str):
        page_data = payload["pages"].get(page)
        if not page_data or "error" in page_data:
            return None
        blocks = json.loads(
            (book / "dataset" / page / "selection_blocks.json").read_text(encoding="utf-8")
        )["selection_blocks"]
        blues = [b["image_bbox"] for b in blocks if not b.get("page_scale")]
        tp, fp, fn = score(page_data["items"], blues)
        return tp, fp, fn, page_data.get("seconds", 0.0)

    print("\n=== 每页（蓝框 / v6:框数 FP FN / v5s:框数 FP FN）===")
    per_page = {"page": [], "blue": [], "v6": [], "v5": []}
    for page in pages:
        scores = {}
        for tier in tiers:
            scores[tier] = tier_score(data[tier], page)
        row = []
        for tier in tiers:
            s = scores[tier]
            row.append(f"{len(data[tier]['pages'][page]['items']):>3} "
                       f"{s[1]:>2} {s[2]:>2}" if s else "  --  -- --")
        blue = scores[tiers[0]][0] + scores[tiers[0]][2] if scores.get(tiers[0]) else 0
        print(f"  page {page:>4}: {blue:>3} | " + " | ".join(row))
        per_page["page"].append(page)
        per_page["blue"].append(blue)
        per_page["v6"].append(scores.get(tiers[0]))
        per_page["v5"].append(scores.get(tiers[1]) if len(tiers) > 1 else None)

    print("\n=== 合计（4 列：真阳性 / 假阳性 / 漏检 / 精确率 / 召回率 / 耗时）===")
    for tier in tiers:
        tp = fp = fn = 0
        seconds = 0.0
        for page in pages:
            s = tier_score(data[tier], page)
            if not s:
                continue
            tp += s[0]
            fp += s[1]
            fn += s[2]
            seconds += s[3]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {tier:<22} TP={tp:<4} FP={fp:<4} FN={fn:<3} "
              f"精确率={precision:.1%} 召回率={recall:.1%} 耗时={seconds:.1f}s")

    if args.no_images:
        return 0

    # 三联对比图：蓝框 / v6_medium / v5_server
    font = ImageFont.truetype(FONT_PATH, 26)
    for page in pages:
        panels = []
        check = book / "check" / f"{page}.bluebox_check.png"
        if check.is_file():
            panels.append(("Photos 蓝框（基准）", Image.open(check).convert("RGB"), None))
        original = book / "dataset" / page / "original.jpg"
        colors = [(255, 40, 40), (0, 120, 255)]
        for tier, color in zip(tiers, colors):
            page_data = data[tier]["pages"].get(page, {})
            items = page_data.get("items", [])
            panels.append((f"{tier}（{page_data.get('seconds', '?')}s）",
                           Image.open(original).convert("RGB"), (items, color)))
        width = 760
        columns = []
        for label, image, draw_args in panels:
            scale = width / image.width
            resized = image.resize((width, round(image.height * scale)), Image.LANCZOS)
            if draw_args:
                items, color = draw_args
                draw = ImageDraw.Draw(resized)
                for poly in items:
                    draw.polygon([(x * scale, y * scale) for x, y in poly],
                                 outline=color, width=3)
            strip = Image.new("RGB", (width, 36), (24, 24, 24))
            ImageDraw.Draw(strip).text((8, 4), label, font=font, fill=(240, 240, 240))
            column = Image.new("RGB", (width, 36 + resized.height), (10, 10, 10))
            column.paste(strip, (0, 0))
            column.paste(resized, (0, 36))
            columns.append(column)
        height = max(c.height for c in columns)
        total_w = len(columns) * width + 6 * (len(columns) - 1)
        board = Image.new("RGB", (total_w, height), (10, 10, 10))
        x = 0
        for column in columns:
            board.paste(column, (x, 0))
            x += width + 6
        board.save(out_dir / f"{page}_档位对比.png")
        print(f"compose {page}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
