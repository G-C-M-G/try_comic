"""PaddleOCR 识别在竖排/横排蓝框上的准确率对比。

阶段区分：检测（找框的位置）与识别（读框里的字）是两个独立模型。
本脚本用 PP-OCRv6_medium_rec（你现用的识别模型）读同一个蓝框，
按框方向（蓝框宽高比）分组，与 manga-ocr 的读数（ocr_readings.json）
算字符相似度，回答"Paddle 是不是只擅长竖排"。

用法（root venv）：
    venv/Scripts/python.exe paddle_orientation_read.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PADDLE_PYTHON = ROOT / "paddle_venv" / "Scripts" / "python.exe"
PADDLE_CACHE = ROOT / "models" / "paddle"
PAD = 6  # 与 stage2 一致

SCRIPT = r"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[3])
CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(CACHE))
os.environ.setdefault("USERPROFILE", str(CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "gpu"
REC = sys.argv[5] if len(sys.argv) > 5 else "PP-OCRv6_medium_rec"

from paddleocr import TextRecognition

engine = TextRecognition(model_name=REC, device=DEVICE)
readings = {}
for i, (key, image_path) in enumerate(jobs.items(), 1):
    try:
        # TextRecognition 的输出字段是 rec_text（单个字符串），
        # 与检测器的 rec_texts（列表）不同
        text = ""
        score = None
        for r in engine.predict(image_path):
            payload = r.json["res"]
            text = str(payload.get("rec_text") or "")
            score = payload.get("rec_score")
            break
        readings[key] = {"text": text, "score": score}
        print(f"{i}/{len(jobs)} {key}: {text!r} ({score})", flush=True)
    except Exception as error:  # noqa: BLE001
        readings[key] = {"error": repr(error)}
        print(f"{i}/{len(jobs)} {key}: ERROR {error!r}", flush=True)
out_path.write_text(json.dumps(readings, ensure_ascii=False), encoding="utf-8")
"""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in " 「」『』()（）…．.、，。\n\r\t":
        text = text.replace(ch, "")
    return text.lower()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="sample-book")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--rec", default="PP-OCRv6_medium_rec")
    parser.add_argument("--crops-per-page", type=int, default=0,
                        help="每页最多取几个蓝框（0=全部）")
    args = parser.parse_args(argv)

    book = ROOT / "outputs" / "oneclick" / args.book
    readings = json.loads((book / "ocr_readings.json").read_text(encoding="utf-8"))
    work = ROOT / "var" / "orientation_test"
    work.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    jobs: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for page_dir in sorted(
        (book / "dataset").iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else 0,
    ):
        page = page_dir.name
        blocks_path = page_dir / "selection_blocks.json"
        masked_path = page_dir / "bluebox_masked.png"
        if not (blocks_path.is_file() and masked_path.is_file()):
            continue
        blocks = json.loads(blocks_path.read_text(encoding="utf-8"))["selection_blocks"]
        image = Image.open(masked_path).convert("RGB")
        width, height = image.size
        count = 0
        for block in blocks:
            if block.get("page_scale"):
                continue
            key = f"{page}:box:{block['id']}"
            reference = readings.get(key)
            if not reference:
                continue
            if args.crops_per_page and count >= args.crops_per_page:
                break
            x1, y1, x2, y2 = block["image_bbox"]
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue
            orientation = "horizontal" if box_w >= box_h else "vertical"
            crop = image.crop((
                max(0, x1 - PAD), max(0, y1 - PAD),
                min(width, x2 + PAD), min(height, y2 + PAD),
            ))
            crop_path = work / f"{page}_{block['id']}.png"
            crop.save(crop_path)
            jobs[key] = str(crop_path)
            # 竖排的另一种喂法：旋转 90° 后按横排识别（PP-OCR 系的常规做法）
            if orientation == "vertical":
                rotated = crop.rotate(-90, expand=True)
                rotated_path = work / f"{page}_{block['id']}_rot.png"
                rotated.save(rotated_path)
                jobs[f"{key}#rot"] = str(rotated_path)
            meta[key] = {
                "orientation": orientation,
                "reference": reference,
                "aspect": round(box_w / box_h, 2),
            }
            count += 1

    if not jobs:
        print("no crops found")
        return 2
    print(f"crops: {len(jobs)} -> {PADDLE_PYTHON} ({args.device}, {args.rec})")

    cache = ROOT / "var" / "ocr_compare" / f"orientation_rec_{args.device}.json"
    payload = cache.with_name("orientation_jobs.json")
    payload.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [str(PADDLE_PYTHON), "-c", SCRIPT, str(payload), str(cache),
         str(ROOT), args.device, args.rec],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=3600,
    )
    if not cache.is_file():
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        return 2
    predicted = json.loads(cache.read_text(encoding="utf-8"))

    groups: dict[str, list] = {"vertical": [], "horizontal": [], "vertical_rot": []}
    scores_by_group: dict[str, list] = {"vertical": [], "horizontal": [], "vertical_rot": []}
    for key, info in meta.items():
        # 本方向原样
        for variant, group in (("", info["orientation"]), ("#rot", "vertical_rot")):
            payload = predicted.get(key + variant)
            if not payload or "error" in payload:
                continue
            prediction = payload.get("text", "")
            a, b = normalize(info["reference"]), normalize(prediction)
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            groups[group].append((key + variant, info["reference"], prediction, ratio))
            if payload.get("score") is not None:
                scores_by_group[group].append(float(payload["score"]))

    print("\n=== 识别准确率（与 manga-ocr 读数比对，归一化相似度；分数=模型自报置信度）===")
    for name, label in (("vertical", "竖排（原样）"),
                        ("vertical_rot", "竖排（旋转90°）"),
                        ("horizontal", "横排（原样）")):
        rows = groups[name]
        if not rows:
            print(f"  {label:<14}: 无样本")
            continue
        mean = sum(r[3] for r in rows) / len(rows)
        exact = sum(1 for r in rows if r[3] == 1.0)
        near = sum(1 for r in rows if r[3] >= 0.8)
        score_mean = (sum(scores_by_group[name]) / len(scores_by_group[name])
                      if scores_by_group[name] else 0.0)
        print(f"  {label:<14}: {len(rows):>3} 框 | 平均相似度 {mean:.3f} | "
              f"完全一致 {exact:>3} ({exact / len(rows):>3.0%}) | "
              f"≥0.8 {near:>3} ({near / len(rows):>3.0%}) | 平均置信度 {score_mean:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
