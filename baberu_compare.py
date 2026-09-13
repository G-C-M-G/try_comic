"""Baberu OCR 与 manga-ocr 的同输入读字对比。

对指定页的每个蓝框，用与 stage2 完全相同的裁剪（bluebox_masked.png 上
image_bbox 扩 6px，即 manga-ocr 吃到的输入）喂 Baberu ONNX，与
ocr_readings.json 里的 manga-ocr 读数逐框对比。

用法（root venv）：
    venv/Scripts/python.exe baberu_compare.py --pages 19,33
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models" / "baberu-ocr"))

from onnx_infer import BaberuOnnxOCR  # noqa: E402

PAD = 6  # 与 stage2 一致：蓝框扩 6px 从填黑图裁剪


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in " 「」『』()（）…．.、，。\n\r\t":
        text = text.replace(ch, "")
    return text.lower()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default="19,33")
    parser.add_argument("--book", default="sample-book")
    parser.add_argument("--vision", default="vision_fp16.onnx")
    args = parser.parse_args(argv)

    book = ROOT / "outputs" / "oneclick" / args.book
    readings = json.loads((book / "ocr_readings.json").read_text(encoding="utf-8"))
    ocr = BaberuOnnxOCR(
        ROOT / "models" / "baberu-ocr" / "onnx",
        ROOT / "models" / "baberu-ocr" / "tokenizer",
        vision=args.vision,
    )

    from PIL import Image

    rows = []
    for page in [p.strip() for p in args.pages.split(",") if p.strip()]:
        blocks = json.loads(
            (book / "dataset" / page / "selection_blocks.json").read_text(encoding="utf-8")
        )["selection_blocks"]
        masked = Image.open(book / "dataset" / page / "bluebox_masked.png").convert("RGB")
        width, height = masked.size
        for block in blocks:
            if block.get("page_scale"):
                continue
            box_id = block["id"]
            x1, y1, x2, y2 = block["image_bbox"]
            crop_box = (
                max(0, x1 - PAD), max(0, y1 - PAD),
                min(width, x2 + PAD), min(height, y2 + PAD),
            )
            reference = readings.get(f"{page}:box:{box_id}", "")
            text = ocr(masked.crop(crop_box))
            same_raw = text.strip() == reference.strip()
            norm_a, norm_b = normalize(text), normalize(reference)
            same_norm = norm_a == norm_b
            ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
            rows.append((page, box_id, reference, text, same_raw, same_norm, ratio))
            flag = "OK " if same_norm else "DIFF"
            print(f"[{flag}] {page}:{box_id}\n  manga-ocr: {reference}\n  baberu   : {text}",
                  flush=True)

    exact = sum(r[4] for r in rows)
    norm_eq = sum(r[5] for r in rows)
    mean_ratio = sum(r[6] for r in rows) / max(1, len(rows))
    print(f"\n=== {len(rows)} 框 ===")
    print(f"原始完全一致: {exact}/{len(rows)}")
    print(f"归一化后一致: {norm_eq}/{len(rows)}")
    print(f"平均相似度  : {mean_ratio:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
