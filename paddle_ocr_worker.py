"""Isolated PP-OCRv6 worker that emits a small, stable JSON schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
PADDLE_CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(PADDLE_CACHE))
os.environ.setdefault("USERPROFILE", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

from paddleocr import PaddleOCR


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run(input_path: str, output_path: str, device: str = "cpu") -> dict:
    det_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_det"
    rec_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_rec"
    model_paths = {}
    if (det_dir / "inference.json").is_file() and (rec_dir / "inference.json").is_file():
        model_paths = {
            "text_detection_model_dir": str(det_dir),
            "text_recognition_model_dir": str(rec_dir),
        }
    engine = PaddleOCR(
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
        text_rec_score_thresh=0.0,
        device=device,
        enable_mkldnn=False,
        **model_paths,
    )
    pages = []
    for result in engine.predict(str(Path(input_path).resolve())):
        payload = result.json["res"]
        polygons = _plain(payload.get("rec_polys", []))
        texts = list(payload.get("rec_texts", []))
        scores = _plain(payload.get("rec_scores", []))
        predictions = []
        for polygon, text, score in zip(polygons, texts, scores):
            predictions.append({
                "polygon": [[int(round(x)), int(round(y))] for x, y in polygon],
                "text": str(text),
                "confidence": float(score),
            })
        pages.append({
            "source": str(Path(payload.get("input_path") or input_path).resolve()),
            "predictions": predictions,
        })
    output = {
        "schema_version": 1,
        "backend": "PaddleOCR PP-OCRv6 medium",
        "device": device,
        "pages": pages,
    }
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated PaddleOCR worker")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run(args.input, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
