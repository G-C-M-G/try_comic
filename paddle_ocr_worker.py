"""Isolated PP-OCRv6 worker that emits a small, stable JSON schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

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


def _parse_pages(value: str | None) -> set[str] | None:
    if not value:
        return None
    # Tolerate full-width punctuation ("4，6" / "4、6" from an IME).
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


def _inputs(path: Path, pages: set[str] | None) -> Iterable[Path]:
    if path.is_file():
        if pages is None or path.stem in pages:
            yield path
        return
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            if pages is None or child.stem in pages:
                yield child


def _resolve_device(device: str) -> str:
    """``auto`` = 有 CUDA 版 paddle 且检测到 GPU 就用 GPU，否则 CPU。

    显式传 ``gpu`` 时不做回退（用户明确要 GPU，失败应当可见）；``auto``
    与 ``cpu`` 永远安全。GPU 实测比 CPU 快 ~7.6 倍（32 页 6.9s vs 52.3s）。
    """
    if device != "auto":
        return device
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:  # noqa: BLE001 - 探测失败一律回退 CPU
        pass
    return "cpu"


def _build_engine(device: str) -> PaddleOCR:
    device = _resolve_device(device)
    det_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_det"
    rec_dir = PADDLE_CACHE / "official_models" / "PP-OCRv6_medium_rec"
    model_paths = {}
    if (det_dir / "inference.json").is_file() and (rec_dir / "inference.json").is_file():
        model_paths = {
            "text_detection_model_dir": str(det_dir),
            "text_recognition_model_dir": str(rec_dir),
        }
    return PaddleOCR(
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


def _predict_frame(engine: PaddleOCR, frame) -> list[dict]:
    """Run detection + recognition on one in-memory frame (numpy/PIL)."""
    predictions = []
    for result in engine.predict(frame):
        payload = result.json["res"]
        polygons = _plain(payload.get("rec_polys", []))
        texts = list(payload.get("rec_texts", []))
        scores = _plain(payload.get("rec_scores", []))
        for polygon, text, score in zip(polygons, texts, scores):
            predictions.append({
                "polygon": [[int(round(x)), int(round(y))] for x, y in polygon],
                "text": str(text),
                "confidence": float(score),
            })
    return predictions


def _variant_frame(region: np.ndarray, variant: str) -> np.ndarray:
    import cv2

    rgb = np.ascontiguousarray(region)
    if variant == "raw":
        return rgb
    if variant == "raw2x":
        return cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if variant == "clahe":
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if variant == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unknown variant {variant!r}")


def _predict_page(engine: PaddleOCR, source_path: Path) -> tuple[str, list[dict]]:
    """Predict one whole image file; returns (resolved_source, predictions)."""
    predictions = []
    resolved = str(source_path.resolve())
    for result in engine.predict(str(source_path)):
        payload = result.json["res"]
        polygons = _plain(payload.get("rec_polys", []))
        texts = list(payload.get("rec_texts", []))
        scores = _plain(payload.get("rec_scores", []))
        for polygon, text, score in zip(polygons, texts, scores):
            predictions.append({
                "polygon": [[int(round(x)), int(round(y))] for x, y in polygon],
                "text": str(text),
                "confidence": float(score),
            })
        resolved = str(Path(payload.get("input_path") or resolved).resolve())
    return resolved, predictions


def run(
    input_path: str,
    output_path: str,
    device: str = "auto",
    selected_pages: set[str] | None = None,
) -> dict:
    engine = _build_engine(device)
    result_pages = []
    for source_path in _inputs(Path(input_path).resolve(), selected_pages):
        resolved, predictions = _predict_page(engine, source_path)
        result_pages.append({"source": resolved, "predictions": predictions})
    output = {
        "schema_version": 1,
        "backend": "PaddleOCR PP-OCRv6 medium",
        "device": _resolve_device(device),
        "pages": result_pages,
    }
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _predict_crops(engine: PaddleOCR, manifest: dict) -> dict:
    source = str(Path(manifest["source"]).resolve())
    from PIL import Image

    image = np.array(Image.open(source).convert("RGB"))
    crops = []
    for crop in manifest.get("crops") or []:
        x1, y1, x2, y2 = (int(value) for value in crop["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        region = image[y1:y2, x1:x2]
        variants = {}
        for variant in crop.get("variants") or ["raw", "raw2x", "clahe", "otsu"]:
            frame = _variant_frame(region, variant)
            predictions = _predict_frame(engine, frame)
            scale = 2.0 if variant == "raw2x" else 1.0
            for prediction in predictions:
                prediction["polygon"] = [
                    [int(round(px / scale + x1)), int(round(py / scale + y1))]
                    for px, py in prediction["polygon"]
                ]
            variants[variant] = predictions
        crops.append({"id": crop["id"], "bbox": [x1, y1, x2, y2], "variants": variants})
    return {
        "schema_version": 1,
        "backend": "PaddleOCR PP-OCRv6 medium (crops)",
        "device": _resolve_device("auto"),
        "source": source,
        "crops": crops,
    }


def run_crops(manifest_path: str, output_path: str, device: str = "auto") -> dict:
    """OCR each requested crop box with preprocessing variants.

    The input manifest is JSON:

        {"source": "image.jpg", "crops": [
            {"id": "island_001", "bbox": [x1, y1, x2, y2],
             "variants": ["raw", "raw2x", "clahe", "otsu"]}, ...
        ]}

    Polygons in the output are expressed in full-image coordinates.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    engine = _build_engine(device)
    output = _predict_crops(engine, manifest)
    output["device"] = _resolve_device(device)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def serve(device: str = "auto") -> int:
    """Persistent worker: read JSON requests from stdin, write results to stdout.

    Each line is one request::

        {"id": "...", "kind": "page", "input": "masked.png"}
        {"id": "...", "kind": "crops", "manifest": {...}}

    Responses are single JSON lines: ``{"id": ..., "ok": true, ...}`` or
    ``{"id": ..., "ok": false, "error": "..."}``.  The engine is built once.
    """
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    engine = _build_engine(device)
    line_number = 0
    for line in sys.stdin:
        line_number += 1
        request_id = f"line_{line_number}"
        try:
            request = json.loads(line)
            request_id = str(request.get("id") or request_id)
            kind = str(request.get("kind") or "")
            if kind == "page":
                source = str(Path(request["input"]).resolve())
                resolved, predictions = _predict_page(engine, Path(source))
                payload = {"id": request_id, "ok": True, "source": resolved, "predictions": predictions}
            elif kind == "crops":
                crops = _predict_crops(engine, request["manifest"])
                payload = {"id": request_id, "ok": True, "crops": crops.get("crops", [])}
            else:
                raise ValueError(f"unknown request kind {kind!r}")
        except Exception as error:
            payload = {"id": request_id, "ok": False, "error": str(error)}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated PaddleOCR worker")
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--device", default="auto",
                        help="auto（有 GPU 就用）/ cpu / gpu")
    parser.add_argument("--pages")
    parser.add_argument(
        "--crops",
        action="store_true",
        help="treat input as a crops manifest instead of an image path",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="persistent mode: read JSON requests from stdin, one per line",
    )
    args = parser.parse_args(argv)
    if args.serve:
        return serve(args.device)
    if args.crops:
        run_crops(args.input, args.output, args.device)
    else:
        run(args.input, args.output, args.device, _parse_pages(args.pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
