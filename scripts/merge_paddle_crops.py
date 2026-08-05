"""Merge Paddle island-crop variants into one line-JSON per island.

For each island pick the variant with the highest mean recognition confidence
(tie-break: longer text, then raw2x > raw > clahe > otsu) and emit its
predictions in full-image coordinates using the shared line-JSON schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_VARIANT_PREFERENCE = {"raw2x": 0, "raw": 1, "clahe": 2, "otsu": 3}


def _mean_confidence(predictions: list[dict]) -> float:
    scores = [float(item["confidence"]) for item in predictions if item.get("confidence") is not None]
    return sum(scores) / len(scores) if scores else 0.0


def _total_text_length(predictions: list[dict]) -> int:
    return sum(len(str(item.get("text") or "")) for item in predictions)


def merge_crops(crops_json: Path) -> list[dict]:
    payload = json.loads(Path(crops_json).read_text(encoding="utf-8-sig"))
    merged = []
    for crop in payload.get("crops", []):
        variants = crop.get("variants") or {}
        candidates = [
            (variant, list(predictions))
            for variant, predictions in variants.items()
            if predictions and any(str(p.get("text") or "").strip() for p in predictions)
        ]
        if not candidates:
            continue
        best_variant, best_predictions = max(
            candidates,
            key=lambda pair: (
                _mean_confidence(pair[1]),
                _total_text_length(pair[1]),
                -_VARIANT_PREFERENCE.get(pair[0], 9),
            ),
        )
        for prediction in best_predictions:
            merged.append(
                {
                    "polygon": prediction["polygon"],
                    "text": str(prediction.get("text") or ""),
                    "confidence": (
                        float(prediction["confidence"])
                        if prediction.get("confidence") is not None
                        else None
                    ),
                    "island_id": crop["id"],
                    "variant": best_variant,
                }
            )
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge Paddle island-crop variants")
    parser.add_argument("--crops-json", required=True, help="paddle_ocr_worker --crops output")
    parser.add_argument("--output", required=True, help="merged line-JSON output")
    args = parser.parse_args(argv)
    merged = merge_crops(Path(args.crops_json))
    payload = {"engine": "paddle-crops", "lines": merged}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_island: dict[str, int] = {}
    for item in merged:
        by_island[item["island_id"]] = by_island.get(item["island_id"], 0) + 1
    print(f"islands_with_text={len(by_island)} lines={len(merged)}")
    for island_id, count in sorted(by_island.items()):
        print(f"  {island_id}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
