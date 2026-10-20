"""PaddleOCR 日文**识别**（rec-only）worker —— 在 `paddle_venv` 里跑，输出 JSON。

为什么要单独一个 worker：rec 必须在 paddle_venv（PaddleOCR 3.7 / paddle 3.3）里跑，
而评测脚本在 root venv 里；项目已有同样的跨解释器模式（`paddle_ocr_worker.py`）。

**rec-only**：`paddleocr.PaddleOCR.predict` 总是跑 det→rec 整条流水线，而蓝框裁剪上
det 常常一行都找不到（实测返回空）→ 这里用 `paddlex.create_model(<rec 模型>)` 直接
调识别头。蓝框本来就是"文本行级"的框，所以这一步是"正确用法"。

**两种朝向**：PaddleOCR 的日文 rec 是**横排**训练的（`japan_PP-OCRv3_mobile_rec`，
且比它自己的中英模型落后两代），竖排必须转 90° 才读得动 → 这里两种都跑、把 rec 分数
一起给出来，让评测脚本能分开看"原样 / 转 90°"。

用法（paddle_venv）：
    paddle_venv/Scripts/python.exe paddle_rec_worker.py --crops crops.json --out out.json
crops.json: ``[{"id": "...", "path": "..."}]``
out.json:   ``{"<id>": {"h": {"text":..., "score":...}, "v": {...}}}``
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PADDLE_CACHE = ROOT / "models" / "paddle"
os.environ.setdefault("HOME", str(PADDLE_CACHE))
os.environ.setdefault("USERPROFILE", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")


def read_one(model, image: np.ndarray) -> tuple[str, float]:
    try:
        for result in model.predict(image):
            payload = result if isinstance(result, dict) else getattr(result, "json", {})
            payload = payload.get("res", payload) if isinstance(payload, dict) else {}
            text = str(payload.get("rec_text") or "").strip()
            score = float(payload.get("rec_score") or 0.0)
            return text, score
    except Exception as error:                      # noqa: BLE001
        return f"<{type(error).__name__}>", 0.0
    return "", 0.0


def resolve_device(device: str) -> str:
    """``auto`` = 有 CUDA 版 paddle 且检测到 GPU 就用 GPU（与 `paddle_ocr_worker`
    同一约定）；显式 ``gpu`` 不回退（失败应当可见）。本机 paddle_venv 是 CUDA 版，
    所以默认就走 GPU——第一版这里默认写死 cpu，等于把 GPU 白关掉。
    """
    if device != "auto":
        return device
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:                       # noqa: BLE001 - 探测失败一律回退 CPU
        pass
    return "cpu"


def load_variants(path: str, pad: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """一张裁剪 → (原样, 转 90°)，四边补白边（rec 模型预期上下留白）。"""
    with Image.open(path).convert("RGB") as handle:
        frame = handle
        if pad:
            padded = Image.new("RGB", (frame.width + 2 * pad, frame.height + 2 * pad),
                               "white")
            padded.paste(frame, (pad, pad))
            frame = padded
        horizontal = np.asarray(frame)
        vertical = np.asarray(frame.rotate(-90, expand=True))     # 竖排 → 横排
        size = [frame.width, frame.height]
    return horizontal, vertical, size


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="japan_PP-OCRv3_mobile_rec")
    parser.add_argument("--device", default="auto", help="auto（默认，能用 GPU 就 GPU）/ cpu / gpu")
    parser.add_argument("--pad", type=int, default=8,
                        help="识别前四边补白边（rec 模型预期上下留白）")
    parser.add_argument("--batch", type=int, default=32,
                        help="一次喂多少张图。实测批量远比逐张快：129 框逐张 "
                             "17s(GPU)/7s(CPU) → 批量 0.7s(GPU)/1.0s(CPU)")
    args = parser.parse_args(argv)

    args.device = resolve_device(args.device)
    from paddlex import create_model

    started = time.time()
    model = create_model(args.model, device=args.device)
    crops = json.loads(Path(args.crops).read_text(encoding="utf-8"))
    print(f"device={args.device} batch={args.batch} crops={len(crops)}"
          f"（加载 {time.time() - started:.1f}s）", file=sys.stderr)

    frames: list[np.ndarray] = []
    sizes: dict[str, list[int]] = {}
    for crop in crops:
        horizontal, vertical, size = load_variants(crop["path"], args.pad)
        frames.append(horizontal)
        frames.append(vertical)
        sizes[crop["id"]] = size

    started = time.time()
    texts: list[tuple[str, float]] = []
    for result in model.predict(frames, batch_size=args.batch):
        payload = result if isinstance(result, dict) else getattr(result, "json", {})
        payload = payload.get("res", payload) if isinstance(payload, dict) else {}
        texts.append((str(payload.get("rec_text") or "").strip(),
                      float(payload.get("rec_score") or 0.0)))
    print(f"识别 {len(frames)} 张耗时 {time.time() - started:.1f}s", file=sys.stderr)

    out: dict[str, dict] = {}
    for index, crop in enumerate(crops):
        text_h, score_h = texts[2 * index]
        text_v, score_v = texts[2 * index + 1]
        out[crop["id"]] = {"h": {"text": text_h, "score": score_h},
                           "v": {"text": text_v, "score": score_v},
                           "size": sizes[crop["id"]]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} ({len(out)} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
