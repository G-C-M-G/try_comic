"""Island-level manga-ocr recognition (whole island per forward pass).

manga-ocr has no detection: each island bbox is cropped from the masked
image, non-island pixels are whitened (polygon available) and the whole
island is recognized in one call.  The model is loaded once per process;
pass multiple ``--dataset`` page directories to process a batch.  Output
uses the same line-JSON schema as ``island_ocr`` so the correspondence-table
generator works unchanged (one synthetic line per island).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_localizer.ocr_adapter import DEFAULT_UPSTREAM

_SUBPROCESS_SCRIPT = """\
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from manga_ocr import MangaOcr

image_and_islands = json.loads(sys.argv[1])
mocr = MangaOcr()
for entry in image_and_islands:
    image_path = entry["image"]
    islands_json_path = entry["islands"]
    image = Image.open(image_path).convert("RGB")
    data = json.loads(Path(islands_json_path).read_text(encoding="utf-8-sig"))
    lines = []
    for island in data["islands"]:
        x1, y1, x2, y2 = (int(v) for v in island["image_bbox"])
        crop = image.crop((x1, y1, x2, y2))
        polygon = island.get("image_polygon")
        if polygon:
            arr = np.array(crop)
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).polygon(
                [(px - x1, py - y1) for px, py in polygon], fill=255
            )
            white = np.full(arr.shape, 255, dtype=np.uint8)
            arr = np.where(np.array(mask.convert("RGB")) == 0, white, arr)
            crop = Image.fromarray(arr)
        text = str(mocr(crop) or "").strip()
        lines.append(
            {
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "text": text,
                "confidence": None,
            }
        )
    print(json.dumps({"engine": "manga-ocr", "source": image_path, "lines": lines}, ensure_ascii=False))
    sys.stdout.flush()
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Island-level manga-ocr recognition (batch)")
    parser.add_argument("--datasets", nargs="+", required=True, help="page directories")
    parser.add_argument("--image-name", default="filled_black.png",
                        help="masked image name inside each page directory")
    parser.add_argument("--output-dir", required=True, help="output directory for line-JSONs")
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="upstream project root")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    python = Path(args.upstream) / "venv" / "Scripts" / "python.exe"
    if not python.is_file():
        print(f"upstream Python missing: {python}", file=sys.stderr)
        return 2

    entries = []
    for dataset_dir in args.datasets:
        dataset_dir = Path(dataset_dir).resolve()
        image = dataset_dir / args.image_name
        islands_json = dataset_dir / "islands.json"
        if not image.is_file() or not islands_json.is_file():
            print(f"missing files in {dataset_dir}", file=sys.stderr)
            return 2
        entries.append({"image": str(image), "islands": str(islands_json)})

    import os

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    completed = subprocess.run(
        [
            str(python),
            "-c",
            _SUBPROCESS_SCRIPT,
            json.dumps(entries),
        ],
        capture_output=True,
        text=False,
        timeout=args.timeout,
        check=False,
        env=env,
    )
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        detail = (stderr_text or stdout_text).strip()[-800:]
        print(f"manga-ocr failed: {detail}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw_line in stdout_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            print(f"manga-ocr emitted non-JSON line: {raw_line[:120]}", file=sys.stderr)
            continue
        page = Path(payload["source"]).resolve().parent.name
        target = out_dir / f"{page}.manga.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{page}: {len(payload['lines'])} islands -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
