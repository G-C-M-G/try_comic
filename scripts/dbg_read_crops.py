"""Debug: manga-ocr readings of crop variants for one island."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path(r"D:\workplace\opencode\try_comic")

SCRIPT = (
    "import json, sys\n"
    "sys.stdout.reconfigure(encoding='utf-8')\n"
    "from PIL import Image\n"
    "from manga_ocr import MangaOcr\n"
    "mocr = MangaOcr()\n"
    "for path in sys.argv[1:]:\n"
    "    print(json.dumps({'path': path, 'text': str(mocr(Image.open(path)) or '').strip()}, ensure_ascii=False))\n"
)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        print("usage: dbg_read_crops.py <dir> <glob>")
        return 2
    import os

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    crops = [str(path) for path in Path(argv[0]).glob(argv[1])]
    completed = subprocess.run(
        [str(UPSTREAM / "venv" / "Scripts" / "python.exe"), "-c", SCRIPT, *crops],
        capture_output=True,
        text=False,
        timeout=900,
        env=env,
    )
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                payload = json.loads(line)
                print(f"{Path(payload['path']).name}: {payload['text']}")
            except json.JSONDecodeError:
                print(line[:200])
    if completed.returncode != 0:
        print("stderr:", completed.stderr.decode("utf-8", errors="replace")[-500:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
