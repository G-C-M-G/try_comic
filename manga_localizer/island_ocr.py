"""Island OCR: headless Windows.Media.Ocr(ja) via the upstream venv.

Windows.Media.Ocr runs through the upstream project's ``windows_ocr.py`` so
this repository needs no winrt dependency and no model download.  The call is
a one-shot subprocess: it never opens a window and never controls the mouse.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .errors import TranslationError
from .ocr_adapter import DEFAULT_UPSTREAM
from .island_dry_run import OcrLine


_SUBPROCESS_SCRIPT = """\
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[2])
from windows_ocr import windows_ocr_predictions
result = windows_ocr_predictions(sys.argv[1])
if result is None:
    print(json.dumps({"engine": None, "lines": []}))
else:
    print(json.dumps({"engine": "windows-ocr", "lines": result}, ensure_ascii=False))
"""


def windows_ocr_lines(
    image_path: Path,
    *,
    upstream: Path = DEFAULT_UPSTREAM,
    timeout: float = 60.0,
) -> Tuple[Optional[str], Sequence[OcrLine]]:
    """Recognize *image_path* headlessly; returns ``(engine, lines)``.

    ``engine`` is ``None`` when the ja recognizer is unavailable; ``lines``
    is empty when nothing was found.
    """
    python = upstream / "venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise TranslationError(f"upstream OCR Python does not exist: {python}")
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                _SUBPROCESS_SCRIPT,
                str(Path(image_path).resolve()),
                str(Path(upstream).resolve()),
            ],
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TranslationError(f"Windows OCR timed out after {timeout}s") from error
    except OSError as error:
        raise TranslationError(f"cannot run Windows OCR: {error}") from error
    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        detail = (stderr_text or stdout_text).strip()[:400]
        raise TranslationError(f"Windows OCR failed: {detail}")
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as error:
        raise TranslationError(f"Windows OCR returned invalid JSON: {error}") from error
    engine = payload.get("engine")
    lines = []
    for item in payload.get("lines", []):
        polygon = tuple((float(point[0]), float(point[1])) for point in item.get("polygon", []))
        lines.append(OcrLine(polygon=polygon, text=str(item.get("text") or "")))
    return engine, tuple(lines)
