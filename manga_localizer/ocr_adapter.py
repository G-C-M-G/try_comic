from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple


DEFAULT_UPSTREAM = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OcrCorrectionRequest:
    images: Path
    pages: str
    capture_dir: Path
    ocr_dir: Path
    upstream: Path = DEFAULT_UPSTREAM
    timeout: float = 20.0
    resume: bool = True
    dialogue_only: bool = True
    ocr_device: str = "cuda"
    paddle_device: str = "cpu"
    cross_check: bool = False

    @property
    def python(self) -> Path:
        return self.upstream / "venv" / "Scripts" / "python.exe"


def _validated(request: OcrCorrectionRequest) -> OcrCorrectionRequest:
    if not request.images.is_dir():
        raise ValueError(f"image directory does not exist: {request.images}")
    if not request.upstream.is_dir():
        raise ValueError(f"upstream OCR project does not exist: {request.upstream}")
    if not request.python.is_file():
        raise ValueError(f"upstream OCR Python does not exist: {request.python}")
    if not request.pages.strip():
        raise ValueError("OCR correction pages are required")
    return request


def build_photos_capture_command(request: OcrCorrectionRequest) -> Tuple[str, ...]:
    request = _validated(request)
    command = [
        str(request.python),
        str(request.upstream / "photos_capture.py"),
        "batch",
        "--images",
        str(request.images.resolve()),
        "--pages",
        request.pages,
        "--output-dir",
        str(request.capture_dir.resolve()),
        "--quick",
        "--timeout",
        str(request.timeout),
    ]
    if request.resume:
        command.append("--resume")
    return tuple(command)


def build_fast_capture_command(request: OcrCorrectionRequest) -> Tuple[str, ...]:
    """Fast-mask capture: Photos islands + masked-page PaddleOCR per page."""
    request = _validated(request)
    command = [
        str(request.python),
        str(request.upstream / "photos_capture.py"),
        "batch",
        "--fast",
        "--images",
        str(request.images.resolve()),
        "--pages",
        request.pages,
        "--output-dir",
        str(request.capture_dir.resolve()),
        "--timeout",
        str(request.timeout),
        "--paddle-device",
        request.paddle_device,
    ]
    if request.resume:
        command.append("--resume")
    return tuple(command)


def build_fast_fusion_command(request: OcrCorrectionRequest) -> Tuple[str, ...]:
    """Background fusion from fast-mask capture manifests (schema 4 output)."""
    request = _validated(request)
    command = [
        str(request.python),
        str(request.upstream / "background_ocr.py"),
        str(request.images.resolve()),
        str(request.ocr_dir.resolve()),
        "--pages",
        request.pages,
        "--photos-fast-dir",
        str(request.capture_dir.resolve()),
        "--device",
        request.ocr_device,
        "--paddle-device",
        request.paddle_device,
    ]
    if request.dialogue_only:
        command.append("--dialogue-only")
    if request.cross_check:
        command.append("--cross-check")
    return tuple(command)


def build_recapture_command(
    request: OcrCorrectionRequest,
    fusion_dir: Path | None = None,
) -> Tuple[str, ...]:
    """Targeted Photos click-copy for conflict columns only."""
    request = _validated(request)
    command = [
        str(request.python),
        str(request.upstream / "photos_capture.py"),
        "recapture",
        "--capture-dir",
        str(request.capture_dir.resolve()),
        "--pages",
        request.pages,
        "--timeout",
        str(request.timeout),
    ]
    if fusion_dir is not None:
        command.extend(("--fusion-dir", str(fusion_dir.resolve())))
    return tuple(command)


def build_background_ocr_command(request: OcrCorrectionRequest) -> Tuple[str, ...]:
    request = _validated(request)
    command = [
        str(request.python),
        str(request.upstream / "background_ocr.py"),
        str(request.images.resolve()),
        str(request.ocr_dir.resolve()),
        "--pages",
        request.pages,
        "--photos-capture-dir",
        str(request.capture_dir.resolve()),
        "--device",
        request.ocr_device,
        "--paddle-device",
        request.paddle_device,
    ]
    if request.dialogue_only:
        command.append("--dialogue-only")
    return tuple(command)


def build_prepare_review_command(
    ocr_path: Path,
    *,
    upstream: Path = DEFAULT_UPSTREAM,
    review_path: Optional[Path] = None,
) -> Tuple[str, ...]:
    python = upstream / "venv" / "Scripts" / "python.exe"
    command = [
        str(python),
        str(upstream / "ocr_review.py"),
        "prepare",
        str(ocr_path.resolve()),
        "--schema-version",
        "2",
    ]
    if review_path is not None:
        command.extend(("--review", str(review_path.resolve())))
    return tuple(command)


def build_apply_review_command(
    ocr_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    upstream: Path = DEFAULT_UPSTREAM,
) -> Tuple[str, ...]:
    python = upstream / "venv" / "Scripts" / "python.exe"
    return (
        str(python),
        str(upstream / "ocr_review.py"),
        "apply",
        str(ocr_path.resolve()),
        str(review_path.resolve()),
        str(output_path.resolve()),
    )


def correction_status(
    pages: Sequence[str],
    capture_dir: Path,
    ocr_dir: Path,
) -> list[dict]:
    rows = []
    for page in pages:
        capture = capture_dir / page / "capture.json"
        ocr = ocr_dir / f"{page}.ocr.json"
        capture_status = "missing"
        if capture.is_file():
            try:
                payload = json.loads(capture.read_text(encoding="utf-8-sig"))
                capture_status = "complete" if payload.get("capture", {}).get("complete") else "failed"
            except (OSError, UnicodeError, json.JSONDecodeError):
                capture_status = "invalid"
        review_status = "missing"
        if ocr.is_file():
            try:
                payload = json.loads(ocr.read_text(encoding="utf-8-sig"))
                review_status = str(payload.get("manual_review", {}).get("status", "unknown"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                review_status = "invalid"
        rows.append({
            "page": page,
            "capture": capture_status,
            "ocr": "ready" if ocr.is_file() else "missing",
            "review": review_status,
        })
    return rows
