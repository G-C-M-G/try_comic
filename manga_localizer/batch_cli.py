from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import sys
from typing import Optional, Sequence

from .cli import main as render_main
from .ocr_contract import SUPPORTED_IMAGE_SUFFIXES, sha256_file


def parse_pages(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
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


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.stem)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless sequential batch manga localization."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", help="Comma-separated image stems or numeric ranges")
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument(
        "--translator",
        choices=("sakura", "openai-compatible", "mock"),
        default="sakura",
    )
    parser.add_argument("--mock-translations-dir", type=Path)
    parser.add_argument("--glossary", type=Path, default=Path("config/glossary.json"))
    parser.add_argument(
        "--translation-base-url", default="http://127.0.0.1:8080/v1"
    )
    parser.add_argument("--translation-model", default="sukinishiro")
    parser.add_argument("--translation-timeout", type=float, default=90.0)
    parser.add_argument("--translation-attempts", type=int, default=2)
    parser.add_argument(
        "--translation-prompt-style",
        choices=("auto", "contextual", "concise"),
        default="auto",
    )
    parser.add_argument("--translation-cache-dir", type=Path)
    parser.add_argument("--no-translation-cache", action="store_true")
    parser.add_argument("--no-manual-review", action="store_true")
    parser.add_argument("--max-layout-overflow", type=int, choices=(10, 20), default=10)
    parser.add_argument("--min-font-size", type=int, default=10)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--terminal-punctuation-font-ratio", type=float, default=0.68)
    parser.add_argument(
        "--preserve-unit-center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Batch report JSON; defaults to <output-dir>/batch.report.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    images_dir = args.images.resolve()
    ocr_dir = args.ocr_dir.resolve()
    output_dir = args.output_dir.resolve()
    review_dir = (args.review_dir or args.ocr_dir).resolve()
    if not images_dir.is_dir():
        parser.error(f"--images is not a directory: {images_dir}")
    if not ocr_dir.is_dir():
        parser.error(f"--ocr-dir is not a directory: {ocr_dir}")
    if args.translator == "mock" and args.mock_translations_dir is None:
        parser.error("--translator mock requires --mock-translations-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (args.report or output_dir / "batch.report.json").resolve()
    cache_dir = (
        args.translation_cache_dir.resolve()
        if args.translation_cache_dir is not None
        else output_dir / ".translation-cache"
    )
    selected_pages = parse_pages(args.pages)

    images = sorted(
        (
            item
            for item in images_dir.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and (selected_pages is None or item.stem in selected_pages)
        ),
        key=_natural_key,
    )
    seen_stems = set()
    entries = []
    for image in images:
        stem_key = image.stem.casefold()
        if stem_key in seen_stems:
            entries.append(
                _missing_entry(
                    image,
                    "duplicate image stem with a different extension",
                )
            )
            continue
        seen_stems.add(stem_key)
        ocr = ocr_dir / f"{image.stem}.ocr.json"
        review = review_dir / f"{image.stem}.ocr.review.json"
        if not ocr.is_file():
            entries.append(_missing_entry(image, f"missing OCR JSON: {ocr}"))
            continue
        page_output = output_dir / f"{image.stem}.translated.png"
        state_output = output_dir / f"{image.stem}.state.json"
        stage_dir = output_dir / f"{image.stem}.stages"
        render_args = [
            "--image",
            str(image),
            "--ocr",
            str(ocr),
            "--output",
            str(page_output),
            "--state-output",
            str(state_output),
            "--stage-dir",
            str(stage_dir),
            "--translator",
            args.translator,
            "--glossary",
            str(args.glossary),
            "--translation-base-url",
            args.translation_base_url,
            "--translation-model",
            args.translation_model,
            "--translation-timeout",
            str(args.translation_timeout),
            "--translation-attempts",
            str(args.translation_attempts),
            "--translation-prompt-style",
            args.translation_prompt_style,
            "--translation-cache-dir",
            str(cache_dir),
            "--max-layout-overflow",
            str(args.max_layout_overflow),
            "--min-font-size",
            str(args.min_font_size),
            "--outline-width",
            str(args.outline_width),
            "--terminal-punctuation-font-ratio",
            str(args.terminal_punctuation_font_ratio),
        ]
        if review.is_file():
            render_args.extend(("--review", str(review)))
        if args.schema is not None:
            render_args.extend(("--schema", str(args.schema)))
        if args.no_manual_review:
            render_args.append("--no-manual-review")
        if args.no_translation_cache:
            render_args.append("--no-translation-cache")
        if not args.preserve_unit_center:
            render_args.append("--no-preserve-unit-center")
        if args.translator == "mock":
            mapping = args.mock_translations_dir.resolve() / f"{image.stem}.json"
            if not mapping.is_file():
                entries.append(
                    _missing_entry(image, f"missing mock translations: {mapping}")
                )
                continue
            render_args.extend(("--mock-translations", str(mapping)))

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = render_main(render_args)
        entry = {
            "image": str(image),
            "image_sha256": sha256_file(image),
            "ocr": str(ocr),
            "ocr_sha256": sha256_file(ocr),
            "output": str(page_output),
            "state": str(state_output),
            "stage_directory": str(stage_dir),
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
            "stdout": stdout.getvalue().strip().splitlines(),
            "stderr": stderr.getvalue().strip().splitlines(),
        }
        if state_output.is_file():
            state_data = json.loads(state_output.read_text(encoding="utf-8"))
            entry["pipeline_status"] = state_data.get("status")
            entry["translation_revision_sha256"] = state_data.get(
                "translation_revision_sha256"
            )
        entries.append(entry)
        print(f"{image.name}: {entry['status']} (exit {exit_code})")

    payload = {
        "artifact": "manga-localizer-batch-report",
        "artifact_version": 1,
        "sequential": True,
        "headless": True,
        "images_directory": str(images_dir),
        "ocr_directory": str(ocr_dir),
        "output_directory": str(output_dir),
        "selected_pages": sorted(
            selected_pages,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value.casefold()),
        )
        if selected_pages
        else None,
        "summary": {
            "images": len(images),
            "completed": sum(item["status"] == "completed" for item in entries),
            "failed": sum(item["status"] != "completed" for item in entries),
        },
        "pages": entries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if payload["summary"]["failed"] == 0 else 2


def _missing_entry(image: Path, reason: str):
    print(f"{image.name}: failed ({reason})")
    return {
        "image": str(image),
        "status": "failed",
        "exit_code": 2,
        "failure_reason": reason,
        "stdout": [],
        "stderr": [reason],
    }


if __name__ == "__main__":
    raise SystemExit(main())
