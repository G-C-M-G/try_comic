"""Batch cross-validation gate: manga-ocr text x Paddle island-crop geometry.

Produces per-page artifacts under the output directory:
  {page}.manga.json            manga-ocr island readings (line-JSON schema)
  {page}.paddle_crops.json     raw Paddle four-variant crop results
  {page}.paddle.lines.json     merged Paddle lines (best variant per island)
  {page}.cv.md / {page}.cv.json cross-validation table
  {page}.cv.annotated.png      filled_black annotation with Paddle boxes
  REPORT_CV.md                 page summary for the user gate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from manga_localizer.cross_validate import (
    cross_validate_page,
    render_verdict_markdown,
    verdicts_to_json,
)
from manga_localizer.island_contract import load_island_dataset
from manga_localizer.island_dry_run import OcrLine, render_island_dry_run
from manga_localizer.ocr_adapter import DEFAULT_UPSTREAM
from manga_localizer.errors import TranslationError


def _run(command: Sequence[str], *, timeout: float, cwd: Optional[Path] = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired as error:
        raise TranslationError(f"command timed out: {' '.join(command[:3])}") from error
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise TranslationError((stderr or stdout).strip()[-800:])
    return stdout


def _parse_pages(value: str) -> list[str]:
    pages = []
    for raw in value.split(","):
        item = raw.strip()
        if "-" in item:
            left, right = item.split("-", 1)
            pages.extend(str(number) for number in range(int(left), int(right) + 1))
        else:
            pages.append(item)
    return pages


def _load_ocr_lines(path: Path) -> list[OcrLine]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    lines = payload.get("lines") or []
    return [
        OcrLine(
            polygon=tuple((float(point[0]), float(point[1])) for point in item["polygon"]),
            text=str(item.get("text") or ""),
            confidence=(
                float(item["confidence"]) if item.get("confidence") is not None else None
            ),
        )
        for item in lines
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch manga x Paddle cross-validation gate")
    parser.add_argument("--dataset-root", required=True, help="upstream outputs/exam/dataset")
    parser.add_argument("--pages", default="26-35", help="page list, e.g. 26-35 or 26,28")
    parser.add_argument("--output-dir", required=True, help="gate artifact directory")
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="upstream project root")
    parser.add_argument("--manga-timeout", type=float, default=1800.0)
    parser.add_argument("--paddle-timeout", type=float, default=600.0)
    parser.add_argument("--skip-manga", action="store_true", help="reuse existing manga JSONs")
    parser.add_argument("--skip-paddle", action="store_true", help="reuse existing paddle crops JSONs")
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).resolve()
    upstream = Path(args.upstream).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _parse_pages(args.pages)
    dataset_dirs = [root / page for page in pages]

    if not args.skip_manga:
        _run(
            [
                sys.executable,
                str(Path(__file__).parent / "manga_ocr_islands.py"),
                "--datasets",
                *[str(directory) for directory in dataset_dirs],
                "--output-dir",
                str(out_dir),
                "--upstream",
                str(upstream),
            ],
            timeout=args.manga_timeout,
        )

    paddle_python = upstream / "paddle_venv" / "Scripts" / "python.exe"
    worker = upstream / "paddle_ocr_worker.py"
    if not paddle_python.is_file() or not worker.is_file():
        print(f"paddle worker unavailable: {paddle_python} / {worker}", file=sys.stderr)
        return 2

    report_rows = []
    for directory in dataset_dirs:
        page = directory.name
        dataset = load_island_dataset(directory / "islands.json")
        if not args.skip_paddle:
            manifest_path = out_dir / f"{page}.crops.manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": str((directory / "filled_black.png").resolve()),
                        "crops": [
                            {
                                "id": island.id,
                                "bbox": list(island.image_bbox),
                                "variants": ["raw", "raw2x", "clahe", "otsu"],
                            }
                            for island in dataset.islands
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            crops_path = out_dir / f"{page}.paddle_crops.json"
            _run(
                [
                    str(paddle_python),
                    str(worker),
                    str(manifest_path.resolve()),
                    str(crops_path.resolve()),
                    "--crops",
                    "--device",
                    "cpu",
                ],
                timeout=args.paddle_timeout,
                cwd=upstream,
            )
        lines_path = out_dir / f"{page}.paddle.lines.json"
        _run(
            [
                sys.executable,
                str(Path(__file__).parent / "merge_paddle_crops.py"),
                "--crops-json",
                str(out_dir / f"{page}.paddle_crops.json"),
                "--output",
                str(lines_path),
            ],
            timeout=120.0,
        )

        manga_path = out_dir / f"{page}.manga.json"
        if not manga_path.is_file():
            print(f"missing manga JSON for page {page}", file=sys.stderr)
            return 2
        manga_lines = _load_ocr_lines(manga_path)
        paddle_lines = _load_ocr_lines(lines_path)
        result = cross_validate_page(dataset, manga_lines, paddle_lines)

        (out_dir / f"{page}.cv.md").write_text(
            render_verdict_markdown(result), encoding="utf-8"
        )
        (out_dir / f"{page}.cv.json").write_text(
            json.dumps(verdicts_to_json(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with Image.open(directory / "filled_black.png") as image:
            canvas = render_island_dry_run(
                image,
                dataset,
                ocr_lines=paddle_lines,
                max_layout_overflow=0,
            )
            canvas.save(out_dir / f"{page}.cv.annotated.png")

        report_rows.append(
            {
                "page": page,
                "islands": len(dataset.islands),
                "pass": result.pass_count,
                "review": result.review_count,
                "dropped_tiny": result.dropped_count,
            }
        )
        print(
            f"{page}: pass={result.pass_count} review={result.review_count} "
            f"dropped={result.dropped_count}"
        )

    report_path = out_dir / "REPORT_CV.md"
    lines = [
        "# 双引擎交叉验证门禁汇总（manga-ocr × Paddle 岛级裁剪）",
        "",
        "| 页 | 岛数 | 通过 | REVIEW | 微岛舍弃 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['page']} | {row['islands']} | {row['pass']} | {row['review']} | {row['dropped_tiny']} |"
        )
    lines.append("")
    (report_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
