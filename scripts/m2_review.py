"""M2: REVIEW-island manga re-read from the ORIGINAL image (no bbox expansion),
cross-validation re-run, and the final-text gate.

Per page:
  {page}.manga.json               manga-ocr re-read (original bbox, original.jpg)
  {page}.paddle_crops.json        Paddle four-variant crops (original bbox)
  {page}.paddle.lines.json        merged Paddle lines
  {page}.m2.cv.md / .m2.cv.json   cross-validation after re-read
  {page}.m2.md                    REVIEW before/after comparison (manga text)
  {page}.m2.annotated.png         original image, 6px yellow-stroke islands
  {page}.final.json               final texts (manga authority; dropped excluded)
  REPORT_M2A.md                   page summary for the user gate
"""

from __future__ import annotations

import argparse
import glob
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


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _lines_from_payload(payload: dict, *, with_island: bool = False) -> list[OcrLine]:
    lines = payload.get("lines") or []
    result = []
    for item in lines:
        line = OcrLine(
            polygon=tuple((float(point[0]), float(point[1])) for point in item["polygon"]),
            text=str(item.get("text") or ""),
            confidence=(
                float(item["confidence"]) if item.get("confidence") is not None else None
            ),
        )
        if with_island and item.get("island_id"):
            object.__setattr__(line, "island_id", item["island_id"])
        result.append(line)
    return result


def _final_texts(dataset, new_cv_json: dict) -> dict:
    """Final text per retained island: manga authority for every island,
    including REVIEW ones (lazy-switch default: accept the manga reading)."""
    final = {}
    for verdict in new_cv_json["verdicts"]:
        if verdict["status"] == "dropped_tiny":
            continue
        final[verdict["island_id"]] = verdict["manga_text"]
    return final


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2 REVIEW re-read and final-text gate")
    parser.add_argument("--cv-dir", required=True, help="P1 cross-validation artifacts dir")
    parser.add_argument("--dataset-root", required=True, help="upstream outputs/exam/dataset")
    parser.add_argument("--pages", default="26-35")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument("--manga-timeout", type=float, default=1800.0)
    parser.add_argument("--paddle-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    cv_dir = Path(args.cv_dir).resolve()
    root = Path(args.dataset_root).resolve()
    upstream = Path(args.upstream).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _parse_pages(args.pages)
    dataset_dirs = [root / page for page in pages]

    paddle_python = upstream / "paddle_venv" / "Scripts" / "python.exe"
    worker = upstream / "paddle_ocr_worker.py"

    _run(
        [
            sys.executable,
            str(Path(__file__).parent / "manga_ocr_islands.py"),
            "--datasets",
            *[str(directory) for directory in dataset_dirs],
            "--image-name",
            "original.jpg",
            "--output-dir",
            str(out_dir),
            "--upstream",
            str(upstream),
        ],
        timeout=args.manga_timeout,
    )

    report_rows = []
    total_review_before = 0
    total_review_after = 0
    for directory in dataset_dirs:
        page = directory.name
        dataset = load_island_dataset(directory / "islands.json")
        manifest = {
            "source": str((directory / "original.jpg").resolve()),
            "crops": [
                {
                    "id": island.id,
                    "bbox": list(island.image_bbox),
                    "variants": ["raw", "raw2x", "clahe", "otsu"],
                }
                for island in dataset.islands
            ],
        }
        manifest_path = out_dir / f"{page}.crops.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _run(
            [
                str(paddle_python),
                str(worker),
                str(manifest_path.resolve()),
                str(out_dir / f"{page}.paddle_crops.json"),
                "--crops",
                "--device",
                "cpu",
            ],
            timeout=args.paddle_timeout,
            cwd=upstream,
        )
        _run(
            [
                sys.executable,
                str(Path(__file__).parent / "merge_paddle_crops.py"),
                "--crops-json",
                str(out_dir / f"{page}.paddle_crops.json"),
                "--output",
                str(out_dir / f"{page}.paddle.lines.json"),
            ],
            timeout=120.0,
        )

        manga_lines = _lines_from_payload(_load_json(out_dir / f"{page}.manga.json"))
        paddle_lines = _lines_from_payload(
            _load_json(out_dir / f"{page}.paddle.lines.json"), with_island=True
        )
        old_cv = _load_json(cv_dir / f"{page}.cv.json")
        new_cv = cross_validate_page(dataset, manga_lines, paddle_lines)
        new_cv_json = verdicts_to_json(new_cv)

        (out_dir / f"{page}.m2.cv.md").write_text(
            render_verdict_markdown(new_cv), encoding="utf-8"
        )
        (out_dir / f"{page}.m2.cv.json").write_text(
            json.dumps(new_cv_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with Image.open(directory / "original.jpg") as image:
            canvas = render_island_dry_run(
                image,
                dataset,
                ocr_lines=paddle_lines,
                max_layout_overflow=0,
                stroke_color=(255, 215, 0),
                stroke_width=6,
            )
            canvas.save(out_dir / f"{page}.m2.annotated.png")

        final = _final_texts(dataset, new_cv_json)
        final_payload = {
            "page": page,
            "image_size": list(dataset.image_size),
            "effective_sha256": dataset.effective_sha256,
            "islands": [
                {
                    "island_id": island.id,
                    "final_text": final.get(island.id, ""),
                    "source": "manga",
                    "review_status": next(
                        (
                            v["status"]
                            for v in new_cv_json["verdicts"]
                            if v["island_id"] == island.id
                        ),
                        "dropped_tiny",
                    ),
                }
                for island in dataset.islands
            ],
        }
        (out_dir / f"{page}.final.json").write_text(
            json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        md = [f"# {page} 页 M2 修正对比（原图重读，无扩区）", ""]
        md.append("| 岛 | 修正前 manga | 修正后 manga | 修正前判定 | 修正后判定 |")
        md.append("|---|---|---|---|---|")
        old_by_id = {verdict["island_id"]: verdict for verdict in old_cv["verdicts"]}
        new_by_id = {verdict["island_id"]: verdict for verdict in new_cv_json["verdicts"]}
        for verdict_id, old in old_by_id.items():
            if old["status"] != "review":
                continue
            new = new_by_id.get(verdict_id)
            md.append(
                f"| {verdict_id} | {old.get('manga_text', '')} |"
                f" {new.get('manga_text', '') if new else ''} |"
                f" {old['status']} | {new['status'] if new else ''} |"
            )
        (out_dir / f"{page}.m2.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        before = sum(1 for v in old_cv["verdicts"] if v["status"] == "review")
        total_review_before += before
        total_review_after += new_cv.review_count
        report_rows.append(
            {
                "page": page,
                "review_before": before,
                "pass_after": new_cv.pass_count,
                "review_after": new_cv.review_count,
                "dropped_after": new_cv.dropped_count,
            }
        )
        print(
            f"{page}: review {before} -> pass={new_cv.pass_count} "
            f"review={new_cv.review_count} dropped={new_cv.dropped_count}"
        )

    lines = [
        "# M2 汇总：REVIEW 岛原图重读 + 重跑交叉验证（无扩区，6px 黄描边）",
        "",
        "| 页 | REVIEW(前) | 通过 | 仍 REVIEW | 微岛 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['page']} | {row['review_before']} | {row['pass_after']} | "
            f"{row['review_after']} | {row['dropped_after']} |"
        )
    lines.append(f"| 合计 | {total_review_before} | | {total_review_after} | |")
    (out_dir / "REPORT_M2A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out_dir / 'REPORT_M2A.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
