"""M4 batch: typeset all pages from erased images + M3 translations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from manga_localizer.island_contract import load_island_dataset
from manga_localizer.island_dry_run import OcrLine
from manga_localizer.island_state import build_page_state, write_page_state
from manga_localizer.island_typeset import (
    TypesetConfig,
    infer_island_directions,
    typeset_island_page,
)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4 batch typesetting")
    parser.add_argument("--dataset-root", required=True, help="upstream outputs/exam/dataset")
    parser.add_argument("--erase-dir", required=True, help="erased images dir (erase2)")
    parser.add_argument("--translation-dir", required=True, help="M3 translation dir")
    parser.add_argument("--paddle-dir", required=True, help="M2 paddle lines dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pages", default="26-35")
    parser.add_argument("--min-ratio", type=float, default=0.6)
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).resolve()
    erase_dir = Path(args.erase_dir).resolve()
    translation_dir = Path(args.translation_dir).resolve()
    paddle_dir = Path(args.paddle_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _parse_pages(args.pages)
    config = TypesetConfig(min_ratio=args.min_ratio)

    report = []
    for page in pages:
        dataset = load_island_dataset(root / page / "islands.json")
        translations_payload = json.loads(
            (translation_dir / f"{page}.translation.json").read_text(encoding="utf-8-sig")
        )
        translations = {
            item["island_id"]: str(item.get("translated_text") or "")
            for item in translations_payload.get("islands", [])
        }
        paddle_payload = json.loads(
            (paddle_dir / f"{page}.paddle.lines.json").read_text(encoding="utf-8-sig")
        )
        paddle_lines = [
            OcrLine(
                polygon=tuple((float(p[0]), float(p[1])) for p in item["polygon"]),
                text=str(item.get("text") or ""),
            )
            for item in paddle_payload.get("lines", [])
        ]
        # Horizontal typesetting for pages whose Photos blue boxes are
        # mostly wider than tall (afterword pages etc.); falls back to
        # vertical when no selection_blocks.json exists.
        selection_json = root / page / "selection_blocks.json"
        direction_overrides = None
        if selection_json.is_file():
            direction_overrides = infer_island_directions(selection_json, dataset.islands)
        started = time.perf_counter()
        with Image.open(erase_dir / f"{page}.erased.png") as erased:
            result = typeset_island_page(
                erased,
                dataset,
                translations,
                paddle_lines,
                config=config,
                direction_overrides=direction_overrides,
            )
        seconds = round(time.perf_counter() - started, 3)
        result.image.save(out_dir / f"{page}.typeset.png")
        state = build_page_state(
            page=page,
            original_path=root / page / "original.jpg",
            erased_path=erase_dir / f"{page}.erased.png",
            dataset=dataset,
            translations_payload=translations_payload,
            audits=result.audits,
            paddle_lines=paddle_lines,
        )
        write_page_state(state, out_dir / f"{page}.state.json")
        audit_payload = {
            "page": page,
            "typeset_seconds": seconds,
            "islands": [
                {
                    "island_id": item.island_id,
                    "status": item.status,
                    "reasons": list(item.reasons),
                    "font_size": item.font_size,
                    "stroke_width": item.stroke_width,
                    "tracking": item.tracking,
                    "column_gap": item.column_gap,
                    "area_base": item.area_base,
                    "area_text": item.area_text,
                    "area_ratio": item.area_ratio,
                    "center_offset": item.center_offset,
                    "center_tolerance": item.center_tolerance,
                }
                for item in result.audits
            ],
        }
        (out_dir / f"{page}.typeset.audit.json").write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from collections import Counter

        statuses = Counter(item.status for item in result.audits)
        report.append((page, dict(statuses), seconds))
        print(f"{page}: {dict(statuses)} {seconds}s")

    md = ["# M4 批量排版验收", "", "| 页 | typeset | pending_review | failed | skipped | 耗时(s) |", "|---:|---:|---:|---:|---:|---:|"]
    totals = {"typeset": 0, "typeset_pending_review": 0, "translation_failed": 0, "skipped": 0}
    for page, statuses, seconds in report:
        totals = {
            key: totals.get(key, 0) + statuses.get(key, 0) for key in totals
        }
        md.append(
            f"| {page} | {statuses.get('typeset', 0)} | {statuses.get('typeset_pending_review', 0)} | "
            f"{statuses.get('translation_failed', 0)} | {statuses.get('skipped', 0)} | {seconds} |"
        )
    md.append("")
    md.append(f"totals: {totals}")
    (out_dir / "REPORT_M4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out_dir / 'REPORT_M4.md'} | totals={totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
