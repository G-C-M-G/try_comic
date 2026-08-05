"""Generate the island -> lines -> full text -> sentences correspondence table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_localizer.island_contract import load_island_dataset
from manga_localizer.island_dry_run import OcrLine
from manga_localizer.sentence_splitter import split_island_dataset


def _load_lines(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    lines = payload.get("lines") or payload.get("pages", [{}])[0].get("predictions", [])
    parsed = []
    for item in lines:
        parsed.append(
            OcrLine(
                polygon=tuple(
                    (float(point[0]), float(point[1])) for point in item["polygon"]
                ),
                text=str(item.get("text") or ""),
                confidence=(
                    float(item["confidence"]) if item.get("confidence") is not None else None
                ),
            )
        )
    engine = payload.get("engine") or payload.get("backend")
    return engine, parsed


def _column_confidence(column) -> float | None:
    scores = [line.confidence for line in column.lines if line.confidence is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def build_table(dataset, engine, lines) -> dict:
    result = split_island_dataset(dataset, lines)
    rows = []
    for unit in result.units:
        rows.append(
            {
                "island_id": unit.island.id,
                "image_bbox": list(unit.island.image_bbox),
                "columns": [
                    {
                        "text": column.text,
                        "direction": column.direction,
                        "bbox": list(column.bbox),
                        "confidence": _column_confidence(column),
                    }
                    for column in unit.columns
                ],
                "full_text": unit.full_text,
                "sentences": [
                    {
                        "text": sentence.text,
                        "column_ids": list(sentence.column_ids),
                        "signal": sentence.boundary_signal,
                        "needs_review": sentence.needs_review,
                    }
                    for sentence in unit.sentences
                ],
            }
        )
    return {
        "page": dataset.page,
        "engine": engine,
        "lines_total": len(lines),
        "assigned_islands": len(rows),
        "empty_islands": [island.id for island in result.empty_islands],
        "unassigned_lines": [line.text for line in result.unassigned],
        "conflicts": [{"text": line.text, "hits": list(hits)} for line, hits in result.conflicts],
        "units": rows,
    }


def render_markdown(table: dict) -> str:
    md = []
    md.append(f"# {table['page']} 页 岛→行→整岛原文→分句 对应表")
    md.append("")
    md.append(
        f"- 引擎: {table['engine']} | OCR 行数: {table['lines_total']} | "
        f"岛: {table['assigned_islands']}"
    )
    md.append(
        f"- 空岛: {table['empty_islands'] or '无'} | 无归属行: {table['unassigned_lines'] or '无'} | "
        f"跨岛冲突: {table['conflicts'] or '无'}"
    )
    md.append("")
    for unit in table["units"]:
        md.append(f"## {unit['island_id']}  bbox={unit['image_bbox']}")
        md.append("")
        md.append("| 列 | 方向 | 置信 | 文本 |")
        md.append("|---|---|---|---|")
        for index, column in enumerate(unit["columns"], 1):
            confidence = column.get("confidence")
            conf_text = "" if confidence is None else f"{confidence:.3f}"
            md.append(f"| c{index} | {column['direction']} | {conf_text} | {column['text']} |")
        md.append("")
        md.append(f"**整岛原文（阅读顺序拼接）**: {unit['full_text']}")
        md.append("")
        for sentence in unit["sentences"]:
            flag = " [REVIEW]" if sentence["needs_review"] else ""
            md.append(
                f"- 句[{sentence['signal']}{flag}]: {sentence['text']}  ← {sentence['column_ids']}"
            )
        md.append("")
    return "\n".join(md)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build island sentence correspondence table")
    parser.add_argument("--dataset", required=True, help="page directory with islands.json")
    parser.add_argument("--ocr-lines", required=True, help="OCR lines JSON")
    parser.add_argument("--output-json", help="output JSON path")
    parser.add_argument("--output-md", help="output Markdown path")
    args = parser.parse_args(argv)
    dataset = load_island_dataset(Path(args.dataset) / "islands.json")
    engine, lines = _load_lines(Path(args.ocr_lines))
    table = build_table(dataset, engine, lines)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.output_md:
        Path(args.output_md).write_text(render_markdown(table), encoding="utf-8")
    print(f"units={len(table['units'])} empty={table['empty_islands']} "
          f"unassigned={table['unassigned_lines']} conflicts={table['conflicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
