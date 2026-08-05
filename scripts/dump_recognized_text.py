"""Dump per-page recognized text: manga-ocr readings + Paddle crop columns."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dump recognized text per page")
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--output", required=True, help="markdown output path")
    args = parser.parse_args(argv)

    cv_dir = Path(args.cv_dir)
    md = []
    for path in sorted(glob.glob(str(cv_dir / "*.cv.json"))):
        page_json = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        md.append(f"# {page_json['page']} 页 识别文本")
        md.append("")
        md.append("| 岛 | manga-ocr 整岛文本(权威) | Paddle 裁剪列(置信度) | 判定 |")
        md.append("|---|---|---|---|")
        for verdict in page_json["verdicts"]:
            if verdict["status"] == "dropped_tiny":
                md.append(f"| {verdict['island_id']} | — | — | 舍弃 {verdict.get('dropped_reason', '')} |")
                continue
            mark = "✅" if verdict["status"] == "pass" else "⚠️"
            paddle = "<br>".join(verdict["paddle_columns"]) or "—"
            md.append(f"| {verdict['island_id']} | {verdict['manga_text'] or '—'} | {paddle} | {mark} |")
        md.append("")
    Path(args.output).write_text("\n".join(md), encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
