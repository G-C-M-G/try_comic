"""Build the M3 gate table: island -> source -> translation per page."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    out_dir = Path(argv[0]) if argv else Path("var/island_m1/m3")
    md = ["# M3 翻译对照表（整岛一段，本地 Sakura）", ""]
    totals = {"translated": 0, "kept_source": 0, "review_lazy": 0}
    for path in sorted(glob.glob(str(out_dir / "*.translation.json"))):
        page = json.loads(Path(path).read_text(encoding="utf-8"))
        md.append(f"## {page['page']} 页（precision={page['precision']}）")
        md.append("")
        md.append("| 岛 | 原文 | 译文 | 状态/引擎 | 缓存 |")
        md.append("|---|---|---|---|---|")
        for item in page["islands"]:
            status = item["status"]
            if status == "translated":
                totals["translated"] += 1
            elif status == "kept-source":
                totals["kept_source"] += 1
            if item.get("review_status") == "review":
                totals["review_lazy"] += 1
            backend = item.get("backend") or "-"
            cache = "命中" if item.get("cache_hit") else "-"
            flag = " [REVIEW]" if item.get("review_status") == "review" else ""
            md.append(
                f"| {item['island_id']} | {item['source_text']} | "
                f"{item['translated_text'] or '—'} | {status}{flag}/{backend} | {cache} |"
            )
        md.append("")
    md.append(f"totals: {totals}")
    out = out_dir / "TRANSLATION_TABLE.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"-> {out} | {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
