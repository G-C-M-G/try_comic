"""Build the M2 final-texts gate table."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    cv_dir = Path(argv[1]) if len(argv) > 1 else Path("var/island_m1/m2")
    md = ["# M2 最终原文清单（manga 权威，懒人开关默认承认 REVIEW）", ""]
    total = 0
    for path in sorted(glob.glob(str(cv_dir / "*.final.json"))):
        page = json.loads(Path(path).read_text(encoding="utf-8"))
        md.append(f"## {page['page']} 页")
        md.append("")
        md.append("| 岛 | 最终原文 | 状态 |")
        md.append("|---|---|---|")
        for item in page["islands"]:
            total += 1
            text = item["final_text"] or "—"
            md.append(f"| {item['island_id']} | {text} | {item['review_status']} |")
        md.append("")
    out = cv_dir / "FINAL_TEXTS.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"islands={total} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
