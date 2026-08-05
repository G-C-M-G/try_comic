"""Print the cross-validation REVIEW detail list for the gate report."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="List REVIEW verdicts from a cv output dir")
    parser.add_argument("--cv-dir", required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in sorted(glob.glob(str(Path(args.cv_dir) / "*.cv.json"))):
        page_json = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for verdict in page_json["verdicts"]:
            if verdict["status"] == "review":
                rows.append(
                    (
                        page_json["page"],
                        verdict["island_id"],
                        verdict["review_reasons"],
                        verdict["manga_text"],
                        verdict["paddle_columns"],
                        verdict["low_confidence_lines"],
                    )
                )
    print(f"total review: {len(rows)}")
    for page, island_id, reasons, manga, paddle, low in rows:
        print(f"{page}/{island_id}: {reasons}")
        print(f"   manga: {manga}")
        print(f"   paddle: {paddle} low={low}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
