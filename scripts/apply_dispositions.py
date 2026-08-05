"""Apply GUI dispositions to the M2 final texts.

Reads ``{page}.dispositions.json`` (from the review GUI) and rewrites
``{page}.final.json``: approve keeps the manga text, correct replaces it,
reject drops the island, sfx marks the island role.  The lazy switch result
is applied the same way (approve = manga).  The output keeps the page hash
binding for the M3 cache signature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_localizer.cross_validate import DEFAULT_MIN_ISLAND_AREA
from manga_localizer.island_contract import load_island_dataset


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def apply_dispositions(
    final_path: Path,
    dispositions_path: Path,
    dataset_dir: Path,
) -> dict:
    """Merge GUI dispositions into the M2 final texts.

    ``dataset_dir`` is only used for the islands.json source hash; the island
    list itself comes from ``final.json`` (which already excludes tiny drops).
    """
    final = _load_json(final_path)
    dispositions = _load_json(dispositions_path)
    if (Path(dataset_dir) / "islands.json").is_file():
        dataset = load_island_dataset(Path(dataset_dir) / "islands.json")
        final["effective_sha256"] = dataset.effective_sha256
    items = dispositions.get("items", [])
    decision_map = {item["island_id"]: item for item in items}
    kept = []
    summary = {"approve": 0, "correct": 0, "reject": 0, "sfx": 0}
    for island in final.get("islands", []):
        island_id = island["island_id"]
        item = decision_map.get(island_id)
        if item is None:
            kept.append(dict(island))
            continue
        decision = item.get("decision") or "approve"
        summary[decision] = summary.get(decision, 0) + 1
        if decision == "reject":
            continue
        if decision == "correct":
            text = str(item.get("corrected_text") or "").strip()
            if not text:
                raise ValueError(f"{island_id}: correct decision needs corrected_text")
            kept.append(
                {
                    "island_id": island_id,
                    "final_text": text,
                    "source": "corrected",
                    "review_status": "corrected",
                }
            )
            continue
        role = "sfx" if decision == "sfx" else island.get("role", "text")
        kept.append(
            {
                **island,
                "review_status": decision,
                "role": role,
            }
        )
    final["islands"] = kept
    final["dispositions"] = summary
    return final


def _existing_text(final: dict, island_id: str) -> str:
    for item in final.get("islands", []):
        if item["island_id"] == island_id:
            return item.get("final_text", "")
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply GUI dispositions to final texts")
    parser.add_argument("--m2-dir", required=True, help="M2 artifacts dir")
    parser.add_argument("--dataset-root", required=True, help="upstream outputs/exam/dataset")
    parser.add_argument("--pages", default="26-35")
    parser.add_argument("--write", action="store_true", help="rewrite {page}.final.json in place")
    args = parser.parse_args(argv)

    m2_dir = Path(args.m2_dir).resolve()
    root = Path(args.dataset_root).resolve()
    pages = []
    for raw in args.pages.split(","):
        item = raw.strip()
        if "-" in item:
            left, right = item.split("-", 1)
            pages.extend(str(n) for n in range(int(left), int(right) + 1))
        else:
            pages.append(item)

    totals = {}
    for page in pages:
        final_path = m2_dir / f"{page}.final.json"
        dispositions_path = m2_dir / f"{page}.dispositions.json"
        if not final_path.is_file() or not dispositions_path.is_file():
            print(f"{page}: skipped (missing final/dispositions)")
            continue
        updated = apply_dispositions(final_path, dispositions_path, root / page)
        if args.write:
            final_path.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        for key, value in updated.get("dispositions", {}).items():
            totals[key] = totals.get(key, 0) + value
        print(f"{page}: {updated.get('dispositions', {})} kept={len(updated['islands'])}")
    print("totals:", totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
