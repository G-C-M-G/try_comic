"""M3: whole-island translation batch (D stage).

Input: M2 final texts (``{page}.final.json``).
Output per page: ``{page}.translation.json`` with island -> translation,
backend/model/cache state, role (sfx), review_status.

Precision:
  local  (default): every island through the local Sakura endpoint.
  online: the whole page is sent to the online AI in ONE numbered request
          (context-rich, colloquial); islands whose line is missing or
          malformed fall back to local Sakura.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_localizer.island_contract import load_island_dataset
from manga_localizer.island_translate import (
    IslandTranslateConfig,
    IslandTranslationCache,
    PageBatchCache,
    translate_island_page,
    translate_page_batch,
)
from manga_localizer.ocr_adapter import DEFAULT_UPSTREAM
from manga_localizer.translator import load_glossary


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


def _final_texts(final_path: Path) -> dict:
    payload = json.loads(Path(final_path).read_text(encoding="utf-8-sig"))
    texts = {}
    roles = {}
    review = {}
    for item in payload.get("islands", []):
        text = str(item.get("final_text") or "").strip()
        if not text:
            continue
        texts[item["island_id"]] = text
        roles[item["island_id"]] = item.get("role", "text")
        review[item["island_id"]] = item.get("review_status", "pass")
    return payload, texts, roles, review


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M3 whole-island translation batch")
    parser.add_argument("--m2-dir", required=True, help="M2 artifacts dir with final.json")
    parser.add_argument("--dataset-root", required=True, help="upstream outputs/exam/dataset")
    parser.add_argument("--pages", default="26-35")
    parser.add_argument("--output-dir", required=True, help="translation output dir")
    parser.add_argument("--precision", choices=("local", "online"), default="local")
    parser.add_argument("--sakura-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--sakura-model", default="sukinishiro")
    parser.add_argument("--online-url", help="OpenAI-compatible base_url (precision=online)")
    parser.add_argument("--online-model", help="model name (precision=online)")
    parser.add_argument("--online-api-key")
    parser.add_argument("--glossary", default="config/glossary.json")
    parser.add_argument("--cache-dir", default="var/island_translation_cache")
    parser.add_argument("--batch-cache-dir", default="var/island_batch_cache")
    args = parser.parse_args(argv)

    m2_dir = Path(args.m2_dir).resolve()
    root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _parse_pages(args.pages)

    glossary = load_glossary(Path(args.glossary))
    cache = IslandTranslationCache(Path(args.cache_dir))
    batch_cache = PageBatchCache(Path(args.batch_cache_dir))

    rows = []
    totals = {"translated": 0, "failed": 0, "cache_hits": 0, "fallback": 0}
    for page in pages:
        dataset = load_island_dataset(root / page / "islands.json")
        final_path = m2_dir / f"{page}.final.json"
        if not final_path.is_file():
            print(f"{page}: missing final.json, skipped")
            continue
        payload, texts, roles, review = _final_texts(final_path)

        config = IslandTranslateConfig(
            precision=args.precision,
            sakura_base_url=args.sakura_url,
            sakura_model=args.sakura_model,
            online_base_url=args.online_url or "",
            online_model=args.online_model or "",
            online_api_key=args.online_api_key,
            glossary_path=str(Path(args.glossary).resolve()),
        )

        batch_translations: dict[str, str] = {}
        batch_meta: dict[str, dict] = {}
        fallback_islands: list[str] = []

        if args.precision == "online":
            result = translate_page_batch(
                dataset,
                texts,
                config,
                glossary=glossary,
                cache=batch_cache,
            )
            batch_translations = dict(result.translations)
            fallback_islands = list(result.failures)
            if result.cache_hit:
                totals["cache_hits"] += len(result.translations)
            for island_id in result.translations:
                batch_meta[island_id] = {
                    "backend": result.backend,
                    "model": result.model,
                    "cache_hit": result.cache_hit,
                }
            print(
                f"{page}: batch online -> {len(result.translations)} ok, "
                f"{len(result.failures)} fallback, errors={len(result.parse_errors)}"
            )

        fallback_texts = {island_id: texts[island_id] for island_id in fallback_islands}
        fallback_result = None
        if fallback_texts:
            fallback_result = translate_island_page(
                dataset,
                fallback_texts,
                IslandTranslateConfig(
                    precision="local",
                    sakura_base_url=args.sakura_url,
                    sakura_model=args.sakura_model,
                    glossary_path=str(Path(args.glossary).resolve()),
                ),
                glossary=glossary,
                cache=cache,
            )
            totals["fallback"] += len(fallback_texts)
        elif args.precision == "local":
            fallback_result = translate_island_page(
                dataset,
                texts,
                config,
                glossary=glossary,
                cache=cache,
            )

        translations: dict[str, str] = dict(batch_translations)
        meta: dict[str, dict] = dict(batch_meta)
        if fallback_result is not None:
            for unit in fallback_result.units:
                translations[unit.island_id] = unit.translated_text or ""
                meta[unit.island_id] = {
                    "backend": unit.backend,
                    "model": unit.model,
                    "cache_hit": unit.cache_hit,
                    "status": unit.status,
                    "failure_reason": unit.failure_reason,
                    "kind": unit.kind,
                }
            totals["cache_hits"] += fallback_result.cache_hit_count

        islands_out = []
        for island_id, source in texts.items():
            translated = translations.get(island_id)
            item_meta = meta.get(island_id, {})
            status = item_meta.get("status") or ("translated" if translated else "failed")
            if status == "translated":
                totals["translated"] += 1
            elif status == "kept-source":
                totals.setdefault("kept_source", 0)
                totals["kept_source"] += 1
            else:
                totals["failed"] += 1
            islands_out.append(
                {
                    "island_id": island_id,
                    "role": roles.get(island_id, "text"),
                    "review_status": review.get(island_id, "pass"),
                    "source_text": source,
                    "translated_text": translated,
                    "status": status,
                    "kind": item_meta.get("kind", "text"),
                    "backend": item_meta.get("backend"),
                    "model": item_meta.get("model"),
                    "cache_hit": bool(item_meta.get("cache_hit")),
                    "failure_reason": item_meta.get("failure_reason"),
                }
            )
        page_payload = {
            "page": page,
            "precision": args.precision,
            "effective_sha256": dataset.effective_sha256,
            "islands": islands_out,
        }
        (out_dir / f"{page}.translation.json").write_text(
            json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok = sum(1 for item in islands_out if item["status"] == "translated")
        rows.append((page, len(islands_out), ok))
        print(f"{page}: {ok}/{len(islands_out)} translated")

    md = ["# M3 汇总（整岛翻译）", "", "| 页 | 岛 | 已译 |", "|---:|---:|---:|"]
    for page, count, ok in rows:
        md.append(f"| {page} | {count} | {ok} |")
    md.append("")
    md.append(f"totals: {totals}")
    (out_dir / "REPORT_M3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out_dir / 'REPORT_M3.md'} | totals={totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
