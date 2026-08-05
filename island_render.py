"""Island pipeline entry points for the M1 (dry-run / erasure / OCR) gates.

Windowless: never opens a window and never controls the mouse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from PIL import Image

from manga_localizer.island_contract import load_island_dataset, validate_dataset_images
from manga_localizer.island_dry_run import OcrLine, render_island_dry_run
from manga_localizer.island_erasure import erase_island_image
from manga_localizer.island_ocr import windows_ocr_lines
from manga_localizer.ocr_adapter import DEFAULT_UPSTREAM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Island-based manga pipeline (M1 gates)")
    parser.add_argument("--dataset", required=True, help="page directory with islands.json")
    parser.add_argument("--image", help="input image (default original.jpg in the dataset dir)")
    parser.add_argument("--output", help="output path")
    parser.add_argument("--audit-json", help="optional audit JSON output path")
    parser.add_argument("--mask-output", help="optional erasure mask PNG output")
    parser.add_argument("--diff-output", help="optional erasure difference PNG output")
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="upstream project root")
    parser.add_argument("--max-layout-overflow", type=int, default=0, choices=(0, 5, 10, 20),
                        help="draw relaxed boundary for review only (0 = hard boundary)")
    parser.add_argument("--erase-strategy", choices=("auto", "solid-fill", "inpaint"), default="auto")
    parser.add_argument("--min-island-area", type=float, default=0.0,
                        help="drop islands with image_area below this (px^2, default 0 = keep all)")
    parser.add_argument("--ocr-only", action="store_true", help="OCR the image and dump line JSON")
    parser.add_argument("--dry-run", action="store_true", help="render the island review diagram")
    parser.add_argument("--erase-only", action="store_true", help="erase text inside islands")
    parser.add_argument("--ocr-lines", help="existing OCR lines JSON (polygon/text list)")
    parser.add_argument("--translate-only", action="store_true",
                        help="translate whole-island texts (precision switch)")
    parser.add_argument("--precision", choices=("local", "online"), default="local",
                        help="translation engine: local Sakura (default) or online OpenAI-compatible AI")
    parser.add_argument("--sakura-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--sakura-model", default="sukinishiro")
    parser.add_argument("--online-url", help="OpenAI-compatible base_url for --precision online")
    parser.add_argument("--online-model", help="model name for --precision online")
    parser.add_argument("--online-api-key", help="optional api key for --precision online")
    parser.add_argument("--glossary", help="glossary JSON path")
    parser.add_argument("--cache-dir", default="var/island_translation_cache",
                        help="translation cache directory")
    parser.add_argument("--mock-translations", help="deterministic mock translations JSON")
    parser.add_argument("--typeset-only", action="store_true", help="typeset translations inside islands")
    parser.add_argument("--erased-image", help="erased image for --typeset-only")
    parser.add_argument("--translation-json", help="translation JSON for --typeset-only")
    parser.add_argument("--min-ratio", type=float, default=None,
                        help="minimum typeset area ratio (default 0.6)")
    parser.add_argument("--reembed-state", help="re-embed from a saved page state.json")
    parser.add_argument("--edit-translations", help="island_id -> text JSON overriding the state translation")
    return parser


def _load_lines(path: Path) -> Sequence[OcrLine]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    lines = []
    for item in payload.get("lines", []):
        polygon = tuple((float(point[0]), float(point[1])) for point in item.get("polygon", []))
        lines.append(OcrLine(polygon=polygon, text=str(item.get("text") or "")))
    return tuple(lines)


def _dump_lines(lines: Sequence[OcrLine], engine: Optional[str]) -> Dict[str, Any]:
    return {
        "engine": engine,
        "lines": [
            {
                "polygon": [[float(point[0]), float(point[1])] for point in line.polygon],
                "text": line.text,
            }
            for line in lines
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset).resolve()
    islands_json = dataset_dir / "islands.json"
    dataset = load_island_dataset(islands_json)
    images = validate_dataset_images(dataset, dataset_dir)
    image_path = Path(args.image).resolve() if args.image else Path(images["original.jpg"])

    if args.ocr_only:
        engine, lines = windows_ocr_lines(image_path, upstream=Path(args.upstream))
        payload = _dump_lines(lines, engine)
        output = Path(args.output) if args.output else dataset_dir / "ocr.lines.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"engine={engine} lines={len(lines)} -> {output}")
        return 0

    if args.dry_run:
        ocr_lines = _load_lines(Path(args.ocr_lines)) if args.ocr_lines else None
        with Image.open(image_path) as image:
            canvas = render_island_dry_run(
                image,
                dataset,
                ocr_lines=ocr_lines,
                max_layout_overflow=args.max_layout_overflow,
            )
        output = Path(args.output) if args.output else dataset_dir / "dry_run.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        audit = {
            "page": dataset.page,
            "islands": len(dataset.islands),
            "image_size": list(dataset.image_size),
            "islands_json_sha256": dataset.source_sha256,
            "output": str(output),
        }
        if args.audit_json:
            Path(args.audit_json).write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"islands={len(dataset.islands)} -> {output}")
        return 0

    if args.erase_only:
        from manga_localizer.eraser import ErasureConfig

        with Image.open(image_path) as image:
            result = erase_island_image(
                image,
                dataset,
                config=ErasureConfig(style_override=args.erase_strategy),
                min_island_area=args.min_island_area,
            )
        output = Path(args.output) if args.output else dataset_dir / "erased.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(output)
        if args.mask_output:
            Path(args.mask_output).parent.mkdir(parents=True, exist_ok=True)
            result.island_masks.combined.save(args.mask_output)
        if args.diff_output:
            Path(args.diff_output).parent.mkdir(parents=True, exist_ok=True)
            result.difference.save(args.diff_output)
        audit = {
            "page": dataset.page,
            "islands": len(result.columns),
            "dropped_tiny": list(result.dropped_tiny),
            "erasure_seconds": round(result.erasure_seconds, 3),
            "pixel_audit_seconds": round(result.pixel_audit_seconds, 3),
            "changed_inside_islands": result.island_pixels.changed_inside_mask_pixels,
            "changed_outside_islands": result.island_pixels.changed_outside_mask_pixels,
            "outside_islands_unchanged": result.island_pixels.outside_mask_unchanged,
            "islands": [
                {
                    "island_id": column.column_id,
                    "strategy": column.strategy,
                    "classifier": column.classifier,
                    "mask_kind": column.mask_kind,
                    "mask_pixels": column.mask_pixels,
                    "changed_pixels": column.changed_pixels,
                    "fill_color": list(column.fill_color) if column.fill_color else None,
                    "backend": column.backend,
                }
                for column in result.columns
            ],
        }
        if args.audit_json:
            Path(args.audit_json).write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(
            f"erased -> {output} "
            f"(changed inside={result.island_pixels.changed_inside_mask_pixels}, "
            f"outside={result.island_pixels.changed_outside_mask_pixels})"
        )
        return 0

    if args.translate_only:
        return _run_translate(args, dataset, dataset_dir)

    if args.typeset_only:
        return _run_typeset(args, dataset, dataset_dir)

    if args.reembed_state:
        return _run_reembed(args, dataset, dataset_dir)

    raise SystemExit(
        "choose one of --ocr-only, --dry-run, --erase-only, --translate-only, "
        "--typeset-only, --reembed-state"
    )


def _run_reembed(args, dataset, dataset_dir: Path) -> int:
    from manga_localizer.island_state import (
        load_page_state,
        reembed_from_state,
        validate_state_inputs,
    )
    from manga_localizer.island_typeset import TypesetConfig

    if not args.erased_image:
        raise SystemExit("--reembed-state requires --erased-image (the same erased canvas)")
    state = load_page_state(Path(args.reembed_state))
    validate_state_inputs(state, dataset, Path(args.erased_image))
    edits = None
    if args.edit_translations:
        edits = json.loads(Path(args.edit_translations).read_text(encoding="utf-8-sig"))
    with Image.open(args.erased_image) as erased:
        result = reembed_from_state(
            state,
            dataset,
            erased,
            edits=edits,
            config=TypesetConfig(min_ratio=args.min_ratio) if args.min_ratio else TypesetConfig(),
        )
    output = Path(args.output) if args.output else dataset_dir / "reembedded.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output)
    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_json).write_text(
            json.dumps(
                {
                    "page": dataset.page,
                    "reembedded": True,
                    "typeset_seconds": round(result.typeset_seconds, 3),
                    "islands": [
                        {
                            "island_id": item.island_id,
                            "status": item.status,
                            "reasons": list(item.reasons),
                            "font_size": item.font_size,
                            "area_ratio": item.area_ratio,
                            "center_offset": item.center_offset,
                        }
                        for item in result.audits
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    statuses = {}
    for item in result.audits:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    print(f"reembedded -> {output} | {statuses}")
    return 0


def _run_typeset(args, dataset, dataset_dir: Path) -> int:
    from manga_localizer.island_typeset import TypesetConfig, typeset_island_page

    if not args.erased_image:
        raise SystemExit("--typeset-only requires --erased-image")
    if not args.translation_json:
        raise SystemExit("--typeset-only requires --translation-json")

    payload = json.loads(Path(args.translation_json).read_text(encoding="utf-8-sig"))
    translations = {
        item["island_id"]: str(item.get("translated_text") or "")
        for item in payload.get("islands", [])
    }
    paddle_lines = ()
    if args.ocr_lines:
        lines_payload = json.loads(Path(args.ocr_lines).read_text(encoding="utf-8-sig"))
        paddle_lines = [
            OcrLine(
                polygon=tuple((float(p[0]), float(p[1])) for p in item["polygon"]),
                text=str(item.get("text") or ""),
            )
            for item in lines_payload.get("lines", [])
        ]
    with Image.open(args.erased_image) as erased:
        result = typeset_island_page(
            erased,
            dataset,
            translations,
            paddle_lines,
            config=TypesetConfig(min_ratio=args.min_ratio) if args.min_ratio else TypesetConfig(),
        )
    output = Path(args.output) if args.output else dataset_dir / "typeset.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output)
    audit = {
        "page": dataset.page,
        "typeset_seconds": round(result.typeset_seconds, 3),
        "islands": [
            {
                "island_id": item.island_id,
                "status": item.status,
                "reasons": list(item.reasons),
                "font_size": item.font_size,
                "stroke_width": item.stroke_width,
                "tracking": item.tracking,
                "column_gap": item.column_gap,
                "direction": "vertical",
                "area_base": item.area_base,
                "area_text": item.area_text,
                "area_ratio": item.area_ratio,
                "center_offset": item.center_offset,
                "center_tolerance": item.center_tolerance,
            }
            for item in result.audits
        ],
    }
    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_json).write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    statuses = {}
    for item in result.audits:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    print(f"typeset -> {output} | {statuses}")
    return 0


def _run_translate(args, dataset, dataset_dir: Path) -> int:
    from manga_localizer.island_translate import (
        IslandTranslateConfig,
        MockIslandTranslator,
        translate_island_page,
    )
    from manga_localizer.translator import TranslationCache

    if not args.ocr_lines:
        raise SystemExit("--translate-only requires --ocr-lines (manga-ocr line JSON)")
    lines = _load_lines(Path(args.ocr_lines))
    texts = {}
    for line in lines:
        text = str(line.text or "").strip()
        if not text:
            continue
        cx = sum(point[0] for point in line.polygon) / len(line.polygon)
        cy = sum(point[1] for point in line.polygon) / len(line.polygon)
        for island in dataset.islands:
            x1, y1, x2, y2 = island.image_bbox
            if x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2:
                texts[island.id] = text
                break

    config = IslandTranslateConfig(
        precision=args.precision,
        sakura_base_url=args.sakura_url,
        sakura_model=args.sakura_model,
        online_base_url=args.online_url or "",
        online_model=args.online_model or "",
        online_api_key=args.online_api_key,
        glossary_path=args.glossary,
    )

    def factory(conf: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"):
        if args.mock_translations:
            payload = json.loads(Path(args.mock_translations).read_text(encoding="utf-8-sig"))
            return MockIslandTranslator(payload if isinstance(payload, dict) else payload.get("translations", {}))
        return None  # real HTTP translator built inside translate_island_page

    from manga_localizer.island_translate import _build_translator

    def real_factory(conf: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"):
        return _build_translator(conf, precision, prompt_style)

    cache = TranslationCache(Path(args.cache_dir))
    result = translate_island_page(
        dataset,
        texts,
        config,
        cache=cache,
        translator_factory=factory if args.mock_translations else real_factory,
    )
    payload = {
        "page": result.page,
        "precision": result.precision,
        "units": [
            {
                "island_id": unit.island_id,
                "source_text": unit.source_text,
                "translated_text": unit.translated_text,
                "status": unit.status,
                "backend": unit.backend,
                "model": unit.model,
                "cache_hit": unit.cache_hit,
                "attempts": list(unit.attempts),
                "failure_reason": unit.failure_reason,
            }
            for unit in result.units
        ],
        "failed_count": result.failed_count,
        "cache_hit_count": result.cache_hit_count,
    }
    output = Path(args.output) if args.output else dataset_dir / "translation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"precision={result.precision} units={len(result.units)} "
        f"failed={result.failed_count} cache_hits={result.cache_hit_count} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
