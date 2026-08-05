from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from .dry_run import render_dry_run
from .eraser import ErasureConfig
from .errors import MangaLocalizerError
from .ocr_contract import load_ocr_contract, resolve_image_path
from .pipeline import (
    render_erasure,
    render_full_pipeline,
    render_state_typesetting,
    render_translation,
    render_typesetting,
)
from .review import (
    apply_review_decisions,
    bypass_pending_reviews,
    unchanged_review,
)
from .translator import (
    MockTranslator,
    OpenAICompatibleTranslator,
    TranslationCache,
    TranslationError,
    load_glossary,
    load_mock_translations,
)
from .typesetter import LayoutConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable manga localization consumer (stages B-F)."
    )
    parser.add_argument("--image", type=Path, help="Source image; overrides stale OCR paths.")
    parser.add_argument("--ocr", type=Path, required=True, help="Read-only schema 4 OCR JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG for dry-run/erasure/typesetting or JSON for translation.",
    )
    parser.add_argument("--schema", type=Path, help="Override the schema 4 JSON schema path.")
    parser.add_argument(
        "--sample-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional image directory used after the OCR JSON directory.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        help="Hash-bound *.ocr.review.json decisions; pending decisions remain blocked.",
    )
    parser.add_argument(
        "--no-manual-review",
        action="store_true",
        help="Explicitly bypass unresolved review while retaining every risk record.",
    )
    parser.add_argument(
        "--max-layout-overflow",
        type=int,
        choices=(10, 20),
        default=10,
        help="Use 10%% soft limit or allow direct 20%% hard fallback; default: 10.",
    )
    parser.add_argument("--audit-json", type=Path, help="Optional audit JSON path.")
    parser.add_argument(
        "--state",
        type=Path,
        help="Existing *.state.json to re-embed without OCR, erasure, or translation.",
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        help="State output path; defaults beside the final image or updates --state.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="Directory for erased/mask/diff/translation/audit stage artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Draw numbered region/mask/layout audit only; no erasure or translation.",
    )
    parser.add_argument(
        "--erase-only",
        action="store_true",
        help="Erase reviewed OCR columns without translating or typesetting.",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Translate complete reviewed units and write an auditable JSON artifact.",
    )
    parser.add_argument(
        "--typeset-only",
        action="store_true",
        help="Typeset an existing stage D translation onto an erased image.",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        help="Lossless combined erasure mask PNG; defaults beside --output.",
    )
    parser.add_argument(
        "--diff-output",
        type=Path,
        help="Lossless absolute pixel-difference PNG; defaults beside --output.",
    )
    parser.add_argument(
        "--erase-strategy",
        choices=("auto", "solid-fill", "inpaint"),
        default="auto",
        help="Replaceable erasure style policy; default: auto.",
    )
    parser.add_argument(
        "--translator",
        choices=("sakura", "openai-compatible", "mock"),
        default="sakura",
        help="Translation backend; default: local Sakura.",
    )
    parser.add_argument(
        "--translation-base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--translation-model",
        default="sukinishiro",
        help="OpenAI-compatible model ID.",
    )
    parser.add_argument(
        "--translation-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing an optional API key; never audited.",
    )
    parser.add_argument(
        "--translation-timeout",
        type=float,
        default=90.0,
        help="Timeout per model request in seconds; default: 90.",
    )
    parser.add_argument(
        "--translation-attempts",
        type=int,
        default=2,
        help="Maximum attempts per translator and complete unit; default: 2.",
    )
    parser.add_argument(
        "--translation-prompt-style",
        choices=("auto", "contextual", "concise"),
        default="auto",
        help="Sakura prompt policy; auto uses contextual then concise fallback.",
    )
    parser.add_argument(
        "--translation-cache-dir",
        type=Path,
        default=Path("var/translation-cache"),
        help="Content-addressed translation cache directory.",
    )
    parser.add_argument(
        "--no-translation-cache",
        action="store_true",
        help="Disable translation cache reads and writes.",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("config/glossary.json"),
        help="Terminology and name mapping JSON.",
    )
    parser.add_argument(
        "--mock-translations",
        type=Path,
        help="unit_id-to-Chinese mapping used only with --translator mock.",
    )
    parser.add_argument(
        "--erased-image",
        type=Path,
        help="Erased PNG used as the background for --typeset-only.",
    )
    parser.add_argument(
        "--translation-json",
        type=Path,
        help="Stage D translation artifact used by --typeset-only.",
    )
    parser.add_argument(
        "--erasure-audit",
        type=Path,
        help="Stage C audit used to select bubble or artwork text style.",
    )
    parser.add_argument(
        "--layout-font",
        type=Path,
        help="Chinese TTF/TTC/OTF font; defaults to an installed CJK font.",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=10,
        help="Smallest font size tried before layout failure; default: 10.",
    )
    parser.add_argument(
        "--outline-width",
        type=int,
        default=2,
        help="White outline width for artwork text; default: 2.",
    )
    parser.add_argument(
        "--terminal-punctuation-font-ratio",
        type=float,
        default=0.68,
        help="Font ratio for a terminal punctuation run occupying one cell; default: 0.68.",
    )
    parser.add_argument(
        "--preserve-unit-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep complete translated-unit and reviewed source-box centers identical.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        contract = load_ocr_contract(args.ocr, args.schema)
        application = (
            apply_review_decisions(contract, args.review)
            if args.review
            else unchanged_review(contract)
        )
        if args.no_manual_review:
            application = bypass_pending_reviews(application)
        selected_modes = sum(
            (
                args.dry_run,
                args.erase_only,
                args.translate_only,
                args.typeset_only,
                args.state is not None,
            )
        )
        if selected_modes > 1:
            parser.error(
                "--dry-run, --erase-only, --translate-only, --typeset-only, and --state "
                "are mutually exclusive"
            )
        if args.state is not None:
            typeset_result, updated_state, invalidated = render_state_typesetting(
                args.state,
                application,
                args.output,
                image_path=args.image,
                erased_image_path=args.erased_image,
                audit_path=args.audit_json,
                state_output_path=args.state_output,
                config=_layout_config(args),
            )
            print(
                f"state-reembed={typeset_result.status} "
                f"columns={typeset_result.placement_count} "
                f"overflow20={typeset_result.overflow_20_count} "
                f"failed={typeset_result.failed_count} "
                f"outside20={typeset_result.hard_limit_outside_changed_pixels} "
                f"invalidated-alignments={len(invalidated)}"
            )
            print(typeset_result.output_path)
            print(typeset_result.audit_path)
            print(updated_state)
            return 0 if typeset_result.status == "typeset" else 2
        if args.translate_only:
            translators, glossary, cache = _translation_resources(args, application)
            translation_result = render_translation(
                application,
                args.output,
                translators=translators,
                glossary=glossary,
                cache=cache,
                max_attempts_per_translator=args.translation_attempts,
            )
            print(
                f"translate-only={translation_result.status} "
                f"units={translation_result.unit_count} "
                f"columns={translation_result.column_count} "
                f"failed={translation_result.failed_unit_count} "
                f"cache-hits={translation_result.cache_hit_count}"
            )
            print(translation_result.output_path)
            return 0 if translation_result.status == "translated" else 2
        image = resolve_image_path(
            application.contract,
            explicit_image=args.image,
            sample_dirs=args.sample_dir,
        )
        if args.typeset_only:
            if args.erased_image is None or args.translation_json is None:
                parser.error(
                    "--typeset-only requires --erased-image and --translation-json"
                )
            typeset_result = render_typesetting(
                image,
                args.erased_image,
                args.translation_json,
                application,
                args.output,
                audit_path=args.audit_json,
                erasure_audit_path=args.erasure_audit,
                config=_layout_config(args),
            )
            print(
                f"typeset-only={typeset_result.status} "
                f"columns={typeset_result.placement_count} "
                f"overflow20={typeset_result.overflow_20_count} "
                f"failed={typeset_result.failed_count} "
                f"outside20={typeset_result.hard_limit_outside_changed_pixels} "
                f"pixel-audit={typeset_result.pixel_audit_seconds:.3f}s"
            )
            print(typeset_result.output_path)
            print(typeset_result.audit_path)
            return 0 if typeset_result.status == "typeset" else 2
        if args.dry_run:
            result = render_dry_run(
                image,
                application,
                args.output,
                audit_path=args.audit_json,
                max_layout_overflow=args.max_layout_overflow,
            )
            print(
                f"dry-run={result.status} columns={result.column_count} "
                f"pending={result.pending_count} bypassed={result.bypassed_count}"
            )
            print(result.output_path)
            print(result.audit_path)
            if result.status == "blocked":
                _print_pending(application)
            return 0
        if not args.erase_only:
            translators, glossary, cache = _translation_resources(args, application)
            full = render_full_pipeline(
                image,
                application,
                args.output,
                translators=translators,
                glossary=glossary,
                cache=cache,
                max_attempts_per_translator=args.translation_attempts,
                state_path=args.state_output,
                stage_directory=args.stage_dir,
                erasure_config=ErasureConfig(
                    style_override=args.erase_strategy.replace("-", "_")
                ),
                layout_config=_layout_config(args),
            )
            print(
                f"full={full.status} columns={full.typesetting.placement_count} "
                f"overflow20={full.typesetting.overflow_20_count} "
                f"failed={full.typesetting.failed_count} "
                f"outside20={full.typesetting.hard_limit_outside_changed_pixels}"
            )
            print(full.output_path)
            print(full.state_path)
            print(full.stage_directory)
            return 0 if full.status == "typeset" else 2
        erase_result = render_erasure(
            image,
            application,
            args.output,
            mask_path=args.mask_output,
            difference_path=args.diff_output,
            audit_path=args.audit_json,
            config=ErasureConfig(
                style_override=args.erase_strategy.replace("-", "_")
            ),
        )
        print(
            f"erase-only={erase_result.status} columns={erase_result.column_count} "
            f"changed={erase_result.changed_pixels} "
            f"outside-mask={erase_result.changed_outside_mask_pixels}"
        )
        print(erase_result.output_path)
        print(erase_result.mask_path)
        print(erase_result.difference_path)
        print(erase_result.audit_path)
        return 0 if erase_result.status != "erasure_failed" else 2
    except MangaLocalizerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _translation_resources(args, application=None):
    glossary = load_glossary(args.glossary)
    if args.translator == "mock":
        mapping = (
            load_mock_translations(args.mock_translations)
            if args.mock_translations
            else None
        )
        expected_ocr_hash = getattr(mapping, "source_ocr_sha256", None)
        if (
            expected_ocr_hash
            and application is not None
            and expected_ocr_hash != application.contract.source_sha256
        ):
            raise TranslationError(
                "source-bound translations do not match the OCR JSON hash"
            )
        translator = MockTranslator(mapping)
    else:
        prompt_style = (
            "contextual"
            if args.translation_prompt_style == "auto"
            else args.translation_prompt_style
        )
        translator = OpenAICompatibleTranslator(
            base_url=args.translation_base_url,
            model=args.translation_model,
            api_key=os.environ.get(args.translation_api_key_env),
            timeout_seconds=args.translation_timeout,
            name=(
                "sakura-openai-compatible"
                if args.translator == "sakura"
                else "openai-compatible"
            ),
            prompt_style=prompt_style,
        )
    translators = (translator,)
    if args.translator == "sakura" and args.translation_prompt_style == "auto":
        translators = (
            translator,
            OpenAICompatibleTranslator(
                base_url=args.translation_base_url,
                model=args.translation_model,
                api_key=os.environ.get(args.translation_api_key_env),
                timeout_seconds=args.translation_timeout,
                name="sakura-openai-compatible-concise",
                prompt_style="concise",
            ),
        )
    cache = (
        None
        if args.no_translation_cache
        else TranslationCache(args.translation_cache_dir)
    )
    return translators, glossary, cache


def _layout_config(args) -> LayoutConfig:
    return LayoutConfig(
        max_overflow_level=args.max_layout_overflow,
        min_font_size=args.min_font_size,
        outline_width=args.outline_width,
        font_path=args.layout_font,
        terminal_punctuation_font_ratio=args.terminal_punctuation_font_ratio,
        preserve_unit_center=args.preserve_unit_center,
    )


def _print_pending(application) -> None:
    print("manual review blocks translation; pending candidates:")
    for item in application.contract.manual_review_queue:
        if item.status != "pending":
            continue
        manga = " / ".join(item.mangaocr_readings) or "<none>"
        paddle = item.paddleocr_text or "<none>"
        reasons = ",".join(item.reason_codes)
        print(
            f"  {item.column_id}: current={item.current_text} | "
            f"MangaOCR={manga} | PaddleOCR={paddle} | reasons={reasons}"
        )


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


if __name__ == "__main__":
    raise SystemExit(main())
