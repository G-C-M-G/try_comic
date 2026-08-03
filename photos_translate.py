"""Translate and typeset OCR captured by Windows Photos.

Pipeline: Photos column OCR -> panel/sentence grouping -> Sakura translation ->
per-column geometry -> text cleanup or white-outline overlay -> rendered image.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MITS_DIR = os.path.join(ROOT, "manga-image-translator")
sys.path.insert(0, MITS_DIR)
sys.path.insert(0, ROOT)
os.environ["SAKURA_API_BASE"] = os.environ.get("SAKURA_API_BASE", "http://127.0.0.1:8080/v1")
os.environ["SAKURA_VERSION"] = os.environ.get("SAKURA_VERSION", "0.10")

from manga_translator.config import Inpainter, InpainterConfig
from manga_translator.inpainting import dispatch as dispatch_inpainting
from manga_translator.utils import TextBlock
from manga_translator.utils.sort import sort_regions

from photos_capture import capture_photos_columns
from text_paint import render_text_image
from text_segments import group_regions_into_sentences, split_translation_for_segments


_OVERRIDES = {
    "やられた！！": "中招了！",
    "やられた!！": "中招了！",
}


def _quad(bbox):
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)


def _make_regions(manifest: dict, image: np.ndarray) -> list[TextBlock]:
    regions = []
    for column in manifest["columns"]:
        x1, y1, x2, y2 = column["image_bbox"]
        font_size = max(12, int((x2 - x1) * 0.88))
        region = TextBlock(
            lines=_quad(column["image_bbox"]),
            texts=[column["text"]],
            font_size=font_size,
            direction="v" if column.get("direction") == "vertical" else "h",
            source_lang="JPN",
            target_lang="CHS",
            fg_color=(0, 0, 0),
            bg_color=(255, 255, 255),
        )
        region.column_id = column["id"]
        region.ocr_backend = manifest["backend"]
        regions.append(region)
    return sort_regions(regions, right_to_left=True, img=image)


async def _translate_queries(queries: list[str]) -> list[str]:
    """Use local Sakura with a marker-preserving, conversational manga prompt."""
    pending = [query for query in queries if query not in _OVERRIDES]
    translated_pending = await asyncio.to_thread(_request_comic_translation, pending) if pending else []
    iterator = iter(translated_pending)
    return [_OVERRIDES[query] if query in _OVERRIDES else next(iterator) for query in queries]


def _request_comic_translation(queries: list[str]) -> list[str]:
    system = (
        "你是日文漫画对白翻译器。请意译成简洁、自然、符合中文口语习惯的简体中文。"
        "不要逐字硬译，不要添加原文没有的人称或解释，避免使用生硬的抽象名词。"
        "译文应像人物当场说出的话，并尽量比原文简短。"
        "必须原样保留输入中的<1>、<2>等位置标记，并把标记放在对应译文片段之前。"
        "不要解释。输入有多行时每行输出一行。"
    )

    def request(batch: list[str]) -> list[str]:
        body = {
            "model": "sukinishiro",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "将下面的日文漫画对白翻译成中文：\n" + "\n".join(batch)},
            ],
            "temperature": 0.1,
            "top_p": 0.3,
            "max_tokens": max(512, sum(len(value) for value in batch) * 3),
            "extra_query": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0},
        }
        api_base = os.environ["SAKURA_API_BASE"].rstrip("/")
        endpoint = api_base + "/chat/completions"
        http_request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(http_request, timeout=180) as response:
            payload = json.loads(response.read())
        content = payload["choices"][0]["message"]["content"]
        return [line.strip().strip("「」") for line in content.splitlines() if line.strip()]

    results = request(queries)
    if len(results) != len(queries):
        results = [request([query])[0] for query in queries]
    for index, (query, translation) in enumerate(zip(queries, results)):
        expected_markers = re.findall(r"<\s*\d+\s*>", query)
        if any(marker not in translation for marker in expected_markers):
            retry = request([query])
            if retry:
                results[index] = retry[0]
    return results


def _is_overlay_text(image: np.ndarray, region: TextBlock) -> bool:
    """Text over artwork is redrawn with white outline instead of inpainting."""
    x1, y1, x2, y2 = region.xyxy
    pad = max(10, int(region.font_size * 1.3))
    crop = image[max(0, y1 - pad):min(image.shape[0], y2 + pad), max(0, x1 - pad):min(image.shape[1], x2 + pad)]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    light_ratio = float(np.mean(gray > 210))
    # A speech bubble provides a broad white field; SFX/text drawn over artwork
    # has substantially less white around the narrow OCR column.
    return light_ratio < 0.42


def _segment_bbox(segment: dict) -> tuple[int, int, int, int]:
    points = np.asarray(segment["lines"], dtype=np.int32).reshape(-1, 2)
    return int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max())


def _cluster_segments(segments: list[dict]) -> list[list[dict]]:
    if len(segments) < 2:
        return [segments]
    ordered = sorted(segments, key=lambda segment: -((_segment_bbox(segment)[0] + _segment_bbox(segment)[2]) / 2))
    widths = [_segment_bbox(segment)[2] - _segment_bbox(segment)[0] for segment in ordered]
    max_gap = max(24.0, float(np.median(widths)) * 2.4)
    clusters = [[ordered[0]]]
    for segment in ordered[1:]:
        previous = clusters[-1][-1]
        prev_box, box = _segment_bbox(previous), _segment_bbox(segment)
        gap = max(0, prev_box[0] - box[2])
        if gap > max_gap:
            clusters.append([segment])
        else:
            clusters[-1].append(segment)
    return clusters


def _translation_query(sentence: TextBlock) -> tuple[str, list[list[dict]]]:
    segments = list(getattr(sentence, "render_segments", None) or [])
    if not segments:
        segments = [{
            "text": sentence.text,
            "lines": np.asarray(sentence.lines).reshape(-1, 4, 2).tolist(),
            "font_size": sentence.font_size,
            "direction": sentence.direction,
        }]
    clusters = _cluster_segments(segments)
    sources = ["".join(segment["text"] for segment in cluster) for cluster in clusters]
    query = sources[0] if len(sources) == 1 else "".join(f"<{index}>{text}" for index, text in enumerate(sources, 1))
    return query, clusters


def _render_regions(grouped: list[TextBlock], translations: list[str], image: np.ndarray) -> list[TextBlock]:
    output = []
    for sentence, translation in zip(grouped, translations):
        query, clusters = _translation_query(sentence)
        cluster_sources = ["".join(segment["text"] for segment in cluster) for cluster in clusters]
        parts = split_translation_for_segments(translation, cluster_sources)
        if not parts:
            clean = re.sub(r"<\s*\d+\s*>", "", translation).strip()
            parts = [clean] + [""] * (len(clusters) - 1)
        sentence.translation = "".join(parts)
        for cluster, translated in zip(clusters, parts):
            lines = np.concatenate([
                np.asarray(segment["lines"], dtype=np.int32).reshape(-1, 4, 2)
                for segment in cluster
            ])
            font_sizes = [int(segment.get("font_size") or sentence.font_size) for segment in cluster]
            region = TextBlock(
                lines=lines,
                texts=["".join(segment["text"] for segment in cluster)],
                translation=translated,
                font_size=max(12, int(np.median(font_sizes))),
                direction="v",
                source_lang="JPN",
                target_lang="CHS",
                fg_color=(0, 0, 0),
                bg_color=(255, 255, 255),
            )
            region.ocr_backend = "Windows Photos Scan Text"
            region.overlay_text = len(clusters) == 1 and len(cluster) == 1 and _is_overlay_text(image, region)
            output.append(region)
    return output


def _text_mask(image: np.ndarray, regions: list[TextBlock]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for region in regions:
        if region.overlay_text:
            continue
        for polygon in np.asarray(region.lines, dtype=np.int32).reshape(-1, 4, 2):
            cv2.fillPoly(mask, [polygon], 255)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _render(image: np.ndarray, regions: list[TextBlock]) -> Image.Image:
    canvas = Image.fromarray(image).convert("RGB")
    for region in regions:
        text = (region.translation or "").strip()
        if not text:
            continue
        x1, y1, x2, y2 = (int(value) for value in region.xyxy)
        width, height = max(8, x2 - x1), max(8, y2 - y1)
        overlay = bool(region.overlay_text)
        # Chinese is denser than Japanese; keep one compact vertical column and
        # allow an outlined SFX translation to be slightly larger than the box.
        if overlay:
            width = max(width + region.font_size, int(region.font_size * 2.2))
            # Preserve the source column height.  A short Chinese translation
            # such as "中招了！" must cover the full Japanese SFX column.
            desired_step = max(region.font_size, int(height / max(1, len(text))))
            height = max(height, len(text) * desired_step)
        font_size = max(14, min(int(region.font_size), max(14, int(width * 0.72))))
        if not overlay:
            while font_size > 14:
                line_height = font_size * 1.08
                chars_per_column = max(1, int(height / line_height))
                column_count = int(np.ceil(len(text) / chars_per_column))
                if column_count * font_size * 1.2 <= width:
                    break
                font_size -= 1
        line_spacing = (height / max(1, len(text)) / font_size) if overlay else 1.06
        if overlay:
            line_spacing = max(1.0, line_spacing)
        rendered = render_text_image(
            text,
            font_size,
            vertical=True,
            max_w=width,
            max_h=height,
            fg=(0, 0, 0),
            bg=(255, 255, 255),
            bold=overlay,
            stroke_width=max(2, int(font_size * (0.24 if overlay else 0.07))),
            line_spacing=line_spacing,
            char_spacing=0,
        )
        if rendered is None:
            continue
        cx, cy = region.center
        canvas.paste(rendered, (int(cx - rendered.width / 2), int(cy - rendered.height / 2)), rendered)
    return canvas


async def _inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return image.copy()
    config = InpainterConfig(inpainter=Inpainter.lama_large, inpainting_size=2048)
    return await dispatch_inpainting(Inpainter.lama_large, image, mask, config, 2048, "cpu", False)


async def translate_manifest(manifest: dict, output: str, state_output: str | None = None) -> dict:
    image = np.array(Image.open(manifest["source"]["path"]).convert("RGB"))
    ordered = _make_regions(manifest, image)
    grouped = group_regions_into_sentences(ordered)
    queries = [_translation_query(region)[0] for region in grouped]
    translations = await _translate_queries(queries)
    render_regions = _render_regions(grouped, translations, image)
    mask = _text_mask(image, render_regions)
    cleaned = await _inpaint(image, mask)
    rendered = _render(cleaned, render_regions)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    rendered.save(output)

    result = dict(manifest)
    result["sentences"] = [
        {
            "source": region.text,
            "translation": region.translation,
            "panel_index": getattr(region, "panel_index", None),
        }
        for region in grouped
    ]
    result["render_regions"] = [
        {
            "source": region.text,
            "translation": region.translation,
            "image_bbox": [int(value) for value in region.xyxy],
            "mode": "white-outline-overlay" if region.overlay_text else "inpaint-and-typeset",
        }
        for region in render_regions
    ]
    result["output"] = os.path.abspath(output)
    if state_output:
        os.makedirs(os.path.dirname(os.path.abspath(state_output)), exist_ok=True)
        with open(state_output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic Windows Photos OCR manga translator")
    parser.add_argument("image")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--manifest")
    parser.add_argument("--state")
    parser.add_argument("--debug-dir")
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or str(
        Path(args.image).with_name(Path(args.image).stem + "_photos_translated" + Path(args.image).suffix)
    )

    manifest_path = args.manifest or str(Path(output).with_suffix(".photos.json"))
    if args.reuse_manifest and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = capture_photos_columns(args.image, debug_dir=args.debug_dir)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if manifest.get("backend") != "Windows Photos Scan Text":
        raise RuntimeError("refusing to translate a non-Photos OCR manifest")
    asyncio.run(translate_manifest(manifest, output, args.state or str(Path(output).with_suffix(".state.json"))))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
