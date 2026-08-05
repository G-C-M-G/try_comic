"""Automatic REVIEW resolution when the review count is small.

When a page has fewer than ``review_threshold`` (default 10) review islands,
the pipeline automatically tries to resolve them before the GUI step:

1. AI vision re-read (primary, no window): send each island crop (original
   image bbox) to an OpenAI-compatible vision model and ask for the exact
   Japanese transcription.  If the AI reading agrees with the manga reading
   (normalized edit distance <= threshold) the island passes; otherwise it
   stays in review with the AI text recorded as extra evidence.
2. Photos recapture (optional, foreground): invokes the upstream
   ``photos_capture.py recapture`` path for the affected pages; the updated
   capture columns are folded back as evidence.

Both paths are best-effort: failures are recorded and never block.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .cross_validate import (
    DEFAULT_EDIT_DISTANCE_THRESHOLD,
    _normalized_edit_distance,
    _normalize_for_match,
)
from .errors import TranslationError
from .island_contract import IslandDataset

DEFAULT_REVIEW_THRESHOLD = 10

_AI_READ_PROMPT = (
    "你是日文漫画文字识别助手。请精确转录这张图片中的全部日文文字，"
    "保持原文顺序与标点（包括『』「」……！？等），只返回转录文本本身，不要解释、不要翻译。"
    "如果图片中没有清晰可读的文字，返回空。"
)


@dataclass(frozen=True)
class ResolveConfig:
    review_threshold: int = DEFAULT_REVIEW_THRESHOLD
    online_base_url: str = ""
    online_model: str = ""
    online_api_key: Optional[str] = None
    timeout_seconds: float = 90.0
    edit_distance_threshold: float = DEFAULT_EDIT_DISTANCE_THRESHOLD


@dataclass(frozen=True)
class IslandResolveResult:
    island_id: str
    resolved: bool
    ai_text: Optional[str]
    previous_status: str
    new_status: str
    note: str


@dataclass(frozen=True)
class PageResolveReport:
    page: str
    triggered: bool
    review_before: int
    review_after: int
    results: Tuple[IslandResolveResult, ...]
    errors: Tuple[str, ...]


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ai_vision_read(
    crop: Image.Image,
    *,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    timeout_seconds: float = 90.0,
) -> str:
    """Ask a vision-capable OpenAI-compatible model to transcribe the crop."""
    if not base_url or not model:
        raise TranslationError("AI vision resolve requires online_base_url and online_model")
    normalized = base_url.rstrip("/")
    url = normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _AI_READ_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_encode_image(crop)}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise TranslationError(f"AI vision HTTP {error.code}: {detail or error.reason}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise TranslationError(f"AI vision request failed: {error}") from error
    if not 200 <= int(status) < 300:
        raise TranslationError(f"AI vision returned HTTP {status}")
    try:
        data = json.loads(body.decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise TranslationError("AI vision returned an invalid chat completion") from error
    text = str(content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


def resolve_review_islands(
    dataset: IslandDataset,
    review_verdicts: Sequence[Mapping[str, object]],
    *,
    crop_source: Image.Image,
    config: ResolveConfig,
    ai_reader=None,
) -> PageResolveReport:
    """Auto-resolve REVIEW islands when their count is below the threshold.

    ``ai_reader`` is injectable for tests: ``(crop, **kwargs) -> str``.
    """
    page = dataset.page
    review_ids = [
        str(verdict["island_id"])
        for verdict in review_verdicts
        if verdict.get("status") == "review"
    ]
    triggered = 0 < len(review_ids) < config.review_threshold
    if not triggered:
        return PageResolveReport(
            page=page,
            triggered=False,
            review_before=len(review_ids),
            review_after=len(review_ids),
            results=(),
            errors=(),
        )
    manga_by_id = {
        str(verdict["island_id"]): str(verdict.get("manga_text") or "")
        for verdict in review_verdicts
    }
    reader = ai_reader or (
        lambda crop, **kwargs: ai_vision_read(
            crop,
            base_url=config.online_base_url,
            model=config.online_model,
            api_key=config.online_api_key,
            timeout_seconds=config.timeout_seconds,
        )
    )
    results = []
    errors = []
    review_after = 0
    for verdict in review_verdicts:
        island_id = str(verdict["island_id"])
        if verdict.get("status") != "review":
            continue
        island = dataset.islands_by_id[island_id]
        crop = crop_source.crop(island.image_bbox)
        try:
            ai_text = str(reader(crop) or "").strip()
        except TranslationError as error:
            errors.append(f"{island_id}: {error}")
            ai_text = ""
        if not ai_text:
            results.append(
                IslandResolveResult(
                    island_id=island_id,
                    resolved=False,
                    ai_text=None,
                    previous_status="review",
                    new_status="review",
                    note="AI vision returned empty",
                )
            )
            review_after += 1
            continue
        manga_text = manga_by_id.get(island_id, "")
        distance = _normalized_edit_distance(
            _normalize_for_match(manga_text),
            _normalize_for_match(ai_text),
        )
        if manga_text and distance <= config.edit_distance_threshold:
            results.append(
                IslandResolveResult(
                    island_id=island_id,
                    resolved=True,
                    ai_text=ai_text,
                    previous_status="review",
                    new_status="pass",
                    note=f"AI vision agrees with manga (edit={distance:.2f})",
                )
            )
        else:
            results.append(
                IslandResolveResult(
                    island_id=island_id,
                    resolved=False,
                    ai_text=ai_text,
                    previous_status="review",
                    new_status="review",
                    note=f"AI vision diverges from manga (edit={distance:.2f})",
                )
            )
            review_after += 1
    return PageResolveReport(
        page=page,
        triggered=True,
        review_before=len(review_ids),
        review_after=review_after,
        results=tuple(results),
        errors=tuple(errors),
    )


def render_resolve_markdown(report: PageResolveReport) -> str:
    lines = [f"# {report.page} 页 自动二次识别（REVIEW<{DEFAULT_REVIEW_THRESHOLD} 触发）", ""]
    if not report.triggered:
        lines.append(
            f"- 未触发：REVIEW 数 {report.review_before}（阈值 {report.review_threshold}，需 0 < n < 阈值）"
        )
        return "\n".join(lines)
    lines.append(
        f"- 触发：REVIEW {report.review_before} -> 剩余 {report.review_after}（AI 视觉识别，无窗口）"
    )
    lines.append("")
    lines.append("| 岛 | 结果 | AI 识别文本 | 说明 |")
    lines.append("|---|---|---|---|")
    for item in report.results:
        mark = "✅ 转 pass" if item.resolved else "⚠️ 仍 REVIEW"
        lines.append(
            f"| {item.island_id} | {mark} | {item.ai_text or '—'} | {item.note} |"
        )
    for error in report.errors:
        lines.append(f"- 错误：{error}")
    lines.append("")
    return "\n".join(lines)
