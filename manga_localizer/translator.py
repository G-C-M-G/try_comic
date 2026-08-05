from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import TranslationError, TranslationResponseError
from .ocr_contract import OCRContract


PROMPT_VERSION = "zh-manga-dialogue-v9-preferred-markers"
CACHE_FORMAT_VERSION = 7
TARGET_LANGUAGE = "zh-CN"


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    category: str
    note: str = ""


@dataclass(frozen=True)
class Glossary:
    entries: Tuple[GlossaryEntry, ...]
    notes: Tuple[str, ...]
    sha256: str

    def prompt_text(self, source_text: Optional[str] = None) -> str:
        lines = [
            f"- [{entry.category}] {entry.source} -> {entry.target}"
            + (f" ({entry.note})" if entry.note else "")
            for entry in self.entries
            if source_text is None or entry.source in source_text
        ]
        lines.extend(f"- [note] {note}" for note in self.notes)
        return "\n".join(lines) if lines else "（无指定术语）"


@dataclass(frozen=True)
class TranslationRequest:
    unit_id: str
    column_ids: Tuple[str, ...]
    source_text: str
    translation_query: str
    panel_index: int
    context_before: Tuple[Tuple[str, str], ...]
    context_after: Tuple[Tuple[str, str], ...]
    ocr_effective_sha256: str


@dataclass(frozen=True)
class TranslationResponse:
    text: str
    backend: str
    model: str
    finish_reason: Optional[str] = None
    column_segments: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationAttempt:
    backend: str
    model: str
    attempt: int
    cache_key: str
    cache_hit: bool
    status: str
    error: Optional[str]


@dataclass(frozen=True)
class UnitTranslation:
    unit_id: str
    column_ids: Tuple[str, ...]
    source_text: str
    translation_query: str
    translated_text: Optional[str]
    column_segments: Tuple[str, ...]
    status: str
    backend: Optional[str]
    model: Optional[str]
    cache_key: Optional[str]
    cache_hit: bool
    attempts: Tuple[TranslationAttempt, ...]
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ColumnTranslation:
    column_id: str
    unit_id: str
    unit_column_index: int
    unit_column_count: int
    source_text: str
    translated_text: Optional[str]
    aligned_text_segment: Optional[str]
    status: str


@dataclass(frozen=True)
class TranslationBundle:
    status: str
    prompt_version: str
    target_language: str
    glossary_sha256: str
    units: Tuple[UnitTranslation, ...]
    columns: Tuple[ColumnTranslation, ...]

    @property
    def failed_units(self) -> Tuple[UnitTranslation, ...]:
        return tuple(item for item in self.units if item.status != "translated")

    @property
    def cache_hit_count(self) -> int:
        return sum(item.cache_hit for item in self.units)


class Translator(Protocol):
    name: str
    model: str

    def cache_identity(self) -> Mapping[str, Any]:
        ...

    def translate(
        self,
        request: TranslationRequest,
        glossary: Glossary,
    ) -> TranslationResponse:
        ...


class MockTranslator:
    name = "mock"
    model = "deterministic-mock-v1"

    def __init__(
        self,
        translations: Optional[Mapping[str, str]] = None,
        *,
        fail_unit_ids: Sequence[str] = (),
        prefix: str = "模拟译文：",
    ) -> None:
        self.translations = dict(translations or {})
        self.expected_sources = dict(
            getattr(translations, "expected_sources", {})
        )
        self.source_ocr_sha256 = getattr(translations, "source_ocr_sha256", None)
        self.fail_unit_ids = frozenset(fail_unit_ids)
        self.prefix = prefix
        self.requests: list[TranslationRequest] = []

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "mapping_sha256": _canonical_sha256(self.translations),
            "source_bindings_sha256": _canonical_sha256(self.expected_sources),
            "source_ocr_sha256": self.source_ocr_sha256,
            "fail_unit_ids": sorted(self.fail_unit_ids),
            "prefix": self.prefix,
            "column_marker_policy": "optional",
        }

    def translate(
        self,
        request: TranslationRequest,
        glossary: Glossary,
    ) -> TranslationResponse:
        self.requests.append(request)
        if request.unit_id in self.fail_unit_ids:
            raise TranslationError(f"mock failure for {request.unit_id}")
        expected_source = self.expected_sources.get(request.unit_id)
        if expected_source is not None and expected_source != request.source_text:
            raise TranslationError(
                f"source-bound translation mismatch for {request.unit_id}: "
                f"expected {expected_source!r}, got {request.source_text!r}"
            )
        text = self.translations.get(
            request.unit_id,
            f"{self.prefix}{request.source_text}",
        )
        return TranslationResponse(text=text, backend=self.name, model=self.model)


@dataclass(frozen=True)
class OpenAICompatibleTranslator:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "sukinishiro"
    api_key: Optional[str] = None
    timeout_seconds: float = 90.0
    temperature: float = 0.1
    name: str = "sakura-openai-compatible"
    prompt_style: str = "contextual"
    headers: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.prompt_style not in {"contextual", "concise"}:
            raise TranslationError(
                f"unsupported translation prompt style: {self.prompt_style}"
            )

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "base_url": self.base_url.rstrip("/"),
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
            "prompt_style": self.prompt_style,
            "column_marker_policy": "preferred",
        }

    def translate(
        self,
        request: TranslationRequest,
        glossary: Glossary,
    ) -> TranslationResponse:
        payload = {
            "model": self.model,
            "messages": _translation_messages(
                request,
                glossary,
                prompt_style=self.prompt_style,
            ),
            "temperature": self.temperature,
            "stream": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(dict(self.headers))
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = Request(
            _chat_completions_url(self.base_url),
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                body = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise TranslationError(
                f"{self.name} HTTP {error.code}: {detail or error.reason}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise TranslationError(f"{self.name} request failed: {error}") from error
        if not 200 <= int(status) < 300:
            raise TranslationError(f"{self.name} returned HTTP {status}")
        try:
            data = json.loads(body.decode("utf-8"))
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise TranslationResponseError(
                f"{self.name} returned an invalid chat completion"
            ) from error
        if not isinstance(content, str):
            raise TranslationResponseError(f"{self.name} returned non-text content")
        return TranslationResponse(
            text=_clean_translation(content),
            backend=self.name,
            model=self.model,
            finish_reason=(
                str(choice.get("finish_reason"))
                if choice.get("finish_reason") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CacheSignature:
    key: str
    data: Mapping[str, Any]


class TranslationCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()

    def signature(
        self,
        request: TranslationRequest,
        glossary: Glossary,
        translator: Translator,
    ) -> CacheSignature:
        data: Dict[str, Any] = {
            "format_version": CACHE_FORMAT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "target_language": TARGET_LANGUAGE,
            "translator": dict(translator.cache_identity()),
            "ocr_effective_sha256": request.ocr_effective_sha256,
            "unit_id": request.unit_id,
            "column_ids": list(request.column_ids),
            "source_text": request.source_text,
            "translation_query": request.translation_query,
            "panel_index": request.panel_index,
            "context_before": [list(item) for item in request.context_before],
            "context_after": [list(item) for item in request.context_after],
            "glossary_sha256": glossary.sha256,
        }
        return CacheSignature(key=_canonical_sha256(data), data=data)

    def load(
        self,
        signature: CacheSignature,
        glossary: Glossary,
    ) -> Optional[TranslationResponse]:
        path = self.directory / f"{signature.key}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("format_version") != CACHE_FORMAT_VERSION:
                return None
            if data.get("signature") != signature.data:
                return None
            result = data["result"]
            response = TranslationResponse(
                text=str(result["text"]),
                backend=str(result["backend"]),
                model=str(result["model"]),
                finish_reason=(
                    str(result["finish_reason"])
                    if result.get("finish_reason") is not None
                    else None
                ),
                column_segments=tuple(str(item) for item in result["column_segments"]),
            )
            if len(response.column_segments) not in (
                0,
                len(signature.data["column_ids"]),
            ):
                return None
            if response.column_segments and "".join(response.column_segments) != response.text:
                return None
            _validate_translation(
                response.text,
                signature.data["source_text"],
                glossary,
            )
            return response
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, TranslationError):
            return None

    def store(
        self,
        signature: CacheSignature,
        response: TranslationResponse,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{signature.key}.json"
        data = {
            "format_version": CACHE_FORMAT_VERSION,
            "signature": signature.data,
            "result": {
                "text": response.text,
                "backend": response.backend,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "column_segments": list(response.column_segments),
            },
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def load_glossary(path: Optional[Path]) -> Glossary:
    if path is None:
        return _build_glossary({}, source="empty")
    glossary_path = Path(path).resolve()
    if not glossary_path.is_file():
        raise TranslationError(f"glossary does not exist: {glossary_path}")
    try:
        data = json.loads(glossary_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TranslationError(f"cannot read glossary {glossary_path}: {error}") from error
    if not isinstance(data, dict):
        raise TranslationError("glossary root must be an object")
    return _build_glossary(data, source=str(glossary_path))


def build_translation_requests(contract: OCRContract) -> Tuple[TranslationRequest, ...]:
    units = contract.translation_units
    requests = []
    for index, unit in enumerate(units):
        requests.append(
            TranslationRequest(
                unit_id=unit.id,
                column_ids=unit.column_ids,
                source_text=unit.source_text,
                translation_query=unit.translation_query,
                panel_index=unit.panel_index,
                context_before=tuple(
                    (other.id, other.source_text) for other in units[:index]
                ),
                context_after=tuple(
                    (other.id, other.source_text) for other in units[index + 1 :]
                ),
                ocr_effective_sha256=contract.effective_sha256,
            )
        )
    return tuple(requests)


def translate_contract(
    contract: OCRContract,
    translators: Sequence[Translator],
    *,
    glossary: Optional[Glossary] = None,
    cache: Optional[TranslationCache] = None,
    max_attempts_per_translator: int = 2,
) -> TranslationBundle:
    if not translators:
        raise TranslationError("at least one translator is required")
    if max_attempts_per_translator < 1:
        raise TranslationError("max_attempts_per_translator must be positive")
    active_glossary = glossary or _build_glossary({}, source="empty")
    unit_results = []
    for request in build_translation_requests(contract):
        unit_results.append(
            _translate_request(
                request,
                translators,
                active_glossary,
                cache,
                max_attempts_per_translator,
            )
        )

    by_unit = {item.unit_id: item for item in unit_results}
    columns = []
    for column_id in contract.reading_order:
        unit, index = contract.units_by_column_id[column_id]
        result = by_unit[unit.id]
        columns.append(
            ColumnTranslation(
                column_id=column_id,
                unit_id=unit.id,
                unit_column_index=index,
                unit_column_count=len(unit.column_ids),
                source_text=contract.columns_by_id[column_id].text,
                translated_text=result.translated_text,
                aligned_text_segment=(
                    result.column_segments[index]
                    if len(result.column_segments) == len(unit.column_ids)
                    else None
                ),
                status=result.status,
            )
        )
    status = (
        "translated"
        if all(item.status == "translated" for item in unit_results)
        else "translation_failed"
    )
    return TranslationBundle(
        status=status,
        prompt_version=PROMPT_VERSION,
        target_language=TARGET_LANGUAGE,
        glossary_sha256=active_glossary.sha256,
        units=tuple(unit_results),
        columns=tuple(columns),
    )


class SourceBoundTranslations(dict[str, str]):
    def __init__(
        self,
        translations: Mapping[str, str],
        expected_sources: Mapping[str, str],
        source_ocr_sha256: Optional[str],
    ) -> None:
        super().__init__(translations)
        self.expected_sources = dict(expected_sources)
        self.source_ocr_sha256 = source_ocr_sha256


def load_mock_translations(path: Path) -> Mapping[str, str]:
    translation_path = Path(path).resolve()
    try:
        data = json.loads(translation_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TranslationError(
            f"cannot read mock translations {translation_path}: {error}"
        ) from error
    if isinstance(data, dict) and data.get("artifact") == "source-bound-translations":
        units = data.get("units")
        if not isinstance(units, dict) or not units:
            raise TranslationError("source-bound translations require a non-empty units mapping")
        translations = {}
        expected_sources = {}
        for unit_id, item in units.items():
            if not isinstance(unit_id, str) or not isinstance(item, dict):
                raise TranslationError("invalid source-bound translation unit")
            source_text = item.get("source_text")
            translated_text = item.get("translated_text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise TranslationError(f"source-bound translation {unit_id} lacks source_text")
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise TranslationError(f"source-bound translation {unit_id} lacks translated_text")
            expected_sources[unit_id] = source_text
            translations[unit_id] = translated_text
        source_ocr_sha256 = data.get("source_ocr_sha256")
        if source_ocr_sha256 is not None and (
            not isinstance(source_ocr_sha256, str)
            or len(source_ocr_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in source_ocr_sha256)
        ):
            raise TranslationError("source_ocr_sha256 must be a 64-character hex digest")
        return SourceBoundTranslations(
            translations,
            expected_sources,
            source_ocr_sha256.lower() if source_ocr_sha256 else None,
        )
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in data.items()
    ):
        raise TranslationError("mock translations must be a non-empty string mapping")
    return dict(data)


def _translate_request(
    request: TranslationRequest,
    translators: Sequence[Translator],
    glossary: Glossary,
    cache: Optional[TranslationCache],
    max_attempts: int,
) -> UnitTranslation:
    attempts = []
    last_error: Optional[str] = None
    for translator in translators:
        signature = (
            cache.signature(request, glossary, translator)
            if cache is not None
            else _uncached_signature(request, glossary, translator)
        )
        cached = cache.load(signature, glossary) if cache is not None else None
        if cached is not None:
            attempts.append(
                TranslationAttempt(
                    backend=cached.backend,
                    model=cached.model,
                    attempt=0,
                    cache_key=signature.key,
                    cache_hit=True,
                    status="translated",
                    error=None,
                )
            )
            return _successful_unit(request, cached, signature.key, True, attempts)
        for attempt_number in range(1, max_attempts + 1):
            try:
                response = translator.translate(request, glossary)
                cleaned = _clean_translation(response.text)
                marker_policy = str(
                    translator.cache_identity().get("column_marker_policy", "required")
                )
                complete_text, column_segments = _normalize_column_segments(
                    cleaned,
                    request,
                    require_markers=marker_policy == "required",
                    prefer_markers=marker_policy == "preferred",
                )
                complete_text = _preserve_terminal_ellipsis(
                    complete_text,
                    request.source_text,
                )
                if column_segments and "".join(column_segments) != complete_text:
                    column_segments = column_segments[:-1] + (
                        _preserve_terminal_ellipsis(
                            column_segments[-1],
                            request.source_text,
                        ),
                    )
                _validate_translation(complete_text, request.source_text, glossary)
                response = TranslationResponse(
                    text=complete_text,
                    backend=response.backend,
                    model=response.model,
                    finish_reason=response.finish_reason,
                    column_segments=column_segments,
                )
                if cache is not None:
                    cache.store(signature, response)
                attempts.append(
                    TranslationAttempt(
                        backend=response.backend,
                        model=response.model,
                        attempt=attempt_number,
                        cache_key=signature.key,
                        cache_hit=False,
                        status="translated",
                        error=None,
                    )
                )
                return _successful_unit(
                    request,
                    response,
                    signature.key,
                    False,
                    attempts,
                )
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                attempts.append(
                    TranslationAttempt(
                        backend=translator.name,
                        model=translator.model,
                        attempt=attempt_number,
                        cache_key=signature.key,
                        cache_hit=False,
                        status="translation_failed",
                        error=last_error,
                    )
                )
    return UnitTranslation(
        unit_id=request.unit_id,
        column_ids=request.column_ids,
        source_text=request.source_text,
        translation_query=request.translation_query,
        translated_text=None,
        column_segments=(),
        status="translation_failed",
        backend=None,
        model=None,
        cache_key=None,
        cache_hit=False,
        attempts=tuple(attempts),
        failure_reason=last_error or "no translator produced a result",
    )


def _successful_unit(
    request: TranslationRequest,
    response: TranslationResponse,
    cache_key: str,
    cache_hit: bool,
    attempts: Sequence[TranslationAttempt],
) -> UnitTranslation:
    return UnitTranslation(
        unit_id=request.unit_id,
        column_ids=request.column_ids,
        source_text=request.source_text,
        translation_query=request.translation_query,
        translated_text=response.text,
        column_segments=response.column_segments,
        status="translated",
        backend=response.backend,
        model=response.model,
        cache_key=cache_key,
        cache_hit=cache_hit,
        attempts=tuple(attempts),
        failure_reason=None,
    )


def _uncached_signature(
    request: TranslationRequest,
    glossary: Glossary,
    translator: Translator,
) -> CacheSignature:
    transient = TranslationCache(Path("."))
    return transient.signature(request, glossary, translator)


def _build_glossary(data: Mapping[str, Any], *, source: str) -> Glossary:
    unknown = set(data) - {"version", "terms", "names", "notes"}
    if unknown:
        raise TranslationError(f"unsupported glossary fields in {source}: {sorted(unknown)}")
    entries = []
    for category in ("terms", "names"):
        values = data.get(category, {})
        if not isinstance(values, dict):
            raise TranslationError(f"glossary.{category} must be an object")
        for original, translated in sorted(values.items()):
            note = ""
            target = translated
            if isinstance(translated, dict):
                target = translated.get("translation")
                note = str(translated.get("note") or "")
            if not isinstance(original, str) or not original.strip():
                raise TranslationError(f"glossary.{category} contains an empty source")
            if not isinstance(target, str) or not target.strip():
                raise TranslationError(
                    f"glossary.{category}.{original} must contain a translation"
                )
            entries.append(
                GlossaryEntry(
                    source=original.strip(),
                    target=target.strip(),
                    category=category[:-1] if category.endswith("s") else category,
                    note=note.strip(),
                )
            )
    raw_notes = data.get("notes", [])
    if not isinstance(raw_notes, list) or not all(
        isinstance(note, str) and note.strip() for note in raw_notes
    ):
        raise TranslationError("glossary.notes must be a list of non-empty strings")
    normalized = {
        "version": int(data.get("version", 1)),
        "entries": [
            {
                "source": item.source,
                "target": item.target,
                "category": item.category,
                "note": item.note,
            }
            for item in entries
        ],
        "notes": [note.strip() for note in raw_notes],
    }
    return Glossary(
        entries=tuple(entries),
        notes=tuple(note.strip() for note in raw_notes),
        sha256=_canonical_sha256(normalized),
    )


def _translation_messages(
    request: TranslationRequest,
    glossary: Glossary,
    *,
    prompt_style: str,
) -> list[Dict[str, str]]:
    if prompt_style == "concise":
        return [
            {
                "role": "system",
                "content": (
                    "你是日文漫画对白翻译器。请意译成简洁、自然、符合中文口语习惯的简体中文。"
                    "不要逐字硬译，不要添加原文没有的人称或解释，并尽量比原文简短。"
                    "用户提供的是一条带<1>、<2>等列标记的完整台词。必须原样保留全部标记并按原顺序"
                    "放在对应中文片段前；这些片段共同组成一句译文，不得逐列孤立直译。"
                    "按中文口语补充必要标点，标点放在前一个语义片段末尾。"
                    "只返回一行带标记的译文，不要解释。"
                    "必须遵守以下术语和口语规则：\n"
                    + glossary.prompt_text(request.source_text)
                ),
            },
            {
                "role": "user",
                "content": "将下面的完整日文漫画对白翻译成中文：\n" + request.translation_query,
            },
        ]
    before = "\n".join(
        f"{unit_id}: {text}" for unit_id, text in request.context_before
    ) or "（无）"
    after = "\n".join(
        f"{unit_id}: {text}" for unit_id, text in request.context_after
    ) or "（无）"
    system = (
        "你是日文漫画翻译。用户消息是一条带<1>、<2>等列标记的完整日文台词。"
        "把它翻成自然、简洁、口语化的简体中文，保持人物语气和情绪；"
        "结合下列本页上下文理解指代，但不要翻译上下文，也不要添加原文没有的信息。"
        "不得把竖排列拆开逐列直译。必须原样保留全部列标记并按原顺序放在对应中文片段前，"
        "各片段合起来必须是一句自然译文。只返回一行带标记的译文，不要解释、序号、引号或提示字段。\n\n"
        "按中文口语补充必要标点，标点放在前一个语义片段末尾。\n\n"
        f"面板索引：{request.panel_index}\n"
        f"本页前文：\n{before}\n\n"
        f"本页后文：\n{after}\n\n"
        f"术语表/人名表：\n{glossary.prompt_text(request.source_text)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": request.translation_query},
    ]


def _clean_translation(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
        if cleaned.startswith(("text\n", "json\n")):
            cleaned = cleaned.split("\n", 1)[1].strip()
    cleaned = re.sub(r"^(?:译文|翻译|中文译文)\s*[:：]\s*", "", cleaned)
    if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"(?:\.{2,}|．{2,}|…+)", "……", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_column_segments(
    translated: str,
    request: TranslationRequest,
    *,
    require_markers: bool,
    prefer_markers: bool = False,
) -> Tuple[str, Tuple[str, ...]]:
    matches = list(re.finditer(r"<\s*(\d+)\s*>", translated))
    if matches:
        if translated[: matches[0].start()].strip():
            raise TranslationResponseError("translator wrote text before the first column marker")
        numbers = [int(match.group(1)) for match in matches]
        expected = list(range(1, len(request.column_ids) + 1))
        if numbers != expected:
            if prefer_markers and numbers == list(range(1, len(numbers) + 1)):
                unaligned = re.sub(r"<\s*\d+\s*>", "", translated).strip()
                if unaligned:
                    return unaligned, ()
            raise TranslationResponseError(
                f"translator returned column markers {numbers}, expected {expected}"
            )
        segments = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(translated)
            segment = translated[match.end() : end].strip()
            if not segment:
                raise TranslationResponseError(
                    f"translator returned an empty column segment <{index + 1}>"
                )
            segments.append(segment)
        return "".join(segments), tuple(segments)
    if require_markers:
        raise TranslationResponseError("translator omitted required OCR column markers")
    if prefer_markers:
        return translated, ()
    segments = _proportional_segments(translated, request.translation_query)
    return translated, segments


def _proportional_segments(text: str, translation_query: str) -> Tuple[str, ...]:
    source_matches = list(re.finditer(r"<\s*(\d+)\s*>", translation_query))
    count = len(source_matches)
    chars = list(text)
    if count < 1 or len(chars) < count:
        return ()
    source_segments = []
    for index, match in enumerate(source_matches):
        end = (
            source_matches[index + 1].start()
            if index + 1 < count
            else len(translation_query)
        )
        source_segments.append(translation_query[match.end() : end])
    allocations = [1] * count
    remaining = len(chars) - count
    weights = [max(1, len(value)) for value in source_segments]
    while remaining:
        selected = max(
            range(count),
            key=lambda index: (weights[index] / allocations[index], -index),
        )
        allocations[selected] += 1
        remaining -= 1
    result = []
    offset = 0
    for size in allocations:
        result.append("".join(chars[offset : offset + size]))
        offset += size
    return tuple(result)


def _validate_translation(
    translated: str,
    source: str,
    glossary: Optional[Glossary] = None,
) -> None:
    if not translated.strip():
        raise TranslationResponseError("translator returned empty text")
    if translated.strip() == source.strip():
        raise TranslationResponseError("translator returned the unchanged source text")
    if re.search(r"<\d+>", translated):
        raise TranslationResponseError("translator leaked OCR column markers")
    if re.search(r"[?？]", source) and not re.search(r"[?？]", translated):
        raise TranslationResponseError("translator dropped the source question mark")
    if re.search(r"[!！]", source) and not re.search(r"[!！]", translated):
        raise TranslationResponseError("translator dropped the source exclamation mark")
    if re.search(
        r"本页前文|本页后文|术语表|当前完整|当前原文|原列标记|列完整性|面板索引|无指定术语|输出当前",
        translated,
    ):
        raise TranslationResponseError("translator leaked prompt fields")
    if len(translated) > max(24, round(len(source) * 2.5)):
        raise TranslationResponseError("translator expanded the dialogue implausibly")
    kana_count = len(re.findall(r"[\u3040-\u30ff]", translated))
    if kana_count > max(2, round(len(translated) * 0.15)):
        raise TranslationResponseError("translator left excessive Japanese kana")
    if glossary is not None:
        leaked = [
            entry.target
            for entry in glossary.entries
            if entry.source not in source
            and len(entry.target) >= 5
            and entry.target in translated
        ]
        if leaked:
            raise TranslationResponseError(
                f"translator leaked unrelated glossary targets: {leaked}"
            )


def _preserve_terminal_ellipsis(translated: str, source: str) -> str:
    source_has_ellipsis = bool(re.search(r"(?:\.{2,}|．{2,}|…+)$", source.strip()))
    if source_has_ellipsis and not translated.endswith("……"):
        return translated.rstrip("。.!！?？") + "……"
    return translated


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _canonical_sha256(data: Any) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
