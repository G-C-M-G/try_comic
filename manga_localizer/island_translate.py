"""Island-level translation with a precision switch.

``precision=local`` (default) sends the whole island text to the local
Sakura endpoint; ``precision=online`` sends it to a configurable
OpenAI-compatible service for higher-quality colloquial translation of
awkward words (curses, chants, onomatopoeia).  OCR architecture is identical
in both modes.  Cache signatures bind the precision switch, model and prompt
version; an online failure silently falls back to local and is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import (
    TranslationError,
    TranslationResponseError,
    TranslationTimeoutError,
)
from .island_contract import IslandDataset
from .ocr_contract import BBox, Polygon
from .text_rules import is_numeric_passthrough
from .translator import (
    Glossary,
    PROMPT_VERSION,
    TranslationCache,
    TranslationRequest,
    TranslationResponse,
    Translator,
    load_glossary,
)

_LOG = logging.getLogger("manga_localizer.island_translate")

ISLAND_PROMPT_VERSION = "island-zh-dialogue-v2-whole-island-length"

PRECISION_LOCAL = "local"
PRECISION_ONLINE = "online"
_PRECISIONS = (PRECISION_LOCAL, PRECISION_ONLINE)


@dataclass(frozen=True)
class IslandTranslateConfig:
    precision: str = PRECISION_LOCAL
    sakura_base_url: str = "http://127.0.0.1:8080/v1"
    sakura_model: str = "sukinishiro"
    online_base_url: str = ""
    online_model: str = ""
    online_api_key: Optional[str] = None
    glossary_path: Optional[str] = None
    timeout_seconds: float = 90.0
    temperature: float = 0.1


@dataclass(frozen=True)
class IslandUnitTranslation:
    island_id: str
    source_text: str
    translated_text: Optional[str]
    status: str
    backend: Optional[str]
    model: Optional[str]
    cache_key: Optional[str]
    cache_hit: bool
    attempts: Tuple[Tuple[str, str], ...]  # (backend, status)
    failure_reason: Optional[str] = None
    kind: str = "text"


@dataclass(frozen=True)
class IslandTranslationResult:
    page: str
    precision: str
    units: Tuple[IslandUnitTranslation, ...]
    failed_count: int
    cache_hit_count: int

    @property
    def translations(self) -> Mapping[str, str]:
        return {
            unit.island_id: unit.translated_text
            for unit in self.units
            if unit.translated_text is not None
        }


class IslandOpenAICompatibleTranslator:
    """Whole-island translation through any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 90.0,
        temperature: float = 0.1,
        prompt_style: str = "contextual",
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.prompt_style = prompt_style
        self.messages_log: list[list[dict]] = []

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "island_prompt_version": ISLAND_PROMPT_VERSION,
            "prompt_style": self.prompt_style,
            "column_marker_policy": "none",
        }

    def translate(
        self,
        request: TranslationRequest,
        glossary: Glossary,
    ) -> TranslationResponse:
        messages = _island_messages(request, glossary, concise=self.prompt_style == "concise")
        self.messages_log.append(messages)
        return TranslationResponse(
            text=_chat_completion(
                self.base_url,
                self.model,
                self.api_key,
                messages,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            ),
            backend=self.name,
            model=self.model,
            finish_reason=None,
        )


class MockIslandTranslator:
    """Deterministic offline translator for tests and CLI demos."""

    name = "mock-island"
    model = "deterministic-mock-v1"

    def __init__(
        self,
        translations: Optional[Mapping[str, str]] = None,
        *,
        prefix: str = "模拟译文：",
        fail_islands: Sequence[str] = (),
    ) -> None:
        self.translations = dict(translations or {})
        self.prefix = prefix
        self.fail_islands = frozenset(fail_islands)
        self.requests: list[TranslationRequest] = []

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "mapping_sha256": _canonical_sha256(self.translations),
            "fail_islands": sorted(self.fail_islands),
            "island_prompt_version": ISLAND_PROMPT_VERSION,
            "column_marker_policy": "none",
        }

    def translate(
        self,
        request: TranslationRequest,
        glossary: Glossary,
    ) -> TranslationResponse:
        self.requests.append(request)
        if request.unit_id in self.fail_islands:
            raise TranslationError(f"mock failure for {request.unit_id}")
        text = self.translations.get(
            request.unit_id,
            f"{self.prefix}{request.source_text}",
        )
        return TranslationResponse(text=text, backend=self.name, model=self.model)


def _canonical_sha256(data: Any) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _chat_completions_url(base_url: str) -> str:
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


def _chat_completion(
    base_url: str,
    model: str,
    api_key: Optional[str],
    messages: Sequence[dict],
    *,
    timeout_seconds: float,
    temperature: float,
) -> str:
    """POST one chat completion and return the assistant content."""
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "stream": False,
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = Request(
        _chat_completions_url(base_url),
        data=encoded,
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
        raise TranslationError(f"HTTP {error.code}: {detail or error.reason}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise TranslationError(f"request failed: {error}") from error
    if not 200 <= int(status) < 300:
        raise TranslationError(f"returned HTTP {status}")
    try:
        data = json.loads(body.decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise TranslationResponseError("invalid chat completion") from error
    if not isinstance(content, str):
        raise TranslationResponseError("non-text content")
    return _clean_translation(content)


def _island_messages(
    request: TranslationRequest,
    glossary: Glossary,
    *,
    concise: bool = False,
) -> list[dict]:
    if concise:
        return [
            {
                "role": "system",
                "content": (
                    "你是日文漫画对白翻译器。把下面日文台词翻译成自然、口语化的简体中文，"
                    "拟声词译成中文拟声。只输出译文一行，不要解释、不要序号、不要引号。"
                    "译文用于嵌进固定的文字框，长度以原文为准（日译中通常压到七成），"
                    "宁短勿长；但不要为了压字数丢掉原意。\n\n"
                    f"术语表/人名表：\n{glossary.prompt_text(request.source_text)}"
                ),
            },
            {"role": "user", "content": request.source_text},
        ]
    before = "\n".join(
        f"{unit_id}: {text}" for unit_id, text in request.context_before
    ) or "（无）"
    after = "\n".join(
        f"{unit_id}: {text}" for unit_id, text in request.context_after
    ) or "（无）"
    system = (
        "你是日文漫画对白翻译器。用户提供的是一个高亮文字框（岛）内的完整日文台词，"
        "可能包含多句。请把它整体意译成自然、简洁、口语化的简体中文，保持人物语气与情绪；"
        "遇到生僻词、咒语、粗话、拟声词（SFX）时按中文语境自然处理，不要回避或加注；"
        "结合本页上下文理解指代，但不要翻译上下文，也不要添加原文没有的信息。"
        "译文用于嵌进固定的文字框（大小由原文字数决定），长度以原文为准，"
        "日译中通常压到七成，宁短勿长，但不要为了压字数丢原意。"
        "不要逐字硬译、不要拆列直译、不要输出解释、序号、引号或提示字段，只返回一行译文。"
        "句与句之间用中文标点分隔。\n\n"
        f"本页前文：\n{before}\n\n"
        f"本页后文：\n{after}\n\n"
        f"术语表/人名表：\n{glossary.prompt_text(request.source_text)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": request.source_text},
    ]


def _clean_translation(text: str) -> str:
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
        if cleaned.startswith(("text\n", "json\n")):
            cleaned = cleaned.split("\n", 1)[1].strip()
    cleaned = re.sub(r"^(?:译文|翻译|中文译文)\s*[:：]\s*", "", cleaned)
    if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
        cleaned = cleaned[1:-1].strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def build_island_request(
    dataset: IslandDataset,
    island_id: str,
    source_text: str,
    *,
    context_before: Sequence[Tuple[str, str]] = (),
    context_after: Sequence[Tuple[str, str]] = (),
) -> TranslationRequest:
    return TranslationRequest(
        unit_id=island_id,
        column_ids=(island_id,),
        source_text=source_text,
        translation_query=source_text,
        panel_index=0,
        context_before=tuple(context_before),
        context_after=tuple(context_after),
        ocr_effective_sha256=dataset.effective_sha256,
    )


def _build_translator(
    config: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"
) -> Translator:
    if precision == PRECISION_ONLINE:
        if not config.online_base_url or not config.online_model:
            raise TranslationError("precision=online requires online_base_url and online_model")
        return IslandOpenAICompatibleTranslator(
            name="online-ai",
            base_url=config.online_base_url,
            model=config.online_model,
            api_key=config.online_api_key,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
            prompt_style=prompt_style,
        )
    return IslandOpenAICompatibleTranslator(
        name="sakura-local",
        base_url=config.sakura_base_url,
        model=config.sakura_model,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        prompt_style=prompt_style,
    )


def translate_island_page(
    dataset: IslandDataset,
    texts: Mapping[str, str],
    config: IslandTranslateConfig,
    *,
    glossary: Optional[Glossary] = None,
    cache: Optional[TranslationCache] = None,
    context: bool = True,
    translator_factory: Optional[Callable[[IslandTranslateConfig, str], Translator]] = None,
) -> IslandTranslationResult:
    """Translate every retained island's whole text as one unit."""
    if config.precision not in _PRECISIONS:
        raise TranslationError(f"unsupported precision: {config.precision!r}")
    if config.precision == PRECISION_ONLINE and (
        not config.online_base_url or not config.online_model
    ):
        raise TranslationError(
            "precision=online requires online_base_url and online_model"
        )
    if glossary is not None:
        active_glossary = glossary
    elif config.glossary_path:
        active_glossary = load_glossary(Path(config.glossary_path))
    else:
        active_glossary = load_glossary(None)
    active_cache = (
        cache
        if cache is not None
        else IslandTranslationCache(Path("var/island_translation_cache"))
    )
    factory = translator_factory or _build_translator

    ordered = [island.id for island in dataset.islands if island.id in texts]
    order_index = {island_id: index for index, island_id in enumerate(ordered)}
    units = []
    for index, island_id in enumerate(ordered):
        source_text = str(texts[island_id] or "").strip()
        if not source_text:
            continue
        before, after = (), ()
        if context:
            before = tuple(
                (other, str(texts[other]))
                for other in ordered[:index]
                if str(texts.get(other) or "").strip()
            )
            after = tuple(
                (other, str(texts[other]))
                for other in ordered[index + 1 :]
                if str(texts.get(other) or "").strip()
            )
        request = build_island_request(
            dataset,
            island_id,
            source_text,
            context_before=before,
            context_after=after,
        )
        primary_precision = config.precision
        sequence = [primary_precision]
        if primary_precision == PRECISION_ONLINE:
            sequence.append(PRECISION_LOCAL)
        attempts: list[Tuple[str, str]] = []
        translated: Optional[str] = None
        backend: Optional[str] = None
        model: Optional[str] = None
        cache_key: Optional[str] = None
        last_signature = None
        cache_hit = False
        failure: Optional[str] = None
        kept_source = False
        if _is_short_symbol(source_text):
            # 1-2 char punctuation/symbol islands need no translation: keep
            # the source without calling the engine (avoids leakage).
            skip_reason = "short-symbol"
        elif is_numeric_passthrough(source_text):
            # 纯数字/符号岛同理：送引擎既白花一次调用，又可能把 ``１２０`` 翻成
            # "一百二十"（画面上的数字被改成汉字，比不译更糟）。见 text_rules。
            skip_reason = "numeric-passthrough"
        else:
            skip_reason = None
        if skip_reason is not None:
            translated = source_text
            kept_source = True
            attempts.append(("skip", skip_reason))
        engine_attempts = 0
        for precision in sequence:
            for style in ("contextual", "concise"):
                if translated is not None:
                    break
                translator = factory(config, precision, style)
                signature = active_cache.signature(request, active_glossary, translator)
                last_signature = signature
                cache_key = signature.key
                cached = active_cache.load(signature, active_glossary)
                if cached is not None:
                    translated = cached.text
                    backend = cached.backend
                    model = cached.model
                    cache_hit = True
                    attempts.append((precision, "cache-hit"))
                    break
                engine_attempts += 1
                if engine_attempts > 3:
                    # hard cap: three engine calls per island, then keep the
                    # source (never leave the island blank)
                    failure = "engine attempts exceeded 3"
                    attempts.append((precision, f"failed: {failure}"))
                    break
                try:
                    response = translator.translate(request, active_glossary)
                except TranslationTimeoutError as error:
                    # 超时 = 本次请求预算已经花光。在**同一个引擎**上再试一次
                    # （只是换 prompt 风格）大概率再花光一次，所以直接换下一档
                    # 精度（本地 Sakura 仍在序列里，岛不会因此没译文）。
                    attempts.append((precision, f"timeout: {error}"))
                    failure = str(error)
                    break
                except TranslationError as error:
                    attempts.append((precision, f"failed: {error}"))
                    failure = str(error)
                    continue
                try:
                    context_texts = tuple(
                        text for _, text in (*request.context_before, *request.context_after)
                    )
                    _validate_island_translation(
                        response.text,
                        source_text,
                        active_glossary,
                        context_texts=context_texts,
                    )
                except TranslationResponseError as error:
                    attempts.append((precision, f"failed: {error}"))
                    failure = str(error)
                    kept_loader = getattr(active_cache, "load_kept_source", None)
                    if kept_loader is not None:
                        kept = kept_loader(signature)
                        if kept is not None:
                            translated = kept
                            kept_source = True
                            attempts.append((precision, "kept-source-cache"))
                            break
                    continue
                active_cache.store(signature, response)
                attempts.append((precision, "translated"))
                translated = response.text
                backend = response.backend
                model = response.model
                failure = None
                break
            if translated is not None:
                break
        if translated is None:
            # Two translation rounds (contextual + concise) both failed, so
            # the island is almost certainly punctuation/symbols or an
            # untranslatable SFX mark: keep the source text as the final
            # translation (never leave the island blank) and cache the
            # decision so later runs skip the doomed requests.
            translated = source_text
            kept_source = True
            attempts.append(("fallback", "kept-source"))
            storer = getattr(active_cache, "store_kept_source", None)
            if storer is not None and last_signature is not None:
                storer(last_signature, source_text)
        if kept_source and not cache_hit:
            _LOG.warning(
                "kept-source island=%s text=%r attempts=%s",
                island_id,
                source_text,
                attempts,
            )
        status = "kept-source" if kept_source else "translated"
        kind = "symbol" if kept_source and _is_symbol_only(source_text) else "text"
        units.append(
            IslandUnitTranslation(
                island_id=island_id,
                source_text=source_text,
                translated_text=translated,
                status=status,
                backend=backend,
                model=model,
                cache_key=cache_key,
                cache_hit=cache_hit,
                attempts=tuple(attempts),
                failure_reason=failure,
                kind=kind,
            )
        )
    return IslandTranslationResult(
        page=dataset.page,
        precision=config.precision,
        units=tuple(units),
        failed_count=sum(unit.status == "failed" for unit in units),
        cache_hit_count=sum(unit.cache_hit for unit in units),
    )


def _is_refused(translated: str, source: str) -> bool:
    """Refuse outputs that merely echo the source (leakage / no translation)."""
    cleaned = _clean_translation(translated)
    source_clean = _clean_translation(source)
    if not cleaned:
        return True
    if cleaned == source_clean:
        return True
    return False


_LEAKED_PROMPT_FIELDS = (
    "本页前文",
    "本页后文",
    "术语表",
    "人名表",
    "当前完整",
    "当前原文",
    "原列标记",
    "列完整性",
    "面板索引",
    "无指定术语",
    "输出当前",
    "（无）",
    "日文漫画翻译",
    "只输出译文",
    "简体中文",
    "不要任何解释",
    "用户给出的日文",
    "翻译成",
)


def _validate_island_translation(
    translated: str,
    source: str,
    glossary: Optional[Glossary] = None,
    context_texts: Sequence[str] = (),
) -> None:
    """Island-specific validation: strict on leakage/kana/length, lenient on
    punctuation (manga OCR noise makes ?/! preservation rules misfire)."""
    import re

    if not translated.strip():
        raise TranslationResponseError("translator returned empty text")
    if translated.strip() == source.strip():
        raise TranslationResponseError("translator returned the unchanged source text")
    if any(field in translated for field in _LEAKED_PROMPT_FIELDS):
        raise TranslationResponseError("translator leaked prompt fields")
    lines = [line for line in translated.splitlines() if line.strip()]
    if len(lines) > 4 and len(set(lines)) == 1:
        raise TranslationResponseError("translator repeated one line >4 times")
    for context in context_texts:
        fragment = _normalize_for_leak(context)
        if len(fragment) >= 2 and fragment in translated and fragment not in source:
            raise TranslationResponseError(
                f"translator leaked context text: {fragment[:12]}"
            )
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


def _is_symbol_only(text: str) -> bool:
    """True when the text carries no kanji/kana/latin/digit characters."""
    import re

    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    return not re.search(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]", cleaned)


_SHORT_SYMBOL_CHARS = frozenset(
    "、。！？…‥．.・,，!?〜ー-–—_·♡♥○●◎◆◇△▲■□☆★"
    "（）()【】[]「」『』\"'　 "
)


def _is_short_symbol(text: str) -> bool:
    """True for a 1-2 character punctuation/symbol island.

    Such islands carry no translatable text; asking the engine wastes a
    call and invites prompt leakage, so they are kept as-is without any
    engine request.
    """
    cleaned = str(text or "").strip()
    return 0 < len(cleaned) <= 2 and all(
        char in _SHORT_SYMBOL_CHARS for char in cleaned
    )


def _normalize_for_leak(text: str) -> str:
    return "".join(
        char
        for char in str(text or "")
        if char not in " 、。！？…‥．.・,，!?\"'「」『』”’‘«»~〜ー-–—_()（）[]【】:：;；"
    )


class IslandTranslationCache(TranslationCache):
    """Cache that validates with the island rules (no ?/! preservation)."""

    def load(
        self,
        signature,
        glossary,
    ):
        from .translator import CACHE_FORMAT_VERSION

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
            context_texts = tuple(
                str(item[1])
                for item in (
                    *signature.data.get("context_before", []),
                    *signature.data.get("context_after", []),
                )
            )
            _validate_island_translation(
                response.text,
                signature.data["source_text"],
                glossary,
                context_texts=context_texts,
            )
            return response
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            TranslationError,
        ):
            return None

    def store_kept_source(self, signature, source_text: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{signature.key}.kept.json").write_text(
            json.dumps(
                {"kept_source": True, "source": source_text},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_kept_source(self, signature) -> Optional[str]:
        path = self.directory / f"{signature.key}.kept.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not payload.get("kept_source"):
            return None
        return str(payload.get("source") or "")


@dataclass(frozen=True)
class PageBatchResult:
    page: str
    backend: str
    model: str
    cache_hit: bool
    translations: Mapping[str, str]
    failures: Tuple[str, ...]
    parse_errors: Tuple[str, ...]
    raw_response: str


def _batch_messages(
    numbered_lines: Sequence[Tuple[int, str]],
    glossary: Glossary,
    previous_lines: Sequence[str] = (),
) -> list[dict]:
    body = "\n".join(f"{index}: {text}" for index, text in numbered_lines)
    # 跨页上下文：**只给上一页的原文，不给译文**。这样不依赖"上一页已经翻完"——
    # 页仍然可以并行翻译（`_stage2_translate` 用线程池按页并发），缓存键也完全
    # 确定（上一页原文在开跑前就已知）。用途是**指代/人称/对话延续**；参考实现
    # （overlay-translator 的 previous_frame）会连译文一起给，但那要求顺序翻译，
    # 在大书上会显著变慢。
    previous_block = ""
    if previous_lines:
        joined = "\n".join(
            f"- {line}" for line in previous_lines if str(line).strip())
        if joined:
            previous_block = (
                "\n\n上一页的原文（仅供理解指代与对话延续，"
                "**不要翻译、不要复述、不要输出**）：\n" + joined)
    system = (
        "你是日文漫画对白翻译器。用户提供一页内多个高亮岛（台词框）的编号日文文本，"
        "编号即阅读顺序。请把每条整体意译成自然、简洁、口语化的简体中文，"
        "拟声词（SFX）译成中文拟声；结合整页上下文理解指代，不要翻译上下文；"
        "不要逐字硬译、不要加原文没有的信息、不要解释。"
        "每条译文用于嵌进固定的文字框（大小由该条原文长度决定），"
        "长度以各自原文为准，日译中通常压到七成，宁短勿长，但不要为了压字数丢原意。"
        "必须逐条输出，格式严格为每行一个『编号: 译文』，不要输出其他任何内容。\n\n"
        f"术语表/人名表：\n{glossary.prompt_text()}"
        f"{previous_block}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


def _parse_batch_response(
    raw: str,
    numbered: Mapping[int, str],
) -> Tuple[Mapping[str, str], Tuple[str, ...], Tuple[str, ...]]:
    """Parse ``1: 译文`` entries back into island texts.

    The model may emit every numbered entry on its own line OR run them
    together on one line (space separated), so the entries are split on the
    ``N: `` markers rather than anchored to line boundaries.

    Returns ``(translations, failures, parse_errors)``; islands without a
    valid numbered entry land in *failures* and fall back to local.
    """
    import re

    translations: dict[str, str] = {}
    seen: set[int] = set()
    errors: list[str] = []
    markers = list(re.finditer(r"(\d+)\s*[:：]\s*", raw))
    for position, match in enumerate(markers):
        index = int(match.group(1))
        start = match.end()
        end = markers[position + 1].start() if position + 1 < len(markers) else len(raw)
        text = _clean_translation(raw[start:end])
        if not text:
            errors.append(f"entry {index}: empty translation")
            continue
        if index in seen:
            errors.append(f"entry {index}: duplicate")
            continue
        seen.add(index)
        island_id = numbered.get(index)
        if island_id is None:
            errors.append(f"entry {index}: unknown number")
            continue
        translations[island_id] = text
    failures = tuple(
        island_id for index, island_id in numbered.items() if island_id not in translations
    )
    return translations, failures, tuple(errors)


class PageBatchCache:
    """Page-level cache for whole-page batched translations."""

    FORMAT_VERSION = 2

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()

    def signature(
        self,
        dataset: IslandDataset,
        texts: Mapping[str, str],
        config: IslandTranslateConfig,
        glossary: Optional[Glossary] = None,
        previous_lines: Sequence[str] = (),
    ) -> str:
        ordered = sorted(
            (island for island in dataset.islands if island.id in texts),
            key=lambda island: -island.image_bbox[2],
        )
        data = {
            "format": "island-page-batch-v1",
            "format_version": self.FORMAT_VERSION,
            "page": dataset.page,
            "effective_sha256": dataset.effective_sha256,
            "base_url": config.online_base_url.rstrip("/"),
            "model": config.online_model,
            "temperature": config.temperature,
            "prompt_version": ISLAND_PROMPT_VERSION,
            # 术语表进签名：术语是提示词的一部分（拟声词自动收录会改它），
            # 不进签名就会出现"改了术语表但命中旧缓存、改动毫无效果"。
            "glossary_sha256": glossary.sha256 if glossary is not None else "",
            "texts": {island.id: str(texts[island.id]) for island in ordered},
            # 上一页原文也是提示词的一部分（跨页上下文），必须进签名
            "previous_lines": [str(line) for line in previous_lines],
        }
        return _canonical_sha256(data)

    def load(self, key: str) -> Optional[dict]:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if payload.get("format_version") != self.FORMAT_VERSION:
            return None
        return payload

    def store(self, key: str, payload: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def translate_page_batch(
    dataset: IslandDataset,
    texts: Mapping[str, str],
    config: IslandTranslateConfig,
    *,
    glossary: Optional[Glossary] = None,
    cache: Optional[PageBatchCache] = None,
    chat_fn=None,
    previous_lines: Sequence[str] = (),
) -> PageBatchResult:
    """Whole-page batched translation through the online AI (precision=online).

    Every island of the page is numbered in reading order and sent in ONE
    request; the response is parsed back line by line.  Islands whose line is
    missing or malformed land in ``failures`` for the local fallback.
    """
    if config.precision != PRECISION_ONLINE:
        raise TranslationError("translate_page_batch requires precision=online")
    if not config.online_base_url or not config.online_model:
        raise TranslationError("precision=online requires online_base_url and online_model")
    active_glossary = glossary if glossary is not None else load_glossary(None)
    active_cache = cache if cache is not None else PageBatchCache(Path("var/island_batch_cache"))

    ordered = sorted(
        (island for island in dataset.islands if island.id in texts),
        key=lambda island: -island.image_bbox[2],
    )
    numbered = {index: island.id for index, island in enumerate(ordered, 1)}
    key = active_cache.signature(
        dataset, texts, config, active_glossary, previous_lines)
    cached = active_cache.load(key)
    if cached is not None:
        return PageBatchResult(
            page=dataset.page,
            backend="online-ai-batch",
            model=cached.get("model") or config.online_model,
            cache_hit=True,
            translations=dict(cached.get("translations", {})),
            failures=tuple(cached.get("failures", [])),
            parse_errors=tuple(cached.get("parse_errors", [])),
            raw_response=str(cached.get("raw_response", "")),
        )
    if chat_fn is None:
        chat_fn = lambda messages: _chat_completion(
            config.online_base_url,
            config.online_model,
            config.online_api_key,
            messages,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
        )
    messages = _batch_messages(
        [(index, str(texts[island_id])) for index, island_id in numbered.items()],
        active_glossary,
        previous_lines,
    )
    raw = chat_fn(messages)
    translations, failures, errors = _parse_batch_response(raw, numbered)
    active_cache.store(
        key,
        {
            "format_version": PageBatchCache.FORMAT_VERSION,
            "model": config.online_model,
            "raw_response": raw,
            "translations": translations,
            "failures": list(failures),
            "parse_errors": list(errors),
        },
    )
    return PageBatchResult(
        page=dataset.page,
        backend="online-ai-batch",
        model=config.online_model,
        cache_hit=False,
        translations=translations,
        failures=failures,
        parse_errors=errors,
        raw_response=raw,
    )
