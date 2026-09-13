from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from manga_localizer.errors import TranslationError, TranslationResponseError
from manga_localizer.translator import (
    OpenAICompatibleTranslator,
    TranslationRequest,
    _chat_completions_url,
    load_glossary,
)

_GLOSSARY = load_glossary(None)


def _request() -> TranslationRequest:
    return TranslationRequest(
        unit_id="unit_001",
        column_ids=("col_001",),
        source_text="おはよう",
        translation_query="<1>おはよう",
        panel_index=0,
        context_before=(),
        context_after=(),
        ocr_effective_sha256="a" * 64,
    )


def _completion(content: str, finish_reason: str | None = "stop") -> bytes:
    return json.dumps({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
    }).encode("utf-8")


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _translator(**overrides) -> OpenAICompatibleTranslator:
    defaults = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "test-model",
        "api_key": "secret-key",
    }
    defaults.update(overrides)
    return OpenAICompatibleTranslator(**defaults)


class ChatCompletionsUrlTest(unittest.TestCase):
    def test_appends_chat_completions(self):
        self.assertEqual(
            _chat_completions_url("http://host:8080/v1"),
            "http://host:8080/v1/chat/completions",
        )

    def test_strips_trailing_slash(self):
        self.assertEqual(
            _chat_completions_url("http://host:8080/v1/"),
            "http://host:8080/v1/chat/completions",
        )

    def test_keeps_full_endpoint(self):
        self.assertEqual(
            _chat_completions_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions",
        )


class OpenAICompatibleTranslatorHTTPTest(unittest.TestCase):
    def test_success_returns_cleaned_text(self):
        translator = _translator()
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(_completion("早上好。")),
        ) as urlopen:
            response = translator.translate(_request(), _GLOSSARY)
        self.assertEqual(response.text, "早上好。")
        self.assertEqual(response.backend, "sakura-openai-compatible")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.finish_reason, "stop")
        http_request = urlopen.call_args.args[0]
        self.assertTrue(http_request.full_url.endswith("/chat/completions"))
        self.assertEqual(http_request.get_method(), "POST")
        self.assertEqual(http_request.headers["Authorization"], "Bearer secret-key")
        self.assertEqual(http_request.headers["Content-type"], "application/json")

    def test_no_api_key_omits_authorization(self):
        translator = _translator(api_key=None)
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(_completion("早上好。")),
        ) as urlopen:
            translator.translate(_request(), _GLOSSARY)
        self.assertNotIn("Authorization", urlopen.call_args.args[0].headers)

    def test_http_error_raises_translation_error(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8080/v1/chat/completions", 500,
            "Internal Server Error", {}, io.BytesIO(b"boom"),
        )
        with mock.patch("manga_localizer.translator.urlopen", side_effect=error):
            with self.assertRaises(TranslationError) as caught:
                _translator().translate(_request(), _GLOSSARY)
        self.assertIn("HTTP 500", str(caught.exception))
        self.assertIn("boom", str(caught.exception))

    def test_url_error_raises_translation_error(self):
        with mock.patch(
            "manga_localizer.translator.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(TranslationError) as caught:
                _translator().translate(_request(), _GLOSSARY)
        self.assertIn("request failed", str(caught.exception))
        self.assertIn("connection refused", str(caught.exception))

    def test_non_2xx_status_raises(self):
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(b"{}", status=502),
        ):
            with self.assertRaises(TranslationError) as caught:
                _translator().translate(_request(), _GLOSSARY)
        self.assertIn("HTTP 502", str(caught.exception))

    def test_malformed_json_raises_response_error(self):
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(b"not json"),
        ):
            with self.assertRaises(TranslationResponseError):
                _translator().translate(_request(), _GLOSSARY)

    def test_missing_choices_raises_response_error(self):
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(b'{"object": "chat.completion"}'),
        ):
            with self.assertRaises(TranslationResponseError):
                _translator().translate(_request(), _GLOSSARY)

    def test_non_text_content_raises_response_error(self):
        payload = json.dumps({"choices": [{"message": {"content": 123}}]}).encode("utf-8")
        with mock.patch(
            "manga_localizer.translator.urlopen",
            return_value=_FakeResponse(payload),
        ):
            with self.assertRaises(TranslationResponseError):
                _translator().translate(_request(), _GLOSSARY)

    def test_invalid_prompt_style_rejected(self):
        with self.assertRaises(TranslationError):
            OpenAICompatibleTranslator(prompt_style="shorthand")


if __name__ == "__main__":
    unittest.main()
