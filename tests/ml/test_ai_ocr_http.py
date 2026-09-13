from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image

from manga_localizer.ai_ocr import ai_ocr_readings


def _completion(content: str) -> bytes:
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }).encode("utf-8")


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Progress:
    def __init__(self):
        self.logs: list[str] = []
        self.subs: list[tuple] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def sub_progress(self, done: int, total: int, label: str) -> None:
        self.subs.append((done, total, label))


class AiOcrReadingsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name) / "page.png"
        Image.new("RGB", (40, 20), "white").save(self.source)
        self.crops = [{
            "id": "001:box:1", "source": str(self.source),
            "bbox": [0, 0, 40, 20],
        }]

    def _run(self, side_effect, progress=None):
        with mock.patch("urllib.request.urlopen", side_effect=side_effect), \
                mock.patch("manga_localizer.ai_ocr.time.sleep"):
            return ai_ocr_readings(
                self.crops, "http://ai.example/v1", "vl-model", "ai-key",
                progress=progress,
            )

    def test_success_parses_content_and_request(self):
        progress = _Progress()
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(
            _completion("おはよう\nそんな"),
        )) as urlopen, mock.patch("manga_localizer.ai_ocr.time.sleep"):
            readings = ai_ocr_readings(
                self.crops, "http://ai.example/v1", "vl-model", "ai-key",
                progress=progress,
            )
        self.assertEqual(readings, {"001:box:1": "おはよう\nそんな"})
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
        self.assertEqual(request.headers["Authorization"], "Bearer ai-key")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "vl-model")
        self.assertEqual(payload["temperature"], 0.0)
        image_url = payload["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(progress.subs, [(1, 1, "001:box:1")])

    def test_transient_error_retries_then_succeeds(self):
        responses = [
            urllib.error.URLError("timed out"),
            _FakeResponse(_completion("読めた")),
        ]
        readings = self._run(responses)
        self.assertEqual(readings, {"001:box:1": "読めた"})

    def test_persistent_error_returns_empty_and_logs(self):
        progress = _Progress()
        responses = [
            urllib.error.URLError("down"),
            urllib.error.URLError("down"),
            urllib.error.HTTPError("http://ai.example/v1/chat/completions",
                                   503, "unavailable", {}, io.BytesIO(b"")),
        ]
        readings = self._run(responses, progress=progress)
        self.assertEqual(readings, {"001:box:1": ""})
        self.assertEqual(len(progress.logs), 1)
        self.assertIn("001:box:1 failed", progress.logs[0])

    def test_malformed_response_treated_as_empty(self):
        responses = [_FakeResponse(b"<html>gateway error</html>")] * 3
        readings = self._run(responses)
        self.assertEqual(readings, {"001:box:1": ""})


if __name__ == "__main__":
    unittest.main()
