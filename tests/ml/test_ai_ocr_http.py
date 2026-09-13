from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
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


class AiOcrParallelTests(unittest.TestCase):
    """并发读取：逐框串行一本 586 框要 20+ 分钟，并发是这里的性能开关。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name) / "page.png"
        Image.new("RGB", (40, 20), "white").save(self.source)
        self.crops = [
            {"id": f"001:box:{index}", "source": str(self.source),
             "bbox": [0, 0, 40, 20]}
            for index in range(1, 13)
        ]

    def _run_parallel(self, workers, side_effect, progress=None):
        with mock.patch("urllib.request.urlopen", side_effect=side_effect),                 mock.patch("manga_localizer.ai_ocr.time.sleep"):
            return ai_ocr_readings(
                self.crops, "http://ai.example/v1", "vl-model", "ai-key",
                progress=progress, max_workers=workers,
            )

    def test_parallel_reads_every_crop_once(self):
        progress = _Progress()
        contents = {f"001:box:{i}": f"テキスト{i}" for i in range(1, 13)}
        seen: list[str] = []
        lock = threading.Lock()

        def urlopen(request, **kwargs):
            with lock:
                seen.append(request)
                index = len(seen)
            return _FakeResponse(_completion(contents[f"001:box:{index}"]))

        readings = self._run_parallel(6, urlopen, progress)
        self.assertEqual(len(seen), 12)
        # 每个框都读到了（内容按处理顺序分配，所以只校验集合与键）
        self.assertEqual(sorted(readings), sorted(contents))
        self.assertEqual(sorted(readings.values()), sorted(contents.values()))
        self.assertEqual(len(progress.subs), 12)
        self.assertEqual({sub[0] for sub in progress.subs}, set(range(1, 13)))

    def test_parallel_partial_failure_keeps_others(self):
        def urlopen(request, **kwargs):
            return _FakeResponse(_completion("読めた"))

        readings = self._run_parallel(4, urlopen)
        self.assertEqual(len(readings), 12)
        self.assertTrue(all(text == "読めた" for text in readings.values()))

    def test_workers_one_is_sequential(self):
        readings = self._run_parallel(
            1, lambda request, **kwargs: _FakeResponse(_completion("ひとつ")),
        )
        self.assertEqual(len(readings), 12)

    def test_empty_crops_no_request(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            readings = ai_ocr_readings(
                [], "http://ai.example/v1", "vl-model", "ai-key",
            )
        self.assertEqual(readings, {})
        urlopen.assert_not_called()


class AiOcrEncodeTests(unittest.TestCase):
    """编码降采样：长边超上限才缩，正常漫画框不动。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.small = Path(tmp.name) / "small.png"
        Image.new("RGB", (200, 100), "white").save(self.small)
        self.big = Path(tmp.name) / "big.png"
        Image.new("RGB", (4000, 1000), "white").save(self.big)

    def _decode(self, data_url):
        payload = data_url.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
            return image.size

    def test_small_crop_not_resized(self):
        from manga_localizer.ai_ocr import _encode_crop

        self.assertEqual(self._decode(_encode_crop(str(self.small), [0, 0, 200, 100])),
                         (200, 100))

    def test_oversized_crop_downscaled(self):
        from manga_localizer.ai_ocr import MAX_SIDE, _encode_crop

        width, height = self._decode(_encode_crop(str(self.big), [0, 0, 4000, 1000]))
        self.assertEqual(width, MAX_SIDE)
        self.assertEqual(height, MAX_SIDE // 4)


if __name__ == "__main__":
    unittest.main()
