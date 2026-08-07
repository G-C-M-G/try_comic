"""OpenAI-compatible vision OCR for blue-box crops.

Posts each crop as a data-URL image message to a vision-capable endpoint
(e.g. GLM-4V, Qwen-VL) and returns the recognised Japanese text.  Used
when the user configures ``--ai-ocr-url/--ai-ocr-model/--ai-ocr-api-key``
instead of the local manga-ocr model.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

from PIL import Image

SYSTEM_PROMPT = (
    "你是日文漫画 OCR 引擎。图片是一块漫画文字区域（可能含多句/多列），"
    "请原样输出其中所有日文文本，保留假名/汉字/标点，不要翻译、不要解释、"
    "不要添加任何额外内容，多个句子用换行分隔。只输出文字本身。"
)

_TIMEOUT = 60.0
_RETRIES = 2


def _encode_crop(source: str, bbox: list[int]) -> str:
    with Image.open(source).convert("RGB") as image:
        crop = image.crop(tuple(bbox))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_one(
    base_url: str,
    model: str,
    api_key: str,
    image_data: str,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return str(content).strip()


def ai_ocr_readings(
    crops: list[dict],
    base_url: str,
    model: str,
    api_key: str,
    *,
    progress=None,
) -> dict[str, str]:
    """OCR every crop through the vision endpoint; per-crop retries."""
    readings: dict[str, str] = {}
    total = len(crops)
    for index, entry in enumerate(crops, 1):
        image_data = _encode_crop(entry["source"], entry["bbox"])
        text = ""
        for attempt in range(_RETRIES + 1):
            try:
                text = _read_one(base_url, model, api_key, image_data)
                break
            except Exception as error:  # noqa: BLE001 - retry then accept empty
                if attempt == _RETRIES:
                    text = ""
                    if progress is not None:
                        progress.log(
                            f"ai-ocr {entry['id']} failed: {error}"
                        )
                else:
                    time.sleep(1.0 + attempt)
        readings[entry["id"]] = text
        if progress is not None:
            progress.sub_progress(index, total, entry["id"])
    return readings
