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
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

SYSTEM_PROMPT = (
    "你是日文漫画 OCR 引擎。图片是一块漫画文字区域（可能含多句/多列），"
    "请原样输出其中所有日文文本，保留假名/汉字/标点，不要翻译、不要解释、"
    "不要添加任何额外内容，多个句子用换行分隔。只输出文字本身。\n"
    "严格约束：\n"
    "1. 框里有几行就输出几行，不增行、不并句、不补全；\n"
    "2. 看不清的字不要猜、不要靠上下文补写，拿不准宁少不编；\n"
    "3. 不要输出「图片中写着」「以下是」「（无法辨认）」这类说明文字；\n"
    "4. 输出长度应与框内文字大致相当，不得扩写、总结或复述。"
)

_TIMEOUT = 60.0
_RETRIES = 2

# 并发度：逐框串行时每框一个请求（实测 1.91s/框），一本 586 框的书写要 18+ 分钟。
# 视觉 OCR 是纯网络等待型负载，实测（deepseek-flash，34 框一批）：
#   1 路 1.91s/框 → 8 路 0.20s/框（9×）→ 16 路 0.10s/框（19×）→ 24 路反而退化
# 默认取 12（实测区间内速度已接近上限，且留有余量）。24 路实测反而退化，但那只是
# 本机/本账号的一次测量，不是硬规律——护栏放宽到 32 只作"防手滑"，真实选值看
# `--ai-ocr-workers` 与 GUI「AI OCR 并发」，撞 429 就调小。
DEFAULT_WORKERS = 12
MAX_WORKERS = 32

# 编码前的降采样上限（长边 px）：漫画单框通常 50~300px，不会触发；
# 触发时只是把大框缩小，传输与推理都更快，视觉模型仍能读清（字远大于此）。
MAX_SIDE = 1280


def _encode_crop(source: str, bbox: list[int], *, max_side: int = MAX_SIDE) -> str:
    with Image.open(source).convert("RGB") as image:
        crop = image.crop(tuple(bbox))
        if max_side and max(crop.size) > max_side:
            scale = max_side / float(max(crop.size))
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.LANCZOS,
            )
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


def _read_crop(
    crop: dict,
    base_url: str,
    model: str,
    api_key: str,
    progress,
    *,
    max_side: int = MAX_SIDE,
) -> tuple[str, str]:
    """单框读取（含重试）：返回 ``(crop_id, text)``，失败给空串。"""
    image_data = _encode_crop(crop["source"], crop["bbox"], max_side=max_side)
    text = ""
    for attempt in range(_RETRIES + 1):
        try:
            text = _read_one(base_url, model, api_key, image_data)
            break
        except Exception as error:  # noqa: BLE001 - retry then accept empty
            if attempt == _RETRIES:
                text = ""
                if progress is not None:
                    progress.log(f"ai-ocr {crop['id']} failed: {error}")
            else:
                time.sleep(1.0 + attempt)
    return crop["id"], text


def ai_ocr_readings(
    crops: list[dict],
    base_url: str,
    model: str,
    api_key: str,
    *,
    progress=None,
    max_workers: int = DEFAULT_WORKERS,
) -> dict[str, str]:
    """OCR every crop through the vision endpoint.

    并发发送（``max_workers``，默认 12、护栏 32）：逐框串行时每框一次请求、一本
    586 框的书要 18+ 分钟，而视觉 OCR 是纯网络等待型负载，实测 8 路约 9×、16 路
    约 19×（见表见模块头注释）。``max_workers<=1`` 退回串行（调试/限流时用）。
    每个框独立重试，单个失败不影响其它框（读空的框由调用方回退 manga-ocr）。
    """
    readings: dict[str, str] = {}
    total = len(crops)
    if total == 0:
        return readings
    workers = max(1, min(MAX_WORKERS, int(max_workers)))
    done = 0
    lock = threading.Lock()

    def _report(crop_id: str) -> None:
        nonlocal done
        with lock:
            done += 1
            current = done
        if progress is not None:
            progress.sub_progress(current, total, crop_id)

    if workers == 1:
        for crop in crops:
            crop_id, text = _read_crop(
                crop, base_url, model, api_key, progress,
            )
            readings[crop_id] = text
            _report(crop_id)
        return readings

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_read_crop, crop, base_url, model, api_key, progress)
            for crop in crops
        ]
        for future in futures:
            crop_id, text = future.result()
            readings[crop_id] = text
            _report(crop_id)
    return readings
