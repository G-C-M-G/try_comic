"""视觉判据：让读图的模型判"这个岛是拟声还是台词"。

用于**自检不住的书**（`art_text_gate` 返回 None，见 thresholds 里的标定表）：
那种书上"岛底平坦度"没有区分度（实测 AUC 0.65），只有看图判才有信号
（实测 F1 0.833，只算拟声口径）。

判三档（**一次调用同时覆盖"空框/乱码"那一类**，不必再做字符统计规则）：

* ``S`` 拟声 / 效果字 → 不翻译，保留原图
* ``D`` 台词 / 旁白 → 照常翻译
* ``U`` 图里没有清晰可读的日文（空框、纯装饰、糊成一团）→ 保留原图

送图是**岛的取样矩形 ± 上下文**（实测 ±60px 比只送框好：台词误伤 90.0% vs 79.2%
精确率）。**不能复用 `ai_ocr._read_one`**——那里的 system prompt 写死做 OCR，
直接复用会得到"复述文字"（第一次跑就是这么翻车的）。
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

SYSTEM = (
    "你是漫画排版分析器。图片是漫画里一处文字区域及周围的裁剪。请判断这处字属于哪一类：\n"
    "S = 拟声词 / 效果字：声音、拟态、纯装饰性大字（例如 ドキドキ / ドカッ / "
    "ふるふる / もみもみ / ぶらあん / ただの ♡ 符号）。这类「字」本身不是一句可翻译的话。\n"
    "D = 台词 / 旁白：人物说的话、内心独白、叙述（例如 よろしく / だいすき / "
    "どういうことだ？ / ごめんなさい）。这类字是可以翻译成中文的句子。\n"
    "U = 图里没有清晰可读的日文：空白/纯色区域、纯装饰线条、或者糊成一团认不出字。\n"
    "判断方法：先读字，再看它是不是一句有意义的话——有实词或助词、能当「话」用的 → D；"
    "只是声音/拟态/装饰符号 → S；根本没有可读的文字 → U。\n"
    "只输出一个字母：S、D 或 U。不要解释，不要输出别的字。"
)

_TIMEOUT = 60.0
_RETRIES = 2
DEFAULT_WORKERS = 12
MAX_WORKERS = 32


def _encode(source: str, bbox: list[int], pad: int) -> str:
    """裁图（岛框 ± pad）→ data URL。"""
    with Image.open(source).convert("RGB") as image:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        crop = image.crop((max(0, x1 - pad), max(0, y1 - pad),
                           min(image.width, x2 + pad), min(image.height, y2 + pad)))
        if crop.width < 4 or crop.height < 4:
            crop = image.crop((0, 0, min(8, image.width), min(8, image.height)))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _ask(base_url: str, model: str, api_key: str, image_data: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data}}]},
        ],
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"]).strip()


def judge_crops(
    crops: list[dict],
    base_url: str,
    model: str,
    api_key: str,
    *,
    progress=None,
    max_workers: int = DEFAULT_WORKERS,
) -> dict[str, str]:
    """逐岛判定：``{id: "S"|"D"|"U"}``（失败给空串，调用方当"读不出"处理）。

    ``crops`` 每项：``{"id", "source", "bbox", "pad"}``。并发发送（视觉判定是
    纯网络等待型负载，与 ai_ocr 同源）；单项失败不影响其它项。
    """
    workers = max(1, min(int(max_workers), MAX_WORKERS))
    lock = threading.Lock()
    answers: dict[str, str] = {}

    def _one(crop: dict) -> tuple[str, str]:
        try:
            image_data = _encode(crop["source"], crop["bbox"], int(crop.get("pad") or 0))
        except (OSError, ValueError) as error:
            if progress is not None:
                progress.log(f"sfx-judge {crop['id']} 裁图失败: {error}")
            return crop["id"], ""
        for attempt in range(_RETRIES + 1):
            try:
                answer = _ask(base_url, model, api_key, image_data)
            except Exception as error:  # noqa: BLE001 - 重试后放弃
                if attempt == _RETRIES:
                    if progress is not None:
                        progress.log(f"sfx-judge {crop['id']} 失败: {error}")
                    return crop["id"], ""
                time.sleep(1.0 + attempt)
                continue
            for char in answer.upper():
                if char in ("S", "D", "U"):
                    return crop["id"], char
            return crop["id"], ""          # 没按格式回 → 当读不出
        return crop["id"], ""

    if workers == 1 or len(crops) <= 1:
        results = [_one(crop) for crop in crops]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, crops))
    for key, answer in results:
        answers[key] = answer
    with lock:
        pass
    return answers
