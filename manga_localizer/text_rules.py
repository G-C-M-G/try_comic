"""跨路径共用的文本归一化规则。

Photos 蓝框路径与 Paddle 检测路径、以及本地/在线两套翻译引擎都走这里，
规则只有一份（避免"同一规则各处各写一遍"的漂移）。
"""

from __future__ import annotations

import re

# 省略号的各种写法：U+2026（…，本身 3 点）、ASCII 点、全角点、日文句点连写。
# U+2026 一次即 3 点，出现两次就是日漫常见的 6 点写法。统一收敛成 3 点。
_ELLIPSIS_RUN = re.compile(r"(?:…+|[.．。]{3,})")


def normalize_ellipsis(text: str) -> str:
    """把省略号统一为 3 点（"…"）。

    日漫里 "……"（6 点）与 "．．．" 很常见，中文排版按 3 点。任何 3 个以上的
    点状字符连写（含两次及以上 U+2026）都收敛成单个 "…"；2 点不改（避免把
    本来 2 点写成 3 点）。
    """
    if not text:
        return text or ""
    return _ELLIPSIS_RUN.sub("…", text)


# 短拟声/语气词：纯假名/符号、长度很短。两处共用同一判据：
# 1) OCR 侧 —— manga-ocr 遇到风格化写法会误读成常见疑问词（实测 ぎゅっ♡ →
#    どうし / どんな / どういう，再被翻成"为什么/什么样"，语义全错），
#    回退读数命中这个判据就判定不可信；
# 2) 术语侧 —— 命中这个判据的原文才有资格作为拟声词自动进术语库。
_SFX_LIKE = re.compile(
    r"^[぀-ヿ々〆ーゝゞ"
    r"…、。.！？!?♡♥★☆"
    r"～〜→\s]{1,6}$"
)


def is_sfx_like(text: str) -> bool:
    """是否"短拟声/语气词"——纯假名/符号且 ≤6 字符。

    竖排对白、含汉字的台词一律不算，所以正常台词不会被误判成拟声词。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(_SFX_LIKE.match(stripped))


def suspicious_short_kana(text: str) -> bool:
    """读数是否"可疑的短假名"（= ``is_sfx_like``，OCR 侧命名）。

    用于 AI OCR 失败回退到 manga-ocr 的那些框：两个引擎都没把握，与其把
    `ぎゅっ♡` 译成"为什么"，不如判定不可信、保留原文（不擦不嵌）。
    """
    return is_sfx_like(text)
