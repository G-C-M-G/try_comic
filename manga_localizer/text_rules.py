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


def short_source(text: str, max_chars: int = 6) -> bool:
    """原文是否"很短"（非空白字符数 ≤ ``max_chars``）。

    短原文的字号必须封顶：漫画里 2~3 个字的台词，蓝框却可能很大（原文那几个
    字本来就写得大），字号跟着框走会嵌出一个 80~90px 的中文短句（实测
    07_6/island_014 = 2 字撑到 88px），与整页文字明显失衡。
    """
    if max_chars is None or int(max_chars) <= 0:
        return False          # 0 或负数 = 关闭该规则
    stripped = (text or "").strip()
    if not stripped:
        return False
    return len([ch for ch in stripped if not ch.isspace()]) <= int(max_chars)


def suspicious_short_kana(text: str) -> bool:
    """读数是否"可疑的短假名"（= ``is_sfx_like``，OCR 侧命名）。

    用于 AI OCR 失败回退到 manga-ocr 的那些框：两个引擎都没把握，与其把
    `ぎゅっ♡` 译成"为什么"，不如判定不可信、保留原文（不擦不嵌）。
    """
    return is_sfx_like(text)


# AI 视觉 OCR 的"说明性输出"：模型没读到字时会给一句描述/道歉，而不是空。
# 实测拿到的原文级样本：'图片中没有任何日文文本。它看起来像是…'、
# '图片中没有文字。' —— 这些被当成读数就会一路翻成中文，塞进原来的小框
# （实测页 8 / 页 14 各一个岛，70 字塞进 96×57 的框 → 12px 兜底、几乎看不见）。
# 判据只看"像不像在描述图片"，不猜内容，命中就当读数空处理（回退 manga-ocr，
# 都读不出就按假选框丢弃、保留原图）。
_COMMENTARY_PATTERNS = (
    re.compile(r"(图片|图中|该图|此图|这张图|图像)[^。！？]{0,12}(没有|无|不存在|看不到|未发现)"),
    re.compile(r"(没有|无|不存在)(任何)?(日文|日语|文字|文本|内容)"),
    re.compile(r"(无法|不能|难以)(辨认|识别|读取|看清)"),
    re.compile(r"(以下是|图中写着|图片中写着|识别结果[:：])"),
    re.compile(r"no\s+(japanese\s+)?text", re.IGNORECASE),
    re.compile(r"there\s+is\s+no", re.IGNORECASE),
)


def is_ocr_commentary(text: str) -> bool:
    """读数是否 AI OCR 的说明性输出（"图里没有文字"这类），而不是真读数。"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _COMMENTARY_PATTERNS)
