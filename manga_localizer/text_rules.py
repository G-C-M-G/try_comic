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
