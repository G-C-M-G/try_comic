"""拟声词（SFX）自动收录进术语库。

背景：拟声词是全书一致性最差的一类文本。同一个 ``ぎゅっ♡`` 在白话 AI 眼里
每次都可能换一个说法（夹紧♡ / 夹住♡ / 挤♡），而且它是 AI OCR 最吃力的一类
字（风格化手绘），读错一次就会被译成完全不相关的词。

做法：把**已经译好、且译法稳定**的拟声词回灌进 ``config/glossary.json``，
下一轮（含全书重跑）所有页面都用同一译法。四个护栏保证"只有准确的才进库"：

1. 原文必须是 ``is_sfx_like``（纯假名/符号 ≤6 字）——不会把正常台词收进来；
2. **原文不能整条都是功能词/标点**（见 ``is_function_word_only``）；
3. 译文必须短（≤6 字）且不在"疑问词黑名单"里——``ぎゅっ♡`` 被读成
   ``どうし`` 再译成"为什么"这类失败样本，永远不会被收进术语库；
4. 译法必须**一致且重复出现**（同一原文至少 2 次、且只有一种译法）。
   一旦同一原文出现第二种译法，说明 AI 自己都没把握，直接撤出术语库，
   宁可让它每次现译，也不要锁死一个可能是错的译法。

用户手写的条目（note 不以自动标记开头）永不被覆盖或删除。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .text_rules import is_sfx_like

# 自动收录条目的 note 前缀（用于识别"这条是自动写的"，以及展示来源）
AUTO_NOTE_PREFIX = "自动收录"

# 译文长度上限：拟声词译文超过这个长度就不是拟声词了（AI 在编句子）
MAX_TARGET_CHARS = 6

# 功能词/标点的词性（fugashi 的 pos1）。**整条都属于这个集合**才算功能词。
# * 含 ``接続詞``：孤立的一个 `が` / `けど` / `じゃ` 会被标成接続詞，而它们作为
#   术语表键是有害的（`が` 出现在大量句子里）。
# * 刻意**不含** ``記号``：感叹词（`あ` `ち` `ぷ` `オ`）被标成 記号，而它们正是
#   合理的拟声/语气收录对象。
# * 已知残留缺口：``や``（形状詞）与 ``わ``（記号）和感叹词同词性，POS 分不开，
#   偶尔会漏进来。要彻底解决得按语料统计（"出现在超过 X% 的岛里"）而不是词性。
_FUNCTION_POS = frozenset({"助詞", "助動詞", "補助記号", "接続詞"})

_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        from fugashi import Tagger

        _TAGGER = Tagger()
    return _TAGGER


def is_function_word_only(text: str) -> bool:
    """整条都是助词/助动词/标点 → 不是拟声词，不能进术语库。

    `is_sfx_like` 只看"纯假名/符号且 ≤6 字"，**分不开助词和拟声**——两者都是
    短假名串。实测（2026-09-17 从 AI 读数换回 manga-ocr 之后）术语库被塞进
    ``の -> 的``、``は -> 哈``、``ン -> 嗯``：而 ``の`` 几乎出现在每个日文句子里，
    于是**几乎每个岛的提示词**都会被注入"の 译作 的"——这是系统性扭曲译文，
    比漏收几个拟声词严重得多。所以补一道词性闸。

    fugashi 缺失时返回 ``False``（不加闸，维持原行为——这道闸是收紧，不是必需）。
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    try:
        tokens = list(_tagger()(cleaned))
    except Exception:                      # noqa: BLE001 - 词性不可用就不加闸
        return False
    if not tokens:
        return True
    return all(
        str(getattr(token.feature, "pos1", "") or "") in _FUNCTION_POS
        for token in tokens
    )

# 疑问词黑名单：manga-ocr 把风格化拟声词读成 どうし/どんな/どういう 之后，
# AI 会照字面译成这些词。命中说明原文本身就是错读数，绝不能进术语库。
SUSPECT_TARGETS = frozenset({
    "为什么", "为什么？", "什么样", "什么", "怎么会", "怎么", "怎么样",
    "怎样", "为何", "为啥", "什么情况", "为什么?", "什么样？",
})


def _target_chars(text: str) -> int:
    return len([char for char in text if not char.isspace()])


def empty_report() -> dict[str, list[str]]:
    """空报告——调用方可以无条件按同一形状读取结果（不用先判空）。"""
    return {"added": [], "updated": [], "removed": [], "pending": []}


def collect_observations(
    texts_per_page: dict[str, dict[str, str]],
    translations_per_page: dict[str, dict[str, str]],
) -> dict[str, dict[str, int]]:
    """从本次翻译结果里挑出候选拟声词：``{原文: {译文: 次数}}``。

    只看通过护栏 1、2、3 的配对；跨页累计次数，所以同一拟声词出现两次以上
    才可能入库。
    """
    observations: dict[str, dict[str, int]] = {}
    for page, source_map in texts_per_page.items():
        target_map = translations_per_page.get(page) or {}
        for island_id, source in source_map.items():
            target = str(target_map.get(island_id) or "").strip()
            if not target or "\n" in target:
                continue
            if not is_sfx_like(source):
                continue
            if is_function_word_only(source):
                continue
            if _target_chars(target) > MAX_TARGET_CHARS:
                continue
            if target in SUSPECT_TARGETS:
                continue
            bucket = observations.setdefault(source.strip(), {})
            bucket[target] = bucket.get(target, 0) + 1
    return observations


def sync_sfx_terms(
    path: Path,
    observations: dict[str, dict[str, int]],
    *,
    min_count: int = 2,
) -> dict[str, list[str]]:
    """把稳定的拟声词译法合并进术语库；返回 ``{"added","updated","removed","pending"}``。

    ``min_count`` 是"同一原文 + 同一译法"至少出现的次数（0 表示关闭自动收录）。
    冲突（同一原文出现多种译法）时撤出自动条目；手写条目不动。

    顺带**自愈**：已入库的自动条目里，若原文其实是功能词/纯标点（历史遗留，
    见 ``is_function_word_only``），一并撤出——否则旧库里那些 ``の -> 的``
    永远留在那里继续污染提示词。
    """
    report: dict[str, list[str]] = empty_report()
    path = Path(path)
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return report
        if not isinstance(data, dict):
            return report
    terms = data.get("terms")
    if not isinstance(terms, dict):
        terms = {}
        data["terms"] = terms

    changed = _drop_function_word_entries(terms, report)
    if min_count <= 0 or not observations:
        if changed:
            _write_atomic(path, data)
        return report
    for source, targets in sorted(observations.items()):
        existing = terms.get(source)
        is_auto = isinstance(existing, dict) and str(
            existing.get("note") or ""
        ).startswith(AUTO_NOTE_PREFIX)
        if isinstance(existing, dict) and not is_auto:
            continue  # 手写条目：永不覆盖
        if len(targets) > 1:
            # AI 自己都不一致 → 不锁死译法；已收录的自动条目撤出
            if is_auto:
                del terms[source]
                report["removed"].append(source)
                changed = True
            continue
        target, count = next(iter(targets.items()))
        if count < min_count:
            report["pending"].append(source)
            continue
        note = f"{AUTO_NOTE_PREFIX}·拟声词（重复 {count} 次）"
        if is_auto:
            if str(existing.get("translation") or "") == target:
                continue
            existing["translation"] = target
            existing["note"] = note
            report["updated"].append(source)
        else:
            terms[source] = {"translation": target, "note": note}
            report["added"].append(source)
        changed = True

    if changed:
        _write_atomic(path, data)
    return report


def _drop_function_word_entries(terms: dict, report: dict) -> bool:
    """撤出"原文其实是功能词/纯标点"的**自动**条目（手写条目不动）。"""
    stale = [
        source for source in list(terms)
        if isinstance(terms.get(source), dict)
        and str(terms[source].get("note") or "").startswith(AUTO_NOTE_PREFIX)
        and is_function_word_only(source)
    ]
    for source in stale:
        del terms[source]
        report["removed"].append(source)
    return bool(stale)


def _write_atomic(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
