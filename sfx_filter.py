"""拟声词(SFX)智能过滤算法

规则（按优先级）:
1. 判断文字是否为拟声词: 纯假名/片假名+符号，不含汉字和字母数字
2. 若拟声词总数 > SFX_MAX_COUNT(10): 全部拟声词不翻译（保留原文）
3. 否则: 气泡内的拟声词（周围白色像素占比高）→ 翻译；气泡外（画面上）→ 不翻译
"""
import re
import logging

import cv2
import numpy as np

logger = logging.getLogger("manga_translator.sfx")

SFX_MAX_COUNT = 10
# 纯拟声词字符集: 平假名、片假名、长音、促音、小写假名、注音点、符号
_SFX_CHARS = (
    r"\u3041-\u30ff"       # 平假名+片假名
    r"\u309b-\u309c"       # 浊点
    r"ー〜～っゃゅょぁぃぅぇぉゎゕゖ・"
    r"♥♡☆★♪♫♬!！?？、。…～"
    r" \-"
    r"\s"
)

_SFX_RE = re.compile(f"^[{_SFX_CHARS}]+$")


def is_sfx(text: str) -> bool:
    """判断文本是否为拟声词（纯假名无汉字，一般不超过3个字）"""
    text = (text or "").strip()
    if not text:
        return False
    if len(text) > 3:  # 拟声词一般不超过三个假名
        return False
    return bool(_SFX_RE.fullmatch(text))


def is_moan_like(text: str) -> bool:
    """判断是否像呻吟/喘息类拟声词（用于决定未收录词典时是否恢复原文）"""
    t = _STRIP_RE.sub("", text or "")
    if not t or len(t) > 6:
        return False
    if any(c in t for c in "んっッふ"):   # あんっ/うん/ふふ/はぁん/ステッ
        return True
    if "にゃ" in t or "みゃ" in t:
        return True
    if re.search(r"(.)\1", t):              # 连续重复: はぁぁ/ふふ/ええ
        return True
    if len(t) <= 2 and t[-1] in "ー〜":     # 长音结尾: う〜/あー
        return True
    if t in ("あん", "うん", "ああ", "ええ", "うう"):
        return True
    return False


def _white_ratio(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 white_threshold: int = 200) -> float:
    """计算矩形区域内白色像素占比"""
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return 0.0
    return float((gray >= white_threshold).mean())


def in_bubble(img_bgr: np.ndarray, region, margin_ratio: float = 0.5,
              white_threshold: int = 190, min_white_ratio: float = 0.5) -> bool:
    """判断文字是否位于白色气泡内: 取文字框外扩 margin 的矩形，白色占比 >= 阈值
    注: 漫画气泡多为米白/粉白（亮度 185~205），阈值取 190 比纯白 200+ 更可靠"""
    x1, y1, x2, y2 = region.xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    margin = max(bw, bh) * margin_ratio
    ratio = _white_ratio(img_bgr, int(x1 - margin), int(y1 - margin),
                         int(x2 + margin), int(y2 + margin), white_threshold)
    return ratio >= min_white_ratio


def make_sfx_hook(sfx_max_count: int = SFX_MAX_COUNT, enable: bool = True, report_path: str = None,
                  white_threshold: int = 190, min_white_ratio: float = 0.5):
    """生成 MITS pre_translate_hook。返回 None 表示不启用过滤。"""
    if not enable:
        return None

    report_lines = []

    def _report(line: str):
        logger.info(f"[SFX] {line}")
        report_lines.append(line)

    def hook(ctx):
        img_bgr = ctx.img_rgb  # BGR numpy array
        regions = ctx.text_regions
        report_lines.clear()

        sfx_regions = [r for r in regions if is_sfx(r.text)]
        if not sfx_regions:
            return None

        _report(f"检测到 {len(sfx_regions)} 处拟声词: {[r.text for r in sfx_regions]}")

        if len(sfx_regions) > sfx_max_count:
            _report(f"拟声词数量({len(sfx_regions)})超过阈值({sfx_max_count})，全部不翻译")
            keep = [r for r in regions if r not in sfx_regions]
            for r in sfx_regions:
                _report(f"保留原文: {r.text}")
            return keep

        keep = []
        skipped = []
        translated = []
        for r in regions:
            if r in sfx_regions:
                bubble = in_bubble(img_bgr, r, white_threshold=white_threshold,
                                   min_white_ratio=min_white_ratio)
                _report(f"拟声词 [{r.text}] 位置{r.xyxy.tolist()} 气泡内={bubble}")
                if bubble:
                    translated.append(r.text)
                    keep.append(r)
                else:
                    skipped.append(r.text)
            else:
                keep.append(r)

        if skipped:
            _report(f"气泡外拟声词保留原文: {skipped}")
        if translated:
            _report(f"气泡内拟声词翻译: {translated}")

        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(report_lines))
            except Exception as e:
                logger.error(f"[SFX] 写报告失败: {e}")
        return keep

    return hook


# ============ 拟声词翻译词典（绕开 sakura 对呻吟类文本的幻觉乱码） ============

# 规范化: 去掉标点/长音/波浪线/空白，去除结尾促音(っ)与拗音小字差异
_STRIP_RE = re.compile(r"[。．…、，！？!?～〜ー\s.·]+")


def _normalize_sfx(text: str) -> str:
    t = _STRIP_RE.sub("", text or "")
    t = t.replace("っ", "").replace("ッ", "")
    t = t.replace("ぇ", "え").replace("ぃ", "い").replace("ぉ", "お").replace("ゃ", "や").replace("ゅ", "ゆ").replace("ょ", "よ")
    return t


# 规范化后的假名 -> 中文拟声
SFX_DICT = {
    "はぁ": "哈啊", "はあ": "哈啊", "はぁぁ": "哈啊", "はぁぁぁ": "哈啊",
    "はぁん": "哈啊嗯", "はあん": "哈啊嗯", "はぁい": "哈啊", "はあい": "哈啊",
    "あん": "啊嗯", "ああん": "啊嗯", "あああん": "啊嗯",
    "うん": "嗯", "ん": "嗯", "んん": "嗯嗯", "んんん": "嗯嗯嗯",
    "ふふ": "呵呵", "ふふふ": "呵呵", "うふふ": "呵呵呵",
    "あ": "啊", "あっ": "啊", "う": "呜", "うっ": "呜",
    "え": "诶", "えっ": "诶", "ええ": "诶诶",
    "にゃ": "喵", "にゃん": "喵", "みゃ": "喵",
    "くす": "嘿嘿", "くすくす": "嘿嘿嘿",
    "ふぅ": "呼", "はぁぁぁん": "哈啊～",
    "せんせ": "",  # 占位防止误匹配（不应出现）
}

_SFX_NORM_DICT = {_normalize_sfx(k): v for k, v in SFX_DICT.items() if v}


def lookup_sfx_translation(text: str):
    """查询拟声词词典翻译；查不到返回 None"""
    key = _normalize_sfx(text)
    if key in _SFX_NORM_DICT:
        return _SFX_NORM_DICT[key]
    return None


def make_sfx_post_hook(enable: bool = True, report_path: str = None):
    """翻译完成后修正拟声词译文: 词典命中→用词典译文；未命中→保留原文(移除区域)"""
    if not enable:
        return None

    report_lines = []

    def _report(line: str):
        logger.info(f"[SFX] {line}")
        report_lines.append(line)

    def hook(ctx):
        regions = ctx.text_regions
        report_lines.clear()
        if not regions:
            return None

        keep = []
        fixed = []
        reverted = []
        for r in regions:
            if not is_sfx(r.text):
                keep.append(r)
                continue
            trans = lookup_sfx_translation(r.text)
            if trans:
                r.translation = trans
                fixed.append(f"{r.text} => {trans}")
                keep.append(r)
            elif is_moan_like(r.text):
                # 呻吟类拟声词且词典未收录: 保留原文（移除后不擦字不嵌字）
                reverted.append(r.text)
            else:
                # 纯假名但非呻吟类（如"この"）: 保留 sakura 翻译
                keep.append(r)
        if fixed:
            _report(f"拟声词词典修正: {fixed}")
        if reverted:
            _report(f"拟声词未收录保留原文: {reverted}")

        if report_path:
            try:
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(report_lines) + "\n")
            except Exception:
                pass
        return keep

    return hook
