"""人名音译: 无法翻译的日文（多为片假名人名）音译为中文

规则: 翻译失败（译文为空 或 与原文相同）且文本含片假名时，按片假名音节
映射为常用中文音译字（阿/伊/卡/塔…）。
"""
import logging

logger = logging.getLogger("manga_translator.name")

# 单音节
_K1 = {
    "ア": "阿", "イ": "伊", "ウ": "乌", "エ": "埃", "オ": "奥",
    "カ": "卡", "キ": "奇", "ク": "库", "ケ": "凯", "コ": "科",
    "サ": "萨", "シ": "希", "ス": "苏", "セ": "塞", "ソ": "索",
    "タ": "塔", "チ": "琪", "ツ": "茨", "テ": "特", "ト": "托",
    "ナ": "娜", "ニ": "妮", "ヌ": "奴", "ネ": "内", "ノ": "诺",
    "ハ": "哈", "ヒ": "希", "フ": "芙", "ヘ": "赫", "ホ": "霍",
    "マ": "玛", "ミ": "米", "ム": "姆", "メ": "梅", "モ": "莫",
    "ヤ": "雅", "ユ": "优", "ヨ": "约",
    "ラ": "拉", "リ": "莉", "ル": "露", "レ": "蕾", "ロ": "罗",
    "ワ": "瓦", "ヲ": "奥", "ン": "恩",
    "ガ": "伽", "ギ": "琪", "グ": "古", "ゲ": "格", "ゴ": "戈",
    "ザ": "扎", "ジ": "吉", "ズ": "祖", "ゼ": "泽", "ゾ": "佐",
    "ダ": "达", "ヂ": "吉", "ヅ": "祖", "デ": "迪", "ド": "多",
    "バ": "巴", "ビ": "比", "ブ": "布", "ベ": "贝", "ボ": "波",
    "パ": "帕", "ピ": "皮", "プ": "普", "ペ": "佩", "ポ": "珀",
    "ヴ": "薇",
    "ァ": "阿", "ィ": "伊", "ゥ": "乌", "ェ": "埃", "ォ": "奥",
    "ャ": "雅", "ュ": "优", "ョ": "约",
}

# 双音节（拗音/长元音组合）
_K2 = {
    "キャ": "琪亚", "キュ": "琪优", "キョ": "琪约",
    "シャ": "夏", "シュ": "秀", "ショ": "晓",
    "チャ": "查", "チュ": "秋", "チョ": "卓",
    "ニャ": "妮亚", "ニュ": "妞", "ニョ": "妮约",
    "ヒャ": "希亚", "ヒュ": "秀", "ヒョ": "晓",
    "ミャ": "米亚", "ミュ": "缪", "ミョ": "妙",
    "リャ": "莉亚", "リュ": "琉", "リョ": "良",
    "ギャ": "琪亚", "ギュ": "玖", "ギョ": "尧",
    "ジャ": "嘉", "ジュ": "珠", "ジョ": "昭",
    "ビャ": "比娅", "ビュ": "标", "ビョ": "表",
    "ピャ": "皮娅", "ピュ": "飘", "ピョ": "瓢",
    "ファ": "法", "フィ": "菲", "フェ": "菲", "フォ": "佛",
    "ティ": "蒂", "ディ": "迪", "ドゥ": "杜",
    "ヴァ": "娃", "ヴィ": "薇", "ヴェ": "维", "ヴォ": "沃",
    "シェ": "谢", "ジェ": "杰", "チェ": "切", "ツェ": "泽",
    "デュ": "迪尤", "フュ": "菲尤", "ヴュ": "维尤",
}

# 三音节
_K3 = {
    "キャ": "琪亚",  # 冗余但保持兼容
}


def katakana_to_chinese(text: str) -> str:
    """片假名 → 中文音译"""
    out = []
    i = 0
    s = (text or "").strip()
    while i < len(s):
        c = s[i]
        if c in "ーッ・":
            i += 1
            continue
        if i + 2 < len(s) and s[i:i + 3] in _K3:
            out.append(_K3[s[i:i + 3]])
            i += 3
            continue
        if i + 1 < len(s) and s[i:i + 2] in _K2:
            out.append(_K2[s[i:i + 2]])
            i += 2
            continue
        if c in _K1:
            out.append(_K1[c])
            i += 1
            continue
        # 非片假名（汉字/英文等）原样保留
        out.append(c)
        i += 1
    return "".join(out)


def contains_katakana(text: str) -> bool:
    return any("\u30a0" <= c <= "\u30ff" for c in (text or ""))


def make_transliteration_hook(enable: bool = True, report_path: str = None, memory=None):
    """翻译失败且含片假名的文本 → 先查翻译记忆，未命中再音译为人名"""
    if not enable:
        return None

    report_lines = []

    def _report(line: str):
        logger.info(f"[音译] {line}")
        report_lines.append(line)

    def hook(ctx):
        report_lines.clear()
        done = []
        for region in ctx.text_regions:
            text = (region.text or "").strip()
            trans = (region.translation or "").strip()
            if not text:
                continue
            failed = (not trans) or (text.lower() == trans.lower())
            if not failed:
                continue
            # 记忆优先（含模糊匹配），其次音译
            if memory is not None:
                dst = memory.get(text) or memory.get_fuzzy(text)
                if dst:
                    region.translation = dst
                    done.append(f"{text} => {dst} (记忆)")
                    continue
            if contains_katakana(text):
                region.translation = katakana_to_chinese(text)
                done.append(f"{text} => {region.translation}")

        if done:
            _report(f"无法翻译的日文按人名音译: {done}")

        if report_path and report_lines:
            try:
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(report_lines) + "\n")
            except Exception:
                pass

    return hook
