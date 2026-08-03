"""文本渲染工具: PIL 渲染译文文字（工作台预览与重新嵌字共用，保证所见即所得）

- 字体: 可指定（默认 msyh.ttc 微软雅黑），支持粗体变体（msyhbd 等，无变体用描边模拟）
- 横排: 按 max_w 换行，行距可调（默认 1.2×字号），字距可调
- 竖排: 按 max_h 分列，列序 RTL(默认右→左)/LTR，字距/行距可调
- 对齐: 横排 左/中/右（行内）+ 上下；竖排 列内 上/中/下 + 左右
- 实心底色: fill_bg 时在画布垫不透明色块（白底抹原文），描边独立可调宽
- 返回画布尺寸 = (max_w, max_h)，内容按对齐放置；调用方以框中心对齐粘贴
"""
import os
import re

from PIL import Image, ImageDraw, ImageFont

from fonts import resolve, default_font_name

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(ROOT, "manga-image-translator", "fonts", "msyh.ttc")

_font_cache = {}
_VERTICAL_ELLIPSIS_RE = re.compile(r"(?:…{2,}|[.．]{2,})")


def normalize_vertical_text(text: str) -> str:
    """Use the Unicode presentation form for ellipses in vertical layouts."""
    return _VERTICAL_ELLIPSIS_RE.sub("︙", str(text or ""))


def get_font(size: int, font_path: str = None, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (font_path or FONT_PATH, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(key[0], size)
    return _font_cache[key]


def _resolve_font_path(font_name: str, bold: bool) -> str:
    p = resolve(font_name or default_font_name(), bold)
    return p or FONT_PATH


def _measure_len(draw: ImageDraw.ImageDraw, text: str, font, char_spacing: int) -> int:
    """文本宽度（含字距，逐字测量）"""
    if not text:
        return 0
    return int(sum(font.getlength(ch) for ch in text)) + char_spacing * (len(text) - 1)


def render_text_image(text: str, font_size: int, vertical: bool,
                      max_w: int, max_h: int, fg=(0, 0, 0), bg=None,
                      rtl: bool = True, font_name: str = None, bold: bool = False,
                      stroke_width: int = None, line_spacing: float = 1.2,
                      char_spacing: int = 0,
                      align_h: str = "center", align_v: str = "center",
                      fill_bg: bool = False, fill_bg_color=(255, 255, 255)):
    """渲染文字到 RGBA 图像（尺寸=max_w×max_h）。
    fg=文字色, bg=描边色(None 时不描边), fill_bg_color=实心底色。
    align_h: left/center/right; align_v: top/center/bottom"""
    text = (text or "").strip()
    if not text or font_size < 6:
        return None
    max_w = max(max_w, 8)
    max_h = max(max_h, 8)
    base_p = resolve(font_name or default_font_name(), False) or FONT_PATH
    font_path = resolve(font_name or default_font_name(), True) if bold else base_p
    has_bold_variant = font_path != base_p
    font = get_font(font_size, font_path or base_p)
    line_h = max(1, int(font_size * line_spacing))
    stroke = int(stroke_width) if stroke_width is not None else max(1, int(font_size * 0.07))
    if bold and not has_bold_variant:
        stroke = max(stroke, int(font_size * 0.14))  # 无粗体变体时描边模拟加粗
    fill = tuple(fg) if fg else (0, 0, 0)
    stroke_fill = tuple(bg) if bg else fill

    # 画布（透明底）
    img = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if fill_bg:
        d.rectangle([0, 0, max_w - 1, max_h - 1], fill=tuple(fill_bg_color) + (255,))

    if vertical:
        text = normalize_vertical_text(text)
        _render_vertical(d, font, text, font_size, max_w, max_h, fill, stroke_fill, stroke,
                         rtl, line_h, char_spacing, align_h, align_v)
    else:
        _render_horizontal(d, font, text, max_w, max_h, fill, stroke_fill, stroke,
                           line_h, char_spacing, align_h, align_v)
    return img


def _has_bold_variant(font_path: str) -> bool:
    """是否已用粗体变体文件渲染（resolve 已处理）"""
    from fonts import resolve_path
    name = os.path.basename(font_path).lower()
    return any(k in name for k in ("bd", "bold")) or "msyhbd" in name


def _render_horizontal(d, font, text, max_w, max_h, fill, stroke_fill, stroke,
                       line_h, char_spacing, align_h, align_v):
    # 换行（考虑字距）
    max_w = max(max_w, font.size)
    lines, cur = [], ""
    for ch in text:
        if cur and _measure_len(d, cur + ch, font, char_spacing) > max_w:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    lines.append(cur)

    line_widths = [_measure_len(d, l, font, char_spacing) for l in lines]
    block_w = max(line_widths)
    block_h = len(lines) * line_h

    # 整体垂直位置
    if align_v == "top":
        y0 = 0
    elif align_v == "bottom":
        y0 = max_h - block_h
    else:
        y0 = max(0, (max_h - block_h) // 2)
    y0 = max(0, y0)

    for li, l in enumerate(lines):
        lw = line_widths[li]
        if align_h == "left":
            x = 0
        elif align_h == "right":
            x = max_w - lw
        else:
            x = max(0, (max_w - lw) // 2)
        y = y0 + li * line_h
        # 逐字绘制（Pillow 12 无 character_spacing，自绘实现字距）
        for ch in l:
            d.text((x, y), ch, font=font, fill=fill,
                   stroke_width=stroke, stroke_fill=stroke_fill)
            x += int(font.getlength(ch)) + char_spacing


def _render_vertical(d, font, text, font_size, max_w, max_h, fill, stroke_fill, stroke,
                     rtl, line_h, char_spacing, align_h, align_v):
    max_chars = max(1, max_h // line_h) if max_h > 0 else len(text)
    cols = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    col_w = int(font_size * 1.2)

    # 列组总宽（用于水平对齐）
    group_w = len(cols) * col_w
    if align_h == "left":
        gx = 0
    elif align_h == "right":
        gx = max_w - group_w
    else:
        gx = max(0, (max_w - group_w) // 2)

    for ci, col in enumerate(cols):
        col_h = len(col) * line_h
        if align_v == "top":
            y0 = 0
        elif align_v == "bottom":
            y0 = max_h - col_h
        else:
            y0 = max(0, (max_h - col_h) // 2)
        # rtl: 第一列在最右
        if rtl:
            x = gx + (len(cols) - 1 - ci) * col_w
        else:
            x = gx + ci * col_w
        x = x + font_size // 5
        step = line_h + char_spacing
        for yi, ch in enumerate(col):
            d.text((x, y0 + yi * step), ch, font=font, fill=fill,
                   stroke_width=stroke, stroke_fill=stroke_fill)
