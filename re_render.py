"""应用手动调整的文本框，重新嵌字渲染

与工作台预览共用同一套 PIL 渲染（text_paint），所见即所得：
- 按状态 JSON 中的字号以真实大小渲染（横排 1.2×字号行距 / 竖排 1.2×字号列距）
- 保留旋转(angle)：文字按检测角度倾斜，与气泡方向一致
- 文字居中于框中心；超过框的部分外溢不裁切（PPT 式，可自行调整框/字号）
- 输出格式可选，默认与原图相同（状态 JSON 中记录的 src_ext）

用法:
    python re_render.py <json或json所在目录> [输出图片或目录] [--format png|jpg|webp|auto] [--direction auto|ltr|rtl]
    - 单页:  python re_render.py "调整状态\181.json" "输出\181.png"
    - 整个目录: python re_render.py "调整状态" "输出目录"
    - --format: 输出格式，默认 auto（与原图相同，无记录则 png）
    - --direction: 竖排阅读方向，默认 auto（右→左）；ltr=左→右
"""
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "manga-image-translator"))

from manual_state import load_page_state, load_src_ext
from manga_translator.rendering import expand_translation_regions
from text_paint import render_text_image

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp")


def resolve_ext(json_path: str, fmt: str = "auto") -> str:
    """解析输出扩展名: fmt 指定则用之；auto 用状态记录的原图扩展名，无则 png"""
    if fmt and fmt.startswith("."):
        fmt = fmt[1:]
    if fmt and f".{fmt}" in SUPPORTED_EXT:
        return f".{fmt}"
    ext = load_src_ext(json_path)
    return ext if ext in SUPPORTED_EXT else ".png"


def rerender_page(json_path: str, out_path: str, fmt: str = "auto", direction: str = "auto"):
    inpainted, _orig, regions = load_page_state(json_path)
    if inpainted is None:
        print(f"缺少擦字图: {json_path}")
        return False
    img = Image.fromarray(inpainted).convert("RGB")
    # 阅读方向: ltr=竖排左→右; rtl/auto=竖排右→左（日漫阅读习惯）；横排不受影响
    rtl = direction != "ltr"

    for r in expand_translation_regions(regions):
        trans = (r.translation or "").strip()
        if not trans:
            continue
        fs = int(r.font_size)
        if fs <= 0:
            # 未记录字号时按框尺寸估算（竖排按列宽，横排按行高）
            w, h = r.unrotated_size
            fs = int((w * 0.9) if r.vertical else (h * 0.75))
        fs = max(6, fs)
        bw, bh = [int(v) for v in r.unrotated_size]
        fg, bg = r.get_font_colors()
        # PS 式样式字段（缺省=默认行为）
        font_name = getattr(r, "font", None) or None
        bold = bool(getattr(r, "font_bold", False))
        stroke_w = getattr(r, "stroke_w", None)
        line_spacing = getattr(r, "line_spacing", None) or 1.2
        char_spacing = int(getattr(r, "char_spacing", 0) or 0)
        align_h = getattr(r, "align_h", "center") or "center"
        align_v = getattr(r, "align_v", "center") or "center"
        fill_bg = bool(getattr(r, "fill_bg", False))
        fill_bg_color = tuple(int(c) for c in getattr(r, "fill_bg_color", (255, 255, 255)))
        # 强制方向: 覆盖框形自动判定
        force = getattr(r, "force_dir", "auto") or "auto"
        vertical = r.vertical
        if force == "h":
            vertical = False
        elif force == "v":
            vertical = True
        try:
            txt = render_text_image(trans, fs, vertical, bw, bh,
                                    tuple(int(c) for c in fg), tuple(int(c) for c in bg),
                                    rtl=rtl, font_name=font_name, bold=bold,
                                    stroke_width=stroke_w, line_spacing=line_spacing,
                                    char_spacing=char_spacing, align_h=align_h, align_v=align_v,
                                    fill_bg=fill_bg, fill_bg_color=fill_bg_color)
        except Exception as e:
            print(f"渲染失败 {trans}: {e}")
            continue
        if txt is None:
            continue
        if r.angle:
            txt = txt.rotate(-float(r.angle), expand=True, resample=Image.BICUBIC)
        cx, cy = r.center
        img.paste(txt, (int(cx - txt.width / 2), int(cy - txt.height / 2)), txt)

    ext = resolve_ext(json_path, fmt)
    if not out_path.lower().endswith(ext):
        out_path = os.path.splitext(out_path)[0] + ext
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    print(f"已生成: {out_path}")
    return True


def main():
    args = [a for a in sys.argv[1:]]
    fmt = "auto"
    direction = "auto"
    if "--format" in args:
        i = args.index("--format")
        fmt = args[i + 1] if len(args) > i + 1 else "auto"
        del args[i:i + 2]
    if "--direction" in args:
        i = args.index("--direction")
        direction = args[i + 1] if len(args) > i + 1 else "auto"
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(1)
    src, dst = args[0], (args[1] if len(args) > 1 else None)

    if os.path.isdir(src):
        jsons = sorted(glob.glob(os.path.join(src, "*.json")))
        if not jsons:
            print("目录中没有 .json 状态文件")
            sys.exit(1)
        out_dir = dst or os.path.join(src, "rerender")
        os.makedirs(out_dir, exist_ok=True)
        for j in jsons:
            name = os.path.splitext(os.path.basename(j))[0]
            rerender_page(j, os.path.join(out_dir, name + ".png"), fmt, direction)
        print(f"完成: {len(jsons)} 页 -> {out_dir}")
    else:
        if not dst:
            dst = os.path.splitext(src)[0] + "_rerender.png"
        rerender_page(src, dst, fmt, direction)


if __name__ == "__main__":
    main()
