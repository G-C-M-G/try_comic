"""手动调整状态: 保存/加载 每页的 擦字图 + 文本框数据 (JSON)

注: 图片读写用 PIL（cv2.imwrite/imread 不支持中文路径）
"""
import json
import os

import numpy as np
from PIL import Image

from manga_translator.utils import TextBlock


def region_to_dict(r: TextBlock) -> dict:
    line_groups = r.lines.reshape(-1, 4, 2)
    source_texts = list(getattr(r, "texts", None) or [])
    d = {
        "text": r.text,
        "translation": r.translation,
        "lines": [[int(x), int(y)] for pt in line_groups for x, y in pt],
        "angle": float(r.angle),
        "font_size": int(r.font_size),
        "fg": [int(c) for c in r.fg_colors],
        "bg": [int(c) for c in r.bg_colors],
        "direction": r.direction,
        "alignment": r.alignment,
    }
    if len(line_groups) == len(source_texts):
        d["source_segments"] = [
            {
                "text": source_texts[i],
                "lines": [[int(x), int(y)] for x, y in line_groups[i]],
            }
            for i in range(len(line_groups))
        ]
    backend = getattr(r, "ocr_backend", None)
    if backend:
        d["ocr_backend"] = str(backend)
    confidence = getattr(r, "ocr_confidence", None)
    if confidence is not None:
        d["ocr_confidence"] = float(confidence)
    score_proxy = getattr(r, "ocr_score_proxy", None)
    if score_proxy is not None:
        d["ocr_score_proxy"] = float(score_proxy)
    translation_segments = getattr(r, "translation_segments", None)
    if translation_segments:
        d["translation_segments"] = list(translation_segments)
    # PS 式文字样式字段（可选，缺省=旧行为）
    for k, default in (
        ("font", None), ("font_bold", False), ("stroke_w", None),
        ("line_spacing", None), ("char_spacing", 0),
        ("align_h", "center"), ("align_v", "center"),
        ("force_dir", "auto"), ("fill_bg", False),
    ):
        v = getattr(r, k, None)
        if v is not None and v != default:
            d[k] = v
    if getattr(r, "fill_bg", False):
        d["fill_bg_color"] = [int(c) for c in getattr(r, "fill_bg_color", (255, 255, 255))]
    return d


def region_from_dict(d: dict) -> TextBlock:
    source_segments = d.get("source_segments") or []
    segment_lines = [s.get("lines") for s in source_segments if s.get("lines")]
    raw_lines = segment_lines or d.get("lines")
    pts = np.array(raw_lines, dtype=np.int32).reshape(-1, 4, 2)
    source_texts = [s.get("text", "") for s in source_segments]
    texts = source_texts if len(source_texts) == len(pts) else [d.get("text", "")]
    fg = d.get("fg") or [0, 0, 0]
    bg = d.get("bg") or [255, 255, 255]
    r = TextBlock(
        lines=pts,
        texts=texts,
        translation=d.get("translation", ""),
        angle=float(d.get("angle", 0)),
        font_size=int(d.get("font_size", 22)),
        fg_color=fg,
        bg_color=bg,
        direction=d.get("direction", "auto"),
        alignment=d.get("alignment", "auto"),
        prob=float(d.get("ocr_score_proxy", d.get("ocr_confidence", d.get("ocr_prob", 1.0))) or 1.0),
    )
    r.target_lang = "CHS"
    # PS 式文字样式字段（附加到对象，缺省 None=默认行为）
    for k in ("font", "font_bold", "stroke_w", "line_spacing", "char_spacing",
              "align_h", "align_v", "force_dir", "fill_bg", "fill_bg_color"):
        if k in d:
            setattr(r, k, d[k])
    if d.get("translation_segments"):
        r.translation_segments = list(d["translation_segments"])
    for key in ("ocr_backend", "ocr_confidence", "ocr_score_proxy"):
        if key in d:
            setattr(r, key, d[key])
    return r


def save_page_state(out_dir: str, page_name: str, img_inpainted: np.ndarray,
                    regions, img_rgb: np.ndarray = None, src_ext: str = ""):
    """保存单页调整状态: {page}.json + {page}_inpainted.png (+ {page}_orig.png)
    src_ext: 原图扩展名（如 .jpg），供重嵌输出"与原图相同"格式"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, page_name)
    Image.fromarray(img_inpainted).save(base + "_inpainted.png")
    if img_rgb is not None:
        Image.fromarray(img_rgb).save(base + "_orig.png")
    data = {
        "page": page_name,
        "src_ext": (src_ext or "").lower(),
        "regions": [region_to_dict(r) for r in regions],
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_src_ext(json_path: str) -> str:
    """读取状态 JSON 中记录的原图扩展名（无则返回空）"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return (json.load(f).get("src_ext") or "").lower()
    except Exception:
        return ""


def load_page_state(json_path: str):
    """读取调整状态; 返回 (inpainted_img_rgb, orig_img_rgb or None, regions)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.splitext(json_path)[0]
    inpainted = np.array(Image.open(base + "_inpainted.png").convert("RGB"))
    orig = None
    if os.path.exists(base + "_orig.png"):
        orig = np.array(Image.open(base + "_orig.png").convert("RGB"))
    regions = [region_from_dict(d) for d in data.get("regions", [])]
    return inpainted, orig, regions
