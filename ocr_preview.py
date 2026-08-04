from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
)

SOURCE_COLORS = {
    "cross-model-agreement": "#009E73",
    "paddleocr-supplement": "#0072B2",
    "cross-model-disagreement": "#D55E00",
}

PADDLE_GEOMETRY_COLOR = "#0072B2"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if current and draw.textlength(trial, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def sorted_columns(data: dict) -> list[dict]:
    return sorted(data.get("columns", []), key=lambda item: item.get("selection_order", 0))


def source_color(column: dict) -> str:
    if column.get("review_required"):
        return "#D55E00"
    return SOURCE_COLORS.get(column.get("text_source", ""), "#CC79A7")


def render_preview(data: dict, output_path: Path) -> None:
    source_path = Path(data["source"]["path"])
    source = Image.open(source_path).convert("RGB")
    columns = sorted_columns(data)

    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    marks = ImageDraw.Draw(overlay)
    label_font = load_font(max(16, round(source.width / 55)))
    review_queue = {
        item.get("column_id"): item for item in data.get("manual_review_queue", [])
    }

    for number, column in enumerate(columns, start=1):
        review_item = review_queue.get(column.get("id"), {})
        paddle_polygon = [
            tuple(point) for point in review_item.get("paddle_polygon", []) or []
        ]
        if "paddle-box-extends-current-region" in review_item.get("reason_codes", []):
            if len(paddle_polygon) < 3 and review_item.get("paddle_image_bbox"):
                px1, py1, px2, py2 = review_item["paddle_image_bbox"]
                paddle_polygon = [(px1, py1), (px2, py1), (px2, py2), (px1, py2)]
            if len(paddle_polygon) >= 3:
                marks.line(
                    paddle_polygon + [paddle_polygon[0]],
                    fill=PADDLE_GEOMETRY_COLOR,
                    width=4,
                    joint="curve",
                )
        polygon = [tuple(point) for point in column.get("polygon", [])]
        if len(polygon) < 3:
            x1, y1, x2, y2 = column["image_bbox"]
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        color = source_color(column)
        marks.polygon(polygon, fill=color + "24", outline=color, width=3)

        x1, y1, _, _ = column["image_bbox"]
        label = f"{number:02d}"
        box = marks.textbbox((0, 0), label, font=label_font, stroke_width=1)
        label_width = box[2] - box[0] + 8
        label_height = box[3] - box[1] + 6
        label_x = max(0, min(x1, source.width - label_width))
        label_y = max(0, y1 - label_height)
        marks.rounded_rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            radius=3,
            fill=color,
        )
        marks.text(
            (label_x + 4, label_y + 2),
            label,
            font=label_font,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )

    annotated = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
    panel_width = max(700, round(source.width * 0.72))
    canvas = Image.new("RGB", (source.width + panel_width, source.height), "#F6F7F8")
    canvas.paste(annotated, (0, 0))
    draw = ImageDraw.Draw(canvas)

    page_name = source_path.stem
    title_font = load_font(max(26, round(source.width / 32)))
    body_font = load_font(22 if len(columns) <= 20 else 18)
    small_font = load_font(17)
    x0 = source.width + 28
    y = 24
    draw.text((x0, y), f"第 {page_name} 页 - OCR 识别列", font=title_font, fill="#181A1B")
    y += title_font.size + 14 if hasattr(title_font, "size") else 50

    summary = data.get("fusion_summary", {})
    review = data.get("manual_review", {})
    agreement = summary.get("cross_validated_columns", 0)
    draw.text(
        (x0, y),
        f"识别 {len(columns)} 列 | 双模型定位确认 {agreement} | "
        f"待核对 {review.get('pending_count', 0)} | 已绕过 {review.get('bypassed_count', 0)}",
        font=small_font,
        fill="#4A4D50",
    )
    y += 38

    legend_items = (
        ("#009E73", "双模型文字一致"),
        ("#0072B2", "Paddle 完整框候选"),
        ("#D55E00", "待人工核对"),
        ("#CC79A7", "区域确认，采用单模型文字"),
    )
    legend_width = (panel_width - 56) // 2
    for index, (color, label) in enumerate(legend_items):
        key_x = x0 + (index % 2) * legend_width
        key_y = y + (index // 2) * 28
        draw.rectangle((key_x, key_y + 4, key_x + 16, key_y + 20), fill=color)
        draw.text((key_x + 22, key_y), label, font=small_font, fill="#34373A")
    y += 68

    list_columns = 2 if len(columns) > 20 else 1
    rows = math.ceil(len(columns) / list_columns)
    list_width = (panel_width - 56) // list_columns
    available_height = source.height - y - 24
    row_height = max(34, available_height // max(rows, 1))

    for index, column in enumerate(columns):
        list_column = index // rows
        row = index % rows
        entry_x = x0 + list_column * list_width
        entry_y = y + row * row_height
        color = source_color(column)
        number = f"{index + 1:02d}"
        draw.rounded_rectangle(
            (entry_x, entry_y + 2, entry_x + 36, entry_y + 29),
            radius=3,
            fill=color,
        )
        draw.text(
            (entry_x + 5, entry_y + 3),
            number,
            font=small_font,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
        text_x = entry_x + 46
        max_width = list_width - 54
        display_text = column.get("text", "")
        if column.get("review_required"):
            reasons = review_queue.get(column.get("id"), {}).get("reason_codes", [])
            display_text += (
                "  [文字/框待核对]"
                if "paddle-box-extends-current-region" in reasons
                else "  [待核对]"
            )
        lines = wrap_text(draw, display_text, body_font, max_width)
        line_height = max(22, getattr(body_font, "size", 18) + 3)
        for line_index, line in enumerate(lines[:2]):
            draw.text(
                (text_x, entry_y + line_index * line_height),
                line,
                font=body_font,
                fill="#111315",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def write_report(items: list[tuple[Path, dict]], output_path: Path) -> None:
    lines = ["# OCR result", ""]
    for json_path, data in items:
        page = Path(data["source"]["path"]).stem
        columns = sorted_columns(data)
        lines.extend((f"## Page {page}", "", "### Columns", ""))
        for index, column in enumerate(columns, start=1):
            sources = ", ".join(column.get("detection_sources", []))
            review_reasons = ", ".join(column.get("review_reasons", []))
            lines.append(
                f"{index:02d}. {column.get('text', '')}  "
                f"`{column.get('id', '')}; {column.get('text_source', '')}; {sources}; "
                f"review={column.get('review_status', 'unknown')}`"
            )
            if review_reasons:
                lines.append(f"    Review reasons: `{review_reasons}`")
        lines.extend(("", "### Grouped sentences", ""))
        for unit in data.get("translation_units", []):
            ids = ", ".join(unit.get("column_ids", []))
            lines.append(f"- {unit.get('source_text', '')}  `{ids}`")
        lines.extend(("", "### Manual review queue", ""))
        queue = data.get("manual_review_queue", [])
        if not queue:
            lines.append("- None")
        for item in queue:
            readings = " / ".join(item.get("mangaocr_readings", []))
            reasons = ", ".join(item.get("reason_codes", []))
            lines.append(
                f"- `{item.get('column_id')}` {item.get('current_text', '')} "
                f"status=`{item.get('status')}` reasons=`{reasons}` "
                f"MangaOCR=`{readings}` PaddleOCR=`{item.get('paddleocr_text', '')}` "
                f"current_box=`{item.get('image_bbox')}` "
                f"PaddleBox=`{item.get('paddle_image_bbox')}` "
                f"coverage=`{item.get('paddle_geometry_coverage')}`"
            )
        lines.extend(("", f"Source JSON: `{json_path}`", ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render OCR bounding boxes and recognized text.")
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ocr_review"))
    args = parser.parse_args()

    items: list[tuple[Path, dict]] = []
    for json_path in args.json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        items.append((json_path, data))
        page = Path(data["source"]["path"]).stem
        render_preview(data, args.output_dir / f"{page}_ocr_preview.png")
    write_report(items, args.output_dir / "OCR_result.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
