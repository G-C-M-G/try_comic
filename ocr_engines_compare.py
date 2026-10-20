"""读字层多引擎对比 + **准确率**评分（补上"只有一致度、没有真值"这个缺口）。

为什么需要它：`baberu_compare.py` 量的是"两个引擎一不一致"，而一致性不等于正确——
开发记录 自己记了两例：manga-ocr 与 Baberu 共同把「闇」读成「間」；manga-ocr 还
幻觉出「１想像するに！！」。要比引擎高低，必须有一小批**人工校对过的**裁剪。

流程（两步，中间那步是人工的）：
  1. 抽样：`--sample 150` → 按"可疑读数"分层（不含假名汉字 / 只有汉字 / 单字 /
     含拉丁数字 / AI 说明性输出 / ≥30 字 / 同页重复）+ 随机补齐，裁剪存到
     `var/ocr_eval/{书}/crops/`，并生成 `sheet.csv` + `sheet.html`（HTML 里图直接
     内嵌，方便逐条填）。**裁剪配方与 stage2 完全一致**（bluebox_masked.png、
     image_bbox 扩 6px），所以"现引擎"那一列直接取 `ocr_readings.json`，不用重跑。
  2. 人工：在 `sheet.csv` 的"正确文本"列填每个裁剪的真实文本（看不清就填 `?`）。
  3. 评分：`--score` → 逐引擎算 CER / 完全一致率（严格 + 归一化两档）。

引擎（装了才跑，没装会明确列出来，不影响其它列）：
  * `现引擎`   —— ocr_readings.json（manga-ocr，stage2 生产读数）
  * `baberu`   —— models/baberu-ocr 的 ONNX（manga-ocr 蒸馏学生）
  * `hayai`    —— hayai-ocr（pip install hayai-ocr；manga-ocr 的 SigLIP2 分支）
  * `glm`      —— GLM-OCR(+漫画 LoRA) 的 OpenAI 兼容端点（--glm-endpoint/--glm-model）

用法（root venv）：
    venv/Scripts/python.exe ocr_engines_compare.py --book 甲书 --sample 150
    venv/Scripts/python.exe ocr_engines_compare.py --book 甲书 --score
    venv/Scripts/python.exe ocr_engines_compare.py --book 甲书 --sample 60 \
        --glm-endpoint http://127.0.0.1:11434/v1 --glm-model glm-ocr --glm-key x
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import difflib
import html
import io
import json
import random
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from manga_localizer.text_rules import is_ocr_commentary  # noqa: E402

PAD = 6                       # 与 stage2 一致：蓝框扩 6px 从填黑图裁剪
KANA = re.compile(r"[\u3040-\u30ffー]")
IDEOGRAPH = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]")
CHINESE_HINT = re.compile(r"的|了|是|这|那|没有|文字")
# Paddle 那两列来自 `paddle_rec_worker.py`（在 paddle_venv 里跑 rec-only）：
# `paddle原样` = 裁剪原样喂进去；`paddle转90` = 先转 90°（Paddle 的日文 rec 是
# 横排训练的，`japan_PP-OCRv3_mobile_rec` 且比它自己的中英模型落后两代）。
PADDLE_COLUMNS = ("paddle原样", "paddle转90")
ENGINES = ("现引擎", "baberu", *PADDLE_COLUMNS, "hayai", "glm")
PADDLE_VENV = ROOT / "paddle_venv" / "Scripts" / "python.exe"


def run_paddle_batch(crops: list[dict]) -> dict[str, dict]:
    """整批跑 Paddle 日文 rec（跨解释器）；不可用就返回 {}（列填 —）。"""
    import subprocess
    import tempfile

    worker = ROOT / "paddle_rec_worker.py"
    if not (PADDLE_VENV.is_file() and worker.is_file()):
        return {}
    with tempfile.TemporaryDirectory(prefix="paddle_rec_") as tmp:
        crops_path = Path(tmp) / "crops.json"
        out_path = Path(tmp) / "out.json"
        crops_path.write_text(json.dumps(crops, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [str(PADDLE_VENV), str(worker), "--crops", str(crops_path),
             "--out", str(out_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if not out_path.is_file():
            print(f"  [paddle] 跑失败（{result.returncode}）："
                  f"{(result.stderr or '')[-300:]}")
            return {}
        return json.loads(out_path.read_text(encoding="utf-8"))


# --- 抽样 -------------------------------------------------------------------

def suspect_reasons(text: str, page_counter: collections.Counter) -> list[str]:
    """这条读数"值得人工核一遍"的理由（与 var/probe_ocr_quality.py 同一套口径）。"""
    stripped = (text or "").strip()
    if not stripped:
        return ["空读数"]
    reasons = []
    if is_ocr_commentary(stripped):
        reasons.append("AI说明性输出")
    if not KANA.search(stripped) and not IDEOGRAPH.search(stripped):
        reasons.append("不含假名汉字")
    elif not KANA.search(stripped):
        reasons.append("只有汉字")
    if len([c for c in stripped if not c.isspace()]) == 1:
        reasons.append("单字")
    if LATIN.search(stripped) and not IDEOGRAPH.search(stripped):
        reasons.append("含拉丁数字")
    if CHINESE_HINT.search(stripped) and not KANA.search(stripped):
        reasons.append("疑似中文")
    if len([c for c in stripped if not c.isspace()]) >= 30:
        reasons.append("≥30字")
    if page_counter[stripped] > 1:
        reasons.append("同页重复")
    return reasons


def build_sample(book_dir: Path, sample: int, seed: int):
    readings = json.loads((book_dir / "ocr_readings.json").read_text(encoding="utf-8-sig"))
    per_page: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for key, text in readings.items():
        per_page[str(key).partition(":")[0]][(text or "").strip()] += 1
    flagged, plain = [], []
    for key, text in readings.items():
        page = str(key).partition(":")[0]
        reasons = suspect_reasons(text, per_page[page])
        (flagged if reasons else plain).append((key, text, reasons))
    random.Random(seed).shuffle(plain)
    picks = flagged + plain[:max(0, sample - len(flagged))]
    return picks, len(flagged), len(plain)


def crop_of(book_dir: Path, page: str, box_id: str):
    """按 stage2 的配方裁剪（返回 图 + 裁剪框 + 裁剪来源）。

    优先 `bluebox_masked.png`（stage2 真正喂给引擎的那张）；个别页缺它（例如只为
    补 `bg_flat_ratio` 重建过 dataset、没重跑 OCR 的页）→ 回退 `original.jpg`，
    并把来源记进表里，免得"引擎看到的是另一张图"被静默混进评分。
    """
    from PIL import Image

    blocks = json.loads(
        (book_dir / "dataset" / page / "selection_blocks.json")
        .read_text(encoding="utf-8-sig"))["selection_blocks"]
    block = next((b for b in blocks if b["id"] == box_id), None)
    if block is None:
        return None, None, ""
    masked = book_dir / "dataset" / page / "bluebox_masked.png"
    source = "masked" if masked.is_file() else "original"
    path = masked if masked.is_file() else book_dir / "dataset" / page / "original.jpg"
    if not path.is_file():
        return None, None, ""
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = block["image_bbox"]
    box = (max(0, x1 - PAD), max(0, y1 - PAD),
           min(image.width, x2 + PAD), min(image.height, y2 + PAD))
    return image, box, source


# --- 引擎适配（装了才跑）-----------------------------------------------------

def engine_baberu():
    try:
        sys.path.insert(0, str(ROOT / "models" / "baberu-ocr"))
        from onnx_infer import BaberuOnnxOCR
    except Exception as error:              # noqa: BLE001
        return None, f"未安装/不可用（{type(error).__name__}）"
    ocr = BaberuOnnxOCR(
        ROOT / "models" / "baberu-ocr" / "onnx",
        ROOT / "models" / "baberu-ocr" / "tokenizer",
        vision="vision_fp16.onnx",
    )
    return (lambda image: ocr(image)), ""


def engine_hayai():
    try:
        from hayai_ocr import HayaiOcr
    except Exception as error:              # noqa: BLE001
        return None, f"未安装（pip install hayai-ocr）：{type(error).__name__}"
    model = HayaiOcr()
    return (lambda image: model(image)), ""


def engine_glm(endpoint: str, model: str, key: str, prompt: str):
    if not endpoint:
        return None, "未配置 --glm-endpoint"

    def read(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = ("data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii"))
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data}},
                {"type": "text", "text": prompt},
            ]}],
        }
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key or 'x'}"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"]).strip()

    return read, ""


# --- 评分 -------------------------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if not ch.isspace())


def cer(reference: str, hypothesis: str) -> float:
    """字符错误率 = 编辑距离 / 参考长度（参考为空时：完全一致 0，否则 1）。"""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return 1.0 - difflib.SequenceMatcher(None, reference, hypothesis).ratio()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True, help="书名关键词即可")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sheet", default="")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--glm-endpoint", default="")
    parser.add_argument("--glm-model", default="glm-ocr")
    parser.add_argument("--glm-key", default="")
    parser.add_argument("--glm-prompt", default="Text Recognition:")
    args = parser.parse_args(argv)

    books = [p for p in (ROOT / "outputs" / "oneclick").iterdir()
             if p.is_dir() and args.book in p.name]
    if len(books) != 1:
        print(f"书名关键词 {args.book!r} 命中 {len(books)} 个产物目录")
        return 2
    book_dir = books[0]
    out_dir = ROOT / "var" / "ocr_eval" / book_dir.name
    sheet_path = Path(args.sheet) if args.sheet else out_dir / "sheet.csv"

    if args.score:
        return score(sheet_path)

    if args.sample <= 0:
        parser.error("要么给 --sample N，要么给 --score")

    picks, n_flagged, n_plain = build_sample(book_dir, args.sample, args.seed)
    print(f"{book_dir.name}")
    print(f"  抽样 {len(picks)} 框：可疑 {n_flagged} 个（全取）+ 普通 "
          f"{len(picks) - min(n_flagged, len(picks))} 个（随机，池 {n_plain}）")

    engines = {"现引擎": (lambda image: None), "baberu": engine_baberu(),
               "hayai": engine_hayai(),
               "glm": engine_glm(args.glm_endpoint, args.glm_model, args.glm_key,
                                 args.glm_prompt)}
    available, notes = {}, {}
    for name in ENGINES:
        if name == "现引擎":
            available[name] = True
            continue
        if name in PADDLE_COLUMNS:
            continue                    # 整批跑，见下面的 run_paddle_batch
        reader, note = engines[name]
        available[name] = reader is not None
        if reader is not None:
            engines[name] = reader
        else:
            notes[name] = note
            print(f"  [{name}] {note}")

    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (key, text, reasons) in enumerate(picks, 1):
        page, _, box_id = key.partition(":box:")
        page = page.partition(":")[0]
        masked, box, source = crop_of(book_dir, page, box_id)
        if masked is None:
            continue
        crop = masked.crop(box)
        name = f"{page}__{box_id}.png"
        crop.save(out_dir / "crops" / name)
        row = {"编号": f"#{index:03d}", "页": page, "框": box_id,
               "可疑理由": "/".join(reasons), "裁剪": name, "裁剪源": source,
               "现引擎": text}
        for engine in ENGINES[1:]:
            if engine in PADDLE_COLUMNS:
                row[engine] = "—"          # 整批跑，稍后统一填（run_paddle_batch）
                continue
            if not available[engine]:
                row[engine] = "—"
                continue
            try:
                row[engine] = engines[engine](crop) or ""
            except Exception as error:      # noqa: BLE001
                row[engine] = f"<{type(error).__name__}>"
        row["正确文本"] = ""
        rows.append(row)
        print(f"  {row['编号']} 页{page} {box_id} {'/'.join(reasons) or '普通'}"
              f" | 现引擎: {text[:20]}", flush=True)

    fieldnames = ["编号", "页", "框", "可疑理由", "裁剪", "裁剪源", *ENGINES,
                  "正确文本"]
    paddle_crops = [
        {"id": f"{row['页']}__{row['框']}",
         "path": str((out_dir / "crops" / row["裁剪"]).resolve())}
        for row in rows
    ]
    paddle_out = run_paddle_batch(paddle_crops) if paddle_crops else {}
    for row in rows:
        entry = paddle_out.get(f"{row['页']}__{row['框']}") or {}
        row["paddle原样"] = (entry.get("h") or {}).get("text", "—") or "—"
        row["paddle转90"] = (entry.get("v") or {}).get("text", "—") or "—"
    if paddle_out:
        print(f"  [paddle] 日文 rec 跑完 {len(paddle_out)} 框"
              f"（japan_PP-OCRv3_mobile_rec，原样 + 转 90° 两种朝向）")
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_html(out_dir / "sheet.html", rows, book_dir.name)
    print(f"\n表：{sheet_path}")
    print(f"HTML：{out_dir / 'sheet.html'}（图内嵌，逐条填「正确文本」）")
    print("填完后跑：--score")
    return 0


def score(sheet_path: Path) -> int:
    if not sheet_path.is_file():
        print(f"没有 {sheet_path}，先跑 --sample")
        return 2
    rows = list(csv.DictReader(sheet_path.open(encoding="utf-8-sig")))
    truth_rows = [r for r in rows if (r.get("正确文本") or "").strip() not in ("", "?")]
    print(f"{sheet_path.parent.name}：{len(rows)} 行，已填真值 {len(truth_rows)} 行")
    if not truth_rows:
        print("（还没填「正确文本」，无法算准确率）")
        return 0
    print(f"\n{'引擎':<10}{'严格一致':>10}{'归一化一致':>12}{'归一化 CER':>12}"
          f"{'平均相似度':>12}   可疑子集归一化一致")
    for engine in ENGINES:
        exact = norm_eq = 0
        cers, ratios = [], []
        sub_exact = sub_total = 0
        for row in truth_rows:
            truth = row["正确文本"].strip()
            got = (row.get(engine) or "").strip()
            if got == "—" or got.startswith("<"):
                continue
            exact += truth == got
            norm_eq += normalize(truth) == normalize(got)
            cers.append(cer(normalize(truth), normalize(got)))
            ratios.append(difflib.SequenceMatcher(
                None, normalize(truth), normalize(got)).ratio())
            if row.get("可疑理由"):
                sub_total += 1
                sub_exact += normalize(truth) == normalize(got)
        n = len(cers)
        if not n:
            print(f"{engine:<10}{'不可用':>10}")
            continue
        print(f"{engine:<10}{exact / n:>10.1%}{norm_eq / n:>12.1%}"
              f"{sum(cers) / n:>12.3f}{sum(ratios) / n:>12.3f}"
              f"   {sub_exact}/{sub_total}")
    return 0


def write_html(path: Path, rows, book: str) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(book)} 读字核对</title>",
        "<style>body{font-family:sans-serif;font-size:13px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;"
        "vertical-align:top}img{max-height:120px;image-rendering:pixelated}"
        ".r{color:#a00}</style>",
        f"<h3>{html.escape(book)} 读字核对（{len(rows)} 框）</h3>",
        "<p>把正确文本填进 sheet.csv 的「正确文本」列（看不清填 ?），"
        "然后跑 <code>--score</code>。</p><table>",
        "<tr><th>编号<th>页<th>框<th>可疑理由<th>裁剪<th>源"
        "<th>现引擎<th>baberu<th>paddle原样<th>paddle转90<th>hayai<th>glm"
        "<th>正确文本</tr>",
    ]
    for row in rows:
        cells = [f"{row['编号']}", f"{row['页']}", f"{row['框']}",
                 f"<span class='r'>{row['可疑理由']}</span>",
                 f"<img src='crops/{row['裁剪']}'>", f"{row['裁剪源']}"]
        cells += [html.escape(str(row.get(k) or "")) for k in ENGINES]
        cells.append("<i>（待填）</i>")
        parts.append("<tr>" + "".join(f"<td>{c}" for c in cells) + "</tr>")
    parts.append("</table>")
    path.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
