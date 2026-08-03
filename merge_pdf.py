"""将翻译后的图片文件夹按文件名自然顺序合并为单个 PDF

用法:
    python merge_pdf.py <图片文件夹> <输出.pdf>
"""
import glob
import os
import re
import sys


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.splitext(name)[0])]


def main():
    if len(sys.argv) < 3:
        print("用法: python merge_pdf.py <图片文件夹> <输出.pdf>")
        sys.exit(1)
    src_dir = sys.argv[1]
    out_pdf = sys.argv[2]

    files = [f for f in glob.glob(os.path.join(src_dir, "*"))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
    if not files:
        print("文件夹中没有图片")
        sys.exit(1)
    files.sort(key=lambda f: natural_key(os.path.basename(f)))

    print(f"共 {len(files)} 页，按名称顺序合并: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")

    import img2pdf
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(files))
    print(f"PDF 已生成: {out_pdf}")


if __name__ == "__main__":
    main()
