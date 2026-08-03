"""图片转PDF: 将文件夹中的图片按编号顺序合并为一个 PDF

用法:
    python 图片转pdf.py [图片文件夹] [输出.pdf]
    - 不带参数时交互询问（支持拖拽文件夹到窗口）
    - 输出默认: 文件夹所在目录下 <文件夹名>.pdf

示例:
    python 图片转pdf.py "C:\\Users\\YLP\\Desktop\\翻译\\2"
"""
import glob
import os
import re
import sys

# Windows 控制台切换 UTF-8，保证中文输出与输入正常（bat 保持 GBK 解析路径）
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def pause(msg="按回车退出…"):
    """等待回车（非交互环境自动跳过）"""
    try:
        input(msg)
    except EOFError:
        pass


def natural_key(name: str):
    """按编号自然排序: 1, 2, ..., 10, 11, ... (而不是 1, 10, 11, 2)"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.splitext(name)[0])]


def main():
    if len(sys.argv) > 1:
        src_dir = sys.argv[1].strip().strip('"')
    else:
        src_dir = input("图片文件夹路径（可拖拽文件夹到此处）: ").strip().strip('"')

    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print(f"[错误] 不是有效文件夹: {src_dir}")
        pause()
        sys.exit(1)

    files = [f for f in glob.glob(os.path.join(src_dir, "*"))
             if f.lower().endswith(IMAGE_EXT)]
    if not files:
        print(f"[错误] 文件夹中没有图片: {src_dir}")
        pause()
        sys.exit(1)
    files.sort(key=lambda f: natural_key(os.path.basename(f)))

    if len(sys.argv) > 2:
        out_pdf = os.path.abspath(sys.argv[2].strip().strip('"'))
    else:
        out_pdf = os.path.join(os.path.dirname(src_dir),
                               os.path.basename(src_dir.rstrip("\\/")) + ".pdf")

    print(f"共 {len(files)} 张图片，按编号顺序合并:")
    print(f"  {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    print(f"输出: {out_pdf}")

    import img2pdf
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(files))
    print(f"[完成] PDF 已生成: {out_pdf}")
    pause()


if __name__ == "__main__":
    main()
