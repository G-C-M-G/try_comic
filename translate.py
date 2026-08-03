"""日文漫画自动翻译 + 嵌字（MITS + sakura 本地部署）

用法:
    python translate.py <输入> [输出] [选项]

输入: 图片文件 / 文件夹 / zip / cbz
输出: 默认与输入同目录，文件夹输入为 <名字>_translated，压缩包输入为 <名字>_translated.zip
选项:
    --config <json>       MITS 配置文件 (默认 config/sakura_chs_ctd_mocr.json)
    --no-sfx              关闭拟声词智能过滤
    --sfx-max <n>         拟声词数量阈值，超过则不翻译 (默认 10)
    --sfx-threshold <n>   气泡白色亮度阈值 (默认 190)
    --sfx-ratio <f>       气泡白色占比阈值 (默认 0.5)
    --overwrite           覆盖已翻译文件
    --save-text           保存 OCR+译文对照文件
    --format <fmt>        输出格式 png/jpg/webp
    --direction <dir>     竖排阅读方向: auto(右→左,默认)/ltr(左→右)/rtl
    --no-glossary         关闭术语表
    --no-memory           关闭翻译记忆库
    --no-cache            关闭译文缓存
"""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
MITS_DIR = os.path.join(ROOT, "manga-image-translator")
sys.path.insert(0, MITS_DIR)
sys.path.insert(0, ROOT)

os.environ["SAKURA_API_BASE"] = os.environ.get("SAKURA_API_BASE", "http://127.0.0.1:8080/v1")
os.environ["SAKURA_VERSION"] = os.environ.get("SAKURA_VERSION", "0.10")
os.environ["PATH"] = os.path.join(ROOT, "sakura", "llama") + ";" + os.environ.get("PATH", "")
# HF 模型缓存兜底（用户级环境变量未生效/未设置时）
_HF_CACHE_DEFAULT = r"D:\tools\hf-cache"
if (not os.environ.get("HF_HOME") or not os.path.isdir(os.environ.get("HF_HOME", ""))) \
        and os.path.isdir(_HF_CACHE_DEFAULT):
    os.environ["HF_HOME"] = _HF_CACHE_DEFAULT
# 无代理时用国内镜像下载兜底（hf-mirror.com）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 模型下载代理（如无代理变量则用默认本地代理）
if not os.environ.get("HTTP_PROXY") and not os.environ.get("http_proxy"):
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# 本机服务(如 sakura API)不得走代理，否则会挂起
if not os.environ.get("NO_PROXY") and not os.environ.get("no_proxy"):
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
# 离线优先: 缓存可用时禁止联网（无网时避免 HF 重试卡死）；关闭方式见 config/settings.json hf_offline
try:
    from settings import load_settings as _ls
    _hf_offline = bool(_ls().get("hf_offline", True))
except Exception:
    _hf_offline = True
if _hf_offline and os.environ.get("HF_HOME"):
    os.environ["HF_HUB_OFFLINE"] = "1"

from manga_translator.mode.local import MangaTranslatorLocal
from manga_translator.args import parser, reparse

from sfx_filter import make_sfx_hook, make_sfx_post_hook
from name_translation import make_transliteration_hook
from manual_state import save_page_state
from glossary import Glossary, make_glossary_dict_file
from memory import TranslationMemory
from cache import PageCache
from text_segments import attach_translation_segments

SUPPORTED_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def notify(title: str, message: str, popup: bool = True):
    """Windows 弹窗通知"""
    print(f"\n=== {title} ===\n{message}")
    if not popup:
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)  # 0x40 = MB_ICONINFORMATION
    except Exception:
        pass


def extract_archive(path: str, dest: str):
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)


def make_archive(src_dir: str, out_zip: str):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(src_dir):
            for f in sorted(files):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, src_dir)
                z.write(full, arc)


def collect_images(folder: str):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(SUPPORTED_IMAGES) and f.lower() != ".thumb":
                imgs.append(os.path.join(root, f))
    return imgs


def next_output_path(ext: str, is_dir: bool = False) -> str:
    """桌面\翻译 文件夹下按序号命名: 1.ext, 2.ext ..."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    folder = os.path.join(desktop, "翻译")
    os.makedirs(folder, exist_ok=True)
    nums = []
    for f in os.listdir(folder):
        base = os.path.splitext(f)[0]
        if base.isdigit():
            nums.append(int(base))
    n = max(nums) + 1 if nums else 1
    if is_dir:
        return os.path.join(folder, str(n))
    return os.path.join(folder, f"{n}{ext}")


def dir_name_for(input_path: str) -> str:
    """用于术语表按目录匹配的卷名: 文件夹输入=文件夹名, 压缩包=包名, 单图=所在文件夹名"""
    base = os.path.basename(input_path.rstrip("\\/"))
    if base.lower().endswith((".zip", ".cbz")):
        base = os.path.splitext(base)[0]
    if os.path.isfile(input_path) and not base.lower().endswith((".zip", ".cbz")):
        base = os.path.basename(os.path.dirname(input_path))
    return base


def build_hooks(input_path: str, params: dict, args, cache: PageCache, cache_signature: str,
                manual_dir: str, glossary: Glossary, memory: TranslationMemory,
                cache_enable: bool, memory_enable: bool):
    """构造并返回钩子集合"""
    sfx_report = os.path.join(ROOT, "sfx_report.txt")
    if os.path.exists(sfx_report):
        os.remove(sfx_report)

    hook = make_sfx_hook(sfx_max_count=args.sfx_max,
                         enable=not args.no_sfx,
                         white_threshold=args.sfx_threshold,
                         min_white_ratio=args.sfx_ratio,
                         report_path=sfx_report)
    sfx_post = make_sfx_post_hook(enable=not args.no_sfx, report_path=sfx_report)
    name_hook = make_transliteration_hook(enable=True, report_path=sfx_report, memory=memory if memory_enable else None)

    def post_hook(ctx):
        """记忆修正 → 拟声词词典修正 → 记忆记录（返回最终 regions）"""
        if memory_enable and memory is not None:
            memory.apply_hook(ctx)
        if sfx_post is not None:
            keep = sfx_post(ctx)
            if keep is not None:
                ctx.text_regions = keep
        if memory_enable and memory is not None:
            memory.record_hook(ctx)
        for region in ctx.text_regions:
            attach_translation_segments(region)
        return None

    # 术语表: 生成 pre_dict 字典文件（翻译前替换原文）
    dict_file = None
    if not args.no_glossary and glossary is not None:
        dict_file = make_glossary_dict_file(glossary, dir_name_for(input_path))
        if dict_file:
            params["pre_dict"] = dict_file

    # 状态导出 + 缓存写入（mid_pipeline_hook: 渲染后，region.font_size 为实际值）
    page_counter = {"n": 0}
    src_basename = os.path.splitext(os.path.basename(input_path))[0] if os.path.isfile(input_path) else ""
    src_ext = os.path.splitext(input_path)[1].lower() if os.path.isfile(input_path) else ""

    def _page_name():
        if src_basename:
            return src_basename
        page_counter["n"] += 1
        return f"page_{page_counter['n']:03d}"

    def mid_hook(ctx, config):
        name = _page_name()
        if manual_dir:
            img_clean = getattr(ctx, "img_inpainted_clean", None)
            save_page_state(manual_dir, name,
                            img_clean if img_clean is not None else ctx.img_inpainted,
                            ctx.text_regions, ctx.img_rgb, src_ext)
        if cache_enable and cache is not None:
            try:
                p = getattr(ctx, "input_path", None)
                if p and os.path.isfile(p):
                    cache.store(cache.key_for(p, cache_signature), name, src_ext,
                                ctx.img_rendered,
                                getattr(ctx, "img_inpainted_clean", None),
                                ctx.text_regions, ctx.img_rgb)
            except Exception as e:
                print(f"[cache] 写入异常: {e}")

    # 缓存跳过（translate() 最开头调用）
    def skip_hook(ctx):
        if not cache_enable or cache is None:
            return None
        p = getattr(ctx, "input_path", None)
        if not p or not os.path.isfile(p):
            return None
        try:
            key = cache.key_for(p, cache_signature)
            info = cache.lookup(key)
            if not info:
                return None
            # 命中: 同步调整状态文件到 manual_dir
            if manual_dir:
                os.makedirs(manual_dir, exist_ok=True)
                keydir = cache._key_dir(key)
                page = info.get("page", "")
                for suffix in (".json", "_inpainted.png", "_orig.png"):
                    f = os.path.join(keydir, page + suffix)
                    if os.path.exists(f):
                        shutil.copy2(f, os.path.join(manual_dir, page + suffix))
            return cache.load_result(key)
        except Exception as e:
            print(f"[cache] 命中检查异常: {e}")
            return None

    return hook, post_hook, name_hook, mid_hook, skip_hook, dict_file


async def run_translation(input_path, output_path, params: dict, hooks: dict, verbose: bool,
                          manual_dir=None) -> str:
    translator = MangaTranslatorLocal(params)
    translator.pre_translate_hook = hooks["pre"]
    translator.post_translate_hook = hooks["post"]
    translator.transliteration_hook = hooks["name"]
    translator.mid_pipeline_hook = hooks["mid"]
    translator.page_skip_hook = hooks["skip"]
    translator.verbose = verbose

    await translator.translate_path(input_path, output_path, params)
    return output_path


async def main():
    ap = argparse.ArgumentParser(description="日文漫画自动翻译+嵌字")
    ap.add_argument("input", help="图片文件/文件夹/zip/cbz")
    ap.add_argument("output", nargs="?", default=None, help="输出目录或文件")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "sakura_chs_ctd_mocr.json"))
    ap.add_argument("--no-sfx", action="store_true")
    ap.add_argument("--sfx-max", type=int, default=10)
    ap.add_argument("--sfx-threshold", type=int, default=190)
    ap.add_argument("--sfx-ratio", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--format", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-popup", action="store_true", help="不弹窗（仅命令行输出）")
    ap.add_argument("--font-min", type=int, default=22, help="嵌字最小字号(px)，防止译文过小 (默认22)")
    ap.add_argument("--manual-dir", default=None, help="保存手动调整状态到该目录（擦字图+文本框JSON）")
    ap.add_argument("--direction", default=None, choices=["auto", "ltr", "rtl"],
                    help="竖排阅读方向: auto(右→左,默认)/ltr(左→右)/rtl")
    ap.add_argument("--no-glossary", action="store_true", help="关闭术语表")
    ap.add_argument("--no-memory", action="store_true", help="关闭翻译记忆库")
    ap.add_argument("--no-cache", action="store_true", help="关闭译文缓存")
    # 诊断: 打印完整调用参数（排障用，异常时可在日志中看到真实调用）
    print(f"[translate] ARGV: {' '.join(sys.argv)}", file=sys.stderr)
    try:
        args = ap.parse_args()
    except SystemExit:
        # 容错: argparse 对 "input <选项> output" 交错顺序会漏认 output（nargs=? 的坑），
        # 把 output 重排到 input 之后（选项之前）再解析
        argv2 = list(sys.argv[1:])
        nonopt = [a for a in argv2 if not a.startswith("-")]
        if len(nonopt) >= 2 and nonopt[-1] != nonopt[0]:
            inp, out = nonopt[0], nonopt[-1]
            argv2 = [a for a in argv2 if a != out]
            argv2.insert(argv2.index(inp) + 1, out)
            args = ap.parse_args(argv2)
        else:
            raise

    from settings import load_settings
    settings = load_settings()
    direction = args.direction or settings.get("direction", "auto")
    glossary_enable = not args.no_glossary and bool(settings.get("enable_glossary", True))
    memory_enable = not args.no_memory and bool(settings.get("enable_memory", True))
    cache_enable = not args.no_cache and bool(settings.get("enable_cache", True))
    cache_limit_mb = int(settings.get("cache_limit_mb", 2048))

    inp = os.path.abspath(args.input)
    if not os.path.exists(inp):
        notify("错误", f"输入路径不存在:\n{inp}", not args.no_popup)
        sys.exit(1)

    # 检查 sakura 服务
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3)
    except Exception:
        notify("错误", "sakura 翻译服务未运行!\n请先运行 start_sakura.bat", not args.no_popup)
        sys.exit(1)

    is_archive = inp.lower().endswith((".zip", ".cbz"))
    is_file = os.path.isfile(inp)
    temp_dir = None

    if is_archive:
        temp_dir = tempfile.mkdtemp(prefix="mits_input_")
        try:
            extract_archive(inp, temp_dir)
        except Exception as e:
            notify("错误", f"解压失败: {e}", not args.no_popup)
            sys.exit(1)
        work_input = temp_dir
    else:
        work_input = inp

    # 构建 MITS params
    cli_args, unknown = parser.parse_known_args(
        ["local", "-i", work_input, "--use-gpu", "--config-file", args.config])
    params = vars(cli_args)
    params["overwrite"] = args.overwrite
    params["verbose"] = args.verbose
    params["save_text"] = args.save_text
    if args.format:
        params["format"] = args.format

    # 注入最小字号与阅读方向到配置（--font-min / --direction 覆盖）
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("render", {})["font_size_minimum"] = args.font_min
    cfg["render"]["vertical_rtl"] = direction != "ltr"
    temp_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(cfg, temp_cfg, ensure_ascii=False)
    temp_cfg.close()
    params["config_file"] = temp_cfg.name

    # 输出路径
    if args.output:
        out = os.path.abspath(args.output)
        if is_archive and not out.endswith((".zip", ".cbz")):
            out = out + ".zip"
    else:
        # 默认输出到 桌面\翻译 按序号命名
        if is_archive:
            out = next_output_path(".zip")
        elif is_file:
            out = next_output_path(os.path.splitext(inp)[1] or ".png")
        else:
            out = next_output_path("", is_dir=True)  # 文件夹输入: 用序号目录名

    if is_archive:
        tmp_out = tempfile.mkdtemp(prefix="mits_out_")
    else:
        tmp_out = out

    glossary = Glossary() if glossary_enable else None
    memory = TranslationMemory() if memory_enable else None
    cache = PageCache(limit_mb=cache_limit_mb) if cache_enable else None
    # 缓存签名: 完整管线配置 + 选项 + 术语表 mtime（内容变化自动失效）
    # 注意: 不含翻译记忆 mtime——记忆自动记录每页都变化，会使缓存永不命中；
    # 手动编辑记忆后想重新应用，可在「词典与缓存」页清空缓存
    sig_parts = [
        "pipeline2",
        "cfg" + hashlib.sha256(
            json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
        f"f{args.font_min}",
        f"sfx{0 if args.no_sfx else 1}",
        f"dir{direction}",
        f"gl{round(glossary.file_mtime() if glossary else 0, 3)}",
    ]
    cache_signature = "|".join(sig_parts)

    manual_dir = args.manual_dir
    if manual_dir:
        manual_dir = os.path.abspath(manual_dir)
        os.makedirs(manual_dir, exist_ok=True)

    pre_hook, post_hook, name_hook, mid_hook, skip_hook, dict_file = build_hooks(
        inp, params, args, cache, cache_signature, manual_dir, glossary, memory,
        cache_enable, memory_enable)

    try:
        hooks = {"pre": pre_hook, "post": post_hook, "name": name_hook,
                 "mid": mid_hook, "skip": skip_hook}
        await run_translation(work_input, tmp_out, params, hooks, args.verbose, manual_dir)

        if is_archive:
            make_archive(tmp_out, out)
            shutil.rmtree(tmp_out, ignore_errors=True)

        msg = f"已处理:\n{inp}\n\n输出:\n{out}"
        if cache_enable:
            st = cache.stats()
            msg += f"\n\n[译文缓存] 累计命中 {st['hits']} 次，当前占用 {st['size_mb']} MB"
        if not args.no_sfx and os.path.exists(os.path.join(ROOT, "sfx_report.txt")):
            with open(os.path.join(ROOT, "sfx_report.txt"), "r", encoding="utf-8") as f:
                sfx_info = f.read().strip()
            if sfx_info:
                msg += f"\n\n[拟声词处理]\n{sfx_info}"
    except Exception as e:
        traceback.print_exc()
        notify("处理失败", f"{e}", not args.no_popup)
        sys.exit(1)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if dict_file and os.path.exists(dict_file):
            try:
                os.unlink(dict_file)
            except Exception:
                pass
        if "config_file" in params and os.path.basename(params["config_file"]).endswith(".json"):
            cfg_path = params["config_file"]
            if cfg_path and cfg_path != args.config and os.path.exists(cfg_path):
                os.unlink(cfg_path)

    notify("翻译完成", msg, not args.no_popup)


if __name__ == "__main__":
    asyncio.run(main())
