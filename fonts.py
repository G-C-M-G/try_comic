"""字体管理: 扫描系统中文字体 + 项目漫画字体，提供友好名称与粗体解析

- 字体清单: 项目 manga-image-translator/fonts + C:\\Windows\\Fonts 常用字体
- resolve(name, bold): 按文件名查找；粗体优先粗体变体文件（msyh→msyhbd 等），无变体则 None（调用方描边模拟）
- JSON 中存字体文件名（如 msyh.ttc / anime_ace.ttf），跨机/跨进程可解析
"""
import os
import shutil

from paths import ROOT

PROJECT_FONTS = os.path.join(ROOT, "manga-image-translator", "fonts")
WINDOWS_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# 友好名称映射: 文件名(小写) -> 显示名
_FRIENDLY = {
    "msyh.ttc": "微软雅黑",
    "msyhbd.ttc": "微软雅黑 粗体",
    "msyhl.ttc": "微软雅黑 细体",
    "simhei.ttf": "黑体",
    "simsun.ttc": "宋体",
    "simkai.ttf": "楷体",
    "STFANGSO.TTF": "仿宋",
    "STKAITI.TTF": "华文楷体",
    "STLITI.TTF": "华文隶书",
    "STSONG.TTF": "华文宋体",
    "STXIHEI.TTF": "华文细黑",
    "Deng.ttf": "等线",
    "Dengb.ttf": "等线 粗体",
    "msgothic.ttc": "MS 明朝(日)",
    "meiryo.ttc": "Meiryo(日)",
    "meiryob.ttc": "Meiryo 粗体(日)",
    "anime_ace.ttf": "动漫体 Anime Ace",
    "anime_ace_3.ttf": "动漫体 Anime Ace 3",
    "comic shanns 2.ttf": "漫画体 Comic Shanns",
    "Arial-Unicode-Regular.ttf": "Arial Unicode",
    "arial.ttf": "Arial",
    "arialbd.ttf": "Arial 粗体",
    "NotoSansMonoCJK-VF.ttf.ttc": "Noto Sans Mono CJK",
}

# 粗体变体: 常规文件名(小写) -> 粗体文件名(小写)
_BOLD_VARIANTS = {
    "msyh.ttc": "msyhbd.ttc",
    "deng.ttf": "dengb.ttf",
    "arial.ttf": "arialbd.ttf",
    "meiryo.ttc": "meiryob.ttc",
    "simsun.ttc": "simsunb.ttf",
    "msgothic.ttc": "msgothic.ttc",   # 同族，加载后使用粗体样式位
}

_cached_fonts = None


def _scan_windows_fonts() -> list:
    """扫描 C:\\Windows\\Fonts 中常用字体文件"""
    wanted = {k.lower() for k in _FRIENDLY.keys()}
    wanted |= {k.lower() for k in _BOLD_VARIANTS.keys()}
    wanted |= {v.lower() for v in _BOLD_VARIANTS.values()}
    out = []
    try:
        for name in os.listdir(WINDOWS_FONTS):
            if name.lower() in wanted and name.lower().endswith((".ttf", ".ttc")):
                out.append(name)
    except Exception:
        pass
    return sorted(out)


def _scan_project_fonts() -> list:
    out = []
    try:
        for name in os.listdir(PROJECT_FONTS):
            if name.lower().endswith((".ttf", ".ttc")) and not name.startswith("."):
                out.append(name)
    except Exception:
        pass
    return sorted(out)


def list_fonts() -> list:
    """返回字体清单 [{"name", "display", "path", "bold_path"}]，按显示名排序"""
    global _cached_fonts
    if _cached_fonts is not None:
        return _cached_fonts
    seen = set()
    fonts = []
    for name in _scan_project_fonts() + _scan_windows_fonts():
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        path = resolve_path(name)
        if not path:
            continue
        display = _FRIENDLY.get(name, name)
        bold_v = _BOLD_VARIANTS.get(name.lower())
        bold_path = None
        if bold_v:
            bold_path = resolve_path(bold_v)
            if bold_v != name and not bold_path:
                bold_path = None
        fonts.append({"name": name, "display": display, "path": path, "bold_path": bold_path})
    fonts.sort(key=lambda f: (f["display"].startswith(("Arial", "MS", "Noto")), f["display"]))
    _cached_fonts = fonts
    return fonts


def resolve_path(name: str):
    """按文件名在 项目 fonts → 系统 Fonts 查找，返回绝对路径或 None"""
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    p1 = os.path.join(PROJECT_FONTS, name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(WINDOWS_FONTS, name)
    if os.path.exists(p2):
        return p2
    return None


def resolve(name: str, bold: bool = False):
    """解析渲染用字体路径；bold 且存在粗体变体时返回变体路径"""
    base = resolve_path(name)
    if bold and base:
        fname = os.path.basename(base)
        bv = _BOLD_VARIANTS.get(fname.lower())
        if bv:
            bp = resolve_path(bv)
            if bp:
                return bp
    return base


def default_font_name() -> str:
    return "msyh.ttc"


def import_font(path: str):
    """导入自定义字体到项目 fonts 目录，返回导入后的文件名（失败返回 None）"""
    if not path or not os.path.isfile(path):
        return None
    os.makedirs(PROJECT_FONTS, exist_ok=True)
    name = os.path.basename(path)
    dest = os.path.join(PROJECT_FONTS, name)
    if os.path.abspath(path) != os.path.abspath(dest):
        try:
            shutil.copy2(path, dest)
        except Exception:
            return None
    global _cached_fonts
    _cached_fonts = None
    return name
