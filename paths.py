"""路径推断: 兼容 源码运行 / PyInstaller exe 两种模式

打包为 exe 后 sys.frozen 为真, 程序文件在临时解压目录,
所有外部资源(venv/sakura/translate.py/调整状态)必须基于 exe 所在目录推断。
exe 约定放在项目根目录(与 venv/ 同级), 桌面快捷方式仅指向 exe。
"""
import os
import sys

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))

PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
PYTHONW = os.path.join(ROOT, "venv", "Scripts", "pythonw.exe")
TRANSLATE = os.path.join(ROOT, "translate.py")
RERENDER = os.path.join(ROOT, "re_render.py")
SAKURA_SERVER = os.path.join(ROOT, "sakura", "sakura_server.py")
SAKURA_LOG = os.path.join(ROOT, "sakura", "gui_server.log")
CONFIG_DIR = os.path.join(ROOT, "config")
UI_CONFIG = os.path.join(CONFIG_DIR, "ui.json")
ICON_PATH = os.path.join(ROOT, "app.ico")

SAKURA_HEALTH = "http://127.0.0.1:8080/health"

# ---- 词典 / 记忆 / 缓存 ----
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.json")
GLOSSARY_FILE = os.path.join(CONFIG_DIR, "glossary.json")
MEMORY_FILE = os.path.join(CONFIG_DIR, "translation_memory.json")
CACHE_DIR = os.path.join(ROOT, "译文缓存")
CACHE_INDEX = os.path.join(CONFIG_DIR, "cache_index.json")
