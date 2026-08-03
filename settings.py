"""全局设置（config/settings.json）: 术语表/记忆/缓存开关与参数

GUI 与 translate.py 共用，任何修改即时生效（缓存签名含相关文件 mtime）。
"""
import json
import os

from paths import SETTINGS_FILE

DEFAULTS = {
    "enable_glossary": True,
    "enable_memory": True,
    "enable_cache": True,
    "cache_limit_mb": 2048,
    "direction": "auto",      # auto | ltr | rtl（auto=跟随原文，竖排右→左）
    "hf_offline": True,       # HF 模型离线优先（缓存可用时不联网；关掉可走 hf-mirror 下载）
}


def load_settings() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save_settings(cfg: dict):
    data = load_settings()
    data.update(cfg)
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
