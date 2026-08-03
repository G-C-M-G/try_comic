"""译文缓存: 整卷重翻时按页跳过已译页（OCR/翻译/擦字全部复用）

- key = sha256(输入图文件字节) + 选项签名（字号/拟声词/方向/术语表与记忆 mtime）
- 每页缓存目录: 译文缓存/<key>/ 内含 成品图 result.png + 调整状态(json/inpainted/orig)
- 上限(默认 2GB)超限按 mtime 淘汰最旧；命中次数统计供 GUI 展示
"""
import hashlib
import json
import logging
import os
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from paths import CACHE_DIR, CACHE_INDEX

logger = logging.getLogger("manga_translator.cache")

CACHE_VERSION = 2


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PageCache:
    def __init__(self, root_dir: str = CACHE_DIR, index_path: str = CACHE_INDEX,
                 limit_mb: int = 2048):
        self.root_dir = root_dir
        self.index_path = index_path
        self.limit_mb = int(limit_mb)
        self.index = {}  # key -> {"page","src_ext","mtime","size","hits"}
        self.load()

    # ---------- IO ----------

    def load(self):
        self.index = {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.index = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[cache] 索引保存失败: {e}")

    def _key_dir(self, key: str) -> str:
        # Windows 目录名不能含 '|'，用 key 的哈希作目录名
        return os.path.join(self.root_dir, hashlib.sha256(key.encode("utf-8")).hexdigest())

    def _key_size(self, key: str) -> int:
        d = self._key_dir(key)
        total = 0
        for root, _, files in os.walk(d):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
        return total

    # ---------- 查询 ----------

    def key_for(self, input_path: str, signature: str) -> str:
        return f"{file_sha256(input_path)}|{signature}|v{CACHE_VERSION}"

    def lookup(self, key: str):
        """命中返回 {"page","src_ext"}，并更新命中次数；未命中返回 None"""
        d = self._key_dir(key)
        result = os.path.join(d, "result.png")
        if key not in self.index or not os.path.exists(result):
            return None
        if not os.path.exists(os.path.join(d, self.index[key].get("page", "") + ".json")):
            return None
        self.index[key]["hits"] = int(self.index[key].get("hits", 0)) + 1
        self.index[key]["mtime"] = time.time()
        self.save()
        return dict(self.index[key])

    def load_result(self, key: str):
        """加载缓存成品图为 PIL RGB"""
        path = os.path.join(self._key_dir(key), "result.png")
        if not os.path.exists(path):
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    # ---------- 写入 ----------

    def store(self, key: str, page: str, src_ext: str,
              img_rendered_bgr: np.ndarray,
              img_inpainted_bgr: np.ndarray,
              regions, img_rgb: np.ndarray = None):
        """保存一页缓存: 成品图 + 调整状态文件"""
        try:
            from manual_state import save_page_state  # 惰性导入，避免 GUI 侧依赖 MITS
            d = self._key_dir(key)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(cv2.cvtColor(img_rendered_bgr, cv2.COLOR_BGR2RGB)).save(
                os.path.join(d, "result.png"))
            # 状态文件（擦字图用干净副本，避免残留旧译文）
            if img_inpainted_bgr is not None:
                save_page_state(d, page, img_inpainted_bgr, regions, img_rgb, src_ext)
            self.index[key] = {
                "page": page,
                "src_ext": (src_ext or "").lower(),
                "mtime": time.time(),
                "size": self._key_size(key),
                "hits": int(self.index.get(key, {}).get("hits", 0)),
            }
            self.save()
            self.evict()
        except Exception as e:
            logger.error(f"[cache] 写入失败: {e}")

    # ---------- 淘汰 / 统计 ----------

    def evict(self):
        """超出上限按 mtime 淘汰最旧条目"""
        limit = self.limit_mb * 1024 * 1024
        if limit <= 0:
            return
        total = sum(int(v.get("size", 0)) for v in self.index.values())
        if total <= limit:
            return
        for key in sorted(self.index, key=lambda k: self.index[k].get("mtime", 0)):
            shutil.rmtree(self._key_dir(key), ignore_errors=True)
            del self.index[key]
            total = sum(int(v.get("size", 0)) for v in self.index.values())
            self.save()
            if total <= limit:
                break

    def stats(self) -> dict:
        """返回 {"entries": n, "size_mb": float, "hits": n}"""
        entries = len(self.index)
        size = sum(int(v.get("size", 0)) for v in self.index.values())
        hits = sum(int(v.get("hits", 0)) for v in self.index.values())
        return {"entries": entries, "size_mb": round(size / 1048576, 1), "hits": hits}

    def clear(self):
        shutil.rmtree(self.root_dir, ignore_errors=True)
        self.index = {}
        self.save()
