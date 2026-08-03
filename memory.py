"""翻译记忆库（越用越稳定）

- 自动记录: 每页翻译完成后，非空且 ≠ 原文的译文写入记忆（上限 2 万条，淘汰最旧）
- 命中: 精确匹配优先；未命中时用 difflib 相似度 ≥0.9 的最接近记录（相近句子沿用）
- 应用时机: post_translate 阶段覆盖翻译结果（先于拟声词词典修正），音译钩子也会先查记忆
"""
import difflib
import json
import logging
import os

from paths import MEMORY_FILE

logger = logging.getLogger("manga_translator.memory")

MAX_ENTRIES = 20000
FUZZY_THRESHOLD = 0.9


class TranslationMemory:
    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        self.data = {}  # 有序 dict: src -> dst
        self.load()

    # ---------- IO ----------

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.data = {str(k): str(v) for k, v in raw.items() if str(k) and str(v)}
        except Exception:
            self.data = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[memory] 保存失败: {e}")

    def file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.path)
        except Exception:
            return 0.0

    # ---------- 查询 ----------

    def get(self, text: str):
        """精确命中"""
        return self.data.get((text or "").strip())

    def get_fuzzy(self, text: str, threshold: float = FUZZY_THRESHOLD):
        """相近句子: 候选 = 长度接近且首字符相同，相似度 >= 阈值"""
        text = (text or "").strip()
        if len(text) < 3:
            return None
        best, best_ratio = None, 0.0
        lo, hi = max(1, len(text) - 3), len(text) + 3
        for k, v in self.data.items():
            if len(k) < lo or len(k) > hi:
                continue
            if k[:1] != text[:1]:
                continue
            ratio = difflib.SequenceMatcher(None, text, k).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, v
        return best if best_ratio >= threshold else None

    # ---------- 写入 ----------

    def put(self, text: str, dst: str):
        text = (text or "").strip()
        dst = (dst or "").strip()
        if not text or not dst or text == dst:
            return
        if text in self.data:
            if self.data[text] == dst:
                return
            del self.data[text]
        self.data[text] = dst
        while len(self.data) > MAX_ENTRIES:
            self.data.pop(next(iter(self.data)))
        self.save()

    def clear(self):
        self.data = {}
        self.save()

    # ---------- 钩子 ----------

    def apply_hook(self, ctx):
        """post_translate 修正: 记忆命中 → 用记忆译文覆盖（精确优先，其次模糊）"""
        for r in ctx.text_regions:
            text = (r.text or "").strip()
            if not text:
                continue
            dst = self.get(text) or self.get_fuzzy(text)
            if dst and dst != (r.translation or ""):
                r.translation = dst

    def record_hook(self, ctx):
        """自动记录本页最终译文（应在拟声词修正后调用）"""
        for r in ctx.text_regions:
            text = (r.text or "").strip()
            dst = (r.translation or "").strip()
            if text and dst and text != dst and len(text) >= 2:
                self.put(text, dst)


def make_memory_post_hook(memory: TranslationMemory, enable: bool = True,
                          apply: bool = True, record: bool = True):
    """生成 post_translate 组合钩子: 记忆修正(apply) → 记录(record)"""
    if not enable:
        return None

    def hook(ctx):
        if apply:
            try:
                memory.apply_hook(ctx)
            except Exception as e:
                logger.error(f"[memory] apply 失败: {e}")
        if record:
            try:
                memory.record_hook(ctx)
            except Exception as e:
                logger.error(f"[memory] record 失败: {e}")

    return hook
