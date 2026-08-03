"""术语表（译名一致性）

- 两级作用域: 全局(global) / 按输入目录(dir，输入文件夹名或压缩包名匹配时生效)
- 生成 MITS pre_dict 字典（正则转义 + 按长度降序），在翻译前替换原文，保证译名一致
- 工作台「记住此译法」→ add(global) 回写；重复条目更新译文而非重复添加
"""
import json
import os
import re

from paths import GLOSSARY_FILE

SCOPES = ("global", "dir")


class Glossary:
    def __init__(self, path: str = GLOSSARY_FILE):
        self.path = path
        self.entries = []  # [{"src","dst","scope","dir"}]
        self.load()

    # ---------- IO ----------

    def load(self):
        self.entries = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                for e in raw:
                    if isinstance(e, dict) and e.get("src") and e.get("dst"):
                        self.entries.append({
                            "src": str(e["src"]),
                            "dst": str(e["dst"]),
                            "scope": e.get("scope", "global"),
                            "dir": e.get("dir", ""),
                        })
        except Exception:
            pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[glossary] 保存失败: {e}")

    def file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.path)
        except Exception:
            return 0.0

    # ---------- 增删改 ----------

    def add(self, src: str, dst: str, scope: str = "global", dir_name: str = ""):
        """新增/更新条目（同 src+scope+dir 视为同一条目，更新译文）"""
        src = (src or "").strip()
        dst = (dst or "").strip()
        if not src or not dst:
            return False
        if scope not in SCOPES:
            scope = "global"
        for e in self.entries:
            if e["src"] == src and e["scope"] == scope and e.get("dir", "") == dir_name:
                e["dst"] = dst
                self.save()
                return True
        self.entries.append({"src": src, "dst": dst, "scope": scope, "dir": dir_name})
        self.save()
        return True

    def remove(self, index: int):
        if 0 <= index < len(self.entries):
            del self.entries[index]
            self.save()

    # ---------- 查询与生成 ----------

    def applicable(self, dir_name: str = "") -> list:
        """返回对指定输入目录生效的条目 (src, dst) 列表"""
        out = []
        for e in self.entries:
            if e["scope"] == "global":
                out.append((e["src"], e["dst"]))
            elif e["scope"] == "dir" and dir_name and e.get("dir", "") == dir_name:
                out.append((e["src"], e["dst"]))
        return out

    def build_dict_text(self, dir_name: str = "") -> str:
        """生成 MITS pre_dict 文件内容（正则转义，按原文长度降序，避免部分替换）"""
        pairs = self.applicable(dir_name)
        if not pairs:
            return ""
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        lines = []
        for src, dst in pairs:
            lines.append(f"{re.escape(src)} {dst}")
        return "\n".join(lines)


def make_glossary_dict_file(glossary: Glossary, dir_name: str = "") -> str:
    """写临时 pre_dict 文件，返回路径（无适用条目返回 None）"""
    text = glossary.build_dict_text(dir_name)
    if not text:
        return None
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name
