from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from manga_localizer.island_gui import IslandReviewWindow

from scripts.apply_dispositions import apply_dispositions

UPSTREAM = Path(r"D:\workplace\opencode\try_comic")
REAL_M2 = Path(r"var/island_m1/m2")


def _synthetic_final(island_ids: list[str], texts: dict[str, str]) -> dict:
    return {
        "page": "synthetic",
        "islands": [
            {"island_id": island_id, "final_text": texts.get(island_id, ""), "source": "manga"}
            for island_id in island_ids
        ],
    }


class ApplyDispositionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "tmp_dispositions"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for path in self.tmp.glob("*"):
            path.unlink(missing_ok=True)
        self.tmp.rmdir()

    def _write(self, name: str, payload: dict) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_apply_correct_and_reject(self):
        final_path = self._write(
            "final.json",
            _synthetic_final(
                ["island_001", "island_002", "island_003"],
                {"island_001": "原文一", "island_002": "原文二", "island_003": "原文三"},
            ),
        )
        dispositions_path = self._write(
            "dispositions.json",
            {
                "page": "synthetic",
                "lazy": False,
                "items": [
                    {"island_id": "island_002", "decision": "correct", "corrected_text": "改后"},
                    {"island_id": "island_003", "decision": "reject", "corrected_text": None},
                ],
            },
        )
        updated = apply_dispositions(final_path, dispositions_path, self.tmp)
        by_id = {item["island_id"]: item for item in updated["islands"]}
        self.assertEqual(by_id["island_001"]["final_text"], "原文一")
        self.assertEqual(by_id["island_002"]["final_text"], "改后")
        self.assertEqual(by_id["island_002"]["source"], "corrected")
        self.assertNotIn("island_003", by_id)
        self.assertEqual(updated["dispositions"]["correct"], 1)
        self.assertEqual(updated["dispositions"]["reject"], 1)

    def test_lazy_approve_default(self):
        final_path = self._write(
            "final.json",
            _synthetic_final(["island_001"], {"island_001": "原文"}),
        )
        dispositions_path = self._write(
            "dispositions.json",
            {"page": "synthetic", "lazy": True, "items": []},
        )
        updated = apply_dispositions(final_path, dispositions_path, self.tmp)
        self.assertEqual(updated["islands"][0]["final_text"], "原文")
        self.assertEqual(updated["dispositions"]["approve"], 0)


@unittest.skipUnless(REAL_M2.is_dir(), "M2 artifacts not present")
class IslandGuiSmokeTest(unittest.TestCase):
    def test_window_builds_offscreen(self):
        application = QApplication.instance() or QApplication([])
        out_dir = Path(__file__).resolve().parent / "tmp_gui_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        window = IslandReviewWindow(
            REAL_M2,
            UPSTREAM / "outputs" / "exam" / "dataset",
            pages=["33"],
            out_dir=out_dir,
            lazy_default=True,
        )
        window._load_page(0)
        window._refresh_table()
        window._update_image()
        payload = window._dispositions_payload("33")
        self.assertEqual(window.page_combo.count(), 1)
        self.assertEqual(len(payload["items"]), 4)  # page 33 has 4 REVIEW items
        self.assertTrue(all(item["decision"] == "approve" for item in payload["items"]))


if __name__ == "__main__":
    unittest.main()
