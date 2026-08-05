from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from manga_localizer.errors import StateError
from manga_localizer.island_contract import dataset_from_data, load_island_dataset
from manga_localizer.island_dry_run import OcrLine
from manga_localizer.island_state import (
    build_page_state,
    load_page_state,
    reembed_from_state,
    validate_state_inputs,
    write_page_state,
)
from manga_localizer.island_typeset import TypesetConfig, typeset_island_page


def _line(x1, y1, x2, y2, text="x", island_id=None):
    line = OcrLine(polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)), text=text)
    if island_id:
        object.__setattr__(line, "island_id", island_id)
    return line


def _dataset(islands, size=(300, 400)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


def _island(bbox, area=9000.0, island_id="island_001"):
    return {
        "id": island_id,
        "image_bbox": list(bbox),
        "image_center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
        "image_area": area,
    }


def _erased_image():
    image = Image.new("RGB", (300, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(20, 100, 6):
        draw.line((x, 30, x, 200), fill=(0, 0, 0), width=2)
    return image


class StateRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "tmp_state"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.dataset = _dataset(
            [_island((10, 10, 110, 210), island_id="island_001"),
             _island((150, 10, 190, 210), island_id="island_002")]
        )
        self.erased = _erased_image()
        self.erased_path = self.tmp / "erased.png"
        self.erased.save(self.erased_path)
        self.original_path = self.tmp / "original.jpg"
        Image.new("RGB", (300, 400), (200, 200, 200)).save(self.original_path)
        self.translations = {"island_001": "这是一段测试文字", "island_002": "测试文字"}
        self.paddle = [
            _line(16, 14, 34, 206, island_id="island_001"),
            _line(42, 14, 60, 206, island_id="island_001"),
            _line(68, 14, 86, 206, island_id="island_001"),
            _line(152, 14, 170, 206, island_id="island_002"),
        ]

    def tearDown(self):
        for path in self.tmp.iterdir():
            path.unlink(missing_ok=True)
        self.tmp.rmdir()

    def _full_typeset(self):
        return typeset_island_page(
            self.erased, self.dataset, self.translations, self.paddle, config=TypesetConfig()
        )

    def test_state_roundtrip_reembed_identical(self):
        full = self._full_typeset()
        payload = {
            "page": "synthetic",
            "islands": [
                {"island_id": island_id, "translated_text": text}
                for island_id, text in self.translations.items()
            ],
        }
        state = build_page_state(
            page="synthetic",
            original_path=self.original_path,
            erased_path=self.erased_path,
            dataset=self.dataset,
            translations_payload=payload,
            audits=full.audits,
            paddle_lines=self.paddle,
        )
        state_path = write_page_state(state, self.tmp / "page.state.json")
        loaded = load_page_state(state_path)
        validate_state_inputs(loaded, self.dataset, self.erased_path)
        reembedded = reembed_from_state(loaded, self.dataset, self.erased, config=TypesetConfig())
        self.assertEqual(reembedded.image.size, full.image.size)
        diff = ImageChops.difference(reembedded.image, full.image).getbbox()
        self.assertIsNone(diff, "re-embed must reproduce the full pipeline image")

    def test_reembed_with_edit(self):
        full = self._full_typeset()
        payload = {
            "page": "synthetic",
            "islands": [
                {"island_id": island_id, "translated_text": text}
                for island_id, text in self.translations.items()
            ],
        }
        state = build_page_state(
            page="synthetic",
            original_path=self.original_path,
            erased_path=self.erased_path,
            dataset=self.dataset,
            translations_payload=payload,
            audits=full.audits,
            paddle_lines=self.paddle,
        )
        reembedded = reembed_from_state(
            state,
            self.dataset,
            self.erased,
            edits={"island_002": "修改后的文字内容"},
            config=TypesetConfig(),
        )
        audits = {item.island_id: item for item in reembedded.audits}
        self.assertEqual(audits["island_002"].text, "修改后的文字内容")
        self.assertTrue(audits["island_002"].status in ("typeset", "typeset_pending_review"))

    def test_reembed_rejects_stale_islands(self):
        full = self._full_typeset()
        payload = {
            "page": "synthetic",
            "islands": [
                {"island_id": island_id, "translated_text": text}
                for island_id, text in self.translations.items()
            ],
        }
        state = build_page_state(
            page="synthetic",
            original_path=self.original_path,
            erased_path=self.erased_path,
            dataset=self.dataset,
            translations_payload=payload,
            audits=full.audits,
            paddle_lines=self.paddle,
        )
        other = _dataset([_island((10, 10, 120, 210), island_id="island_001")])
        with self.assertRaises(StateError):
            validate_state_inputs(state, other, self.erased_path)

    def test_reembed_rejects_stale_erased(self):
        full = self._full_typeset()
        payload = {
            "page": "synthetic",
            "islands": [
                {"island_id": island_id, "translated_text": text}
                for island_id, text in self.translations.items()
            ],
        }
        state = build_page_state(
            page="synthetic",
            original_path=self.original_path,
            erased_path=self.erased_path,
            dataset=self.dataset,
            translations_payload=payload,
            audits=full.audits,
            paddle_lines=self.paddle,
        )
        changed = self.erased.copy()
        changed.putpixel((5, 5), (10, 10, 10))
        changed_path = self.tmp / "changed.png"
        changed.save(changed_path)
        with self.assertRaises(StateError):
            validate_state_inputs(state, self.dataset, changed_path)


if __name__ == "__main__":
    unittest.main()
