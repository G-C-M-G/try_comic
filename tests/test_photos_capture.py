import unittest

import numpy as np

from photos_capture import OverlayBlock, _display_bbox_to_image, _normalize_text, detect_overlay_blocks


class PhotosCaptureTests(unittest.TestCase):
    def test_normalizes_photos_column_text(self):
        self.assertEqual(_normalize_text("槍\nに\n塗布"), "槍に塗布")
        self.assertEqual(_normalize_text("やられた !!"), "やられた！！")

    def test_maps_letterboxed_viewport_to_source_pixels(self):
        self.assertEqual(
            _display_bbox_to_image([52, 39, 78, 166], (663, 942), (985, 1400)),
            [77, 58, 116, 247],
        )

    def test_detects_white_outline_overlay_islands(self):
        base = np.full((120, 160, 3), 220, dtype=np.uint8)
        active = np.full_like(base, 128)
        active[10:55, 20:48] = base[10:55, 20:48]
        active[65:110, 100:145] = base[65:110, 100:145]

        blocks = detect_overlay_blocks(base, active)

        self.assertEqual(blocks, [OverlayBlock(20, 10, 28, 45), OverlayBlock(100, 65, 45, 45)])


if __name__ == "__main__":
    unittest.main()
