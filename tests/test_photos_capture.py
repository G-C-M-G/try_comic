import unittest
from unittest.mock import Mock, patch

import numpy as np

from photos_capture import (
    OverlayBlock,
    _display_bbox_to_image,
    _glyph_clicks,
    _launch_photos,
    _normalize_text,
    detect_overlay_blocks,
)


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

    def test_finds_dark_glyph_centres_on_white_background(self):
        image = np.full((80, 80, 3), 245, dtype=np.uint8)
        image[20:40, 25:40] = 20
        image[45:65, 50:65] = 20

        self.assertEqual(_glyph_clicks(image), [(32, 30), (57, 55)])

    @patch("photos_capture.subprocess.Popen")
    @patch("photos_capture.os.startfile")
    def test_falls_back_to_photos_executable_without_file_association(
        self, startfile: Mock, popen: Mock
    ):
        error = OSError("no file association")
        error.winerror = 1155
        startfile.side_effect = error

        _launch_photos(r"C:\manga\8.jpg")

        popen.assert_called_once_with(
            ["explorer.exe", r"C:\manga\8.jpg"], close_fds=True
        )


if __name__ == "__main__":
    unittest.main()
