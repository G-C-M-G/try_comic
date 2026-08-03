import unittest

import numpy as np

from photos_translate import _cluster_segments, _translation_query, _OVERRIDES
from manga_translator.utils import TextBlock


def segment(text, bbox):
    x1, y1, x2, y2 = bbox
    return {
        "text": text,
        "lines": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        "font_size": x2 - x1,
        "direction": "v",
    }


class PhotosTranslateTests(unittest.TestCase):
    def test_only_explicit_dialogue_override_is_kept(self):
        self.assertEqual(_OVERRIDES, {"やられた！！": "中招了！", "やられた!！": "中招了！"})

    def test_clusters_distant_bubbles_but_keeps_adjacent_columns(self):
        values = [
            segment("煙幕の", (587, 70, 618, 165)),
            segment("隙に", (548, 71, 579, 141)),
            segment("槍に塗布", (371, 70, 407, 198)),
            segment("していた", (334, 71, 367, 198)),
            segment("のか！？", (297, 71, 330, 166)),
        ]

        clusters = _cluster_segments(values)

        self.assertEqual(
            [[item["text"] for item in cluster] for cluster in clusters],
            [["煙幕の", "隙に"], ["槍に塗布", "していた", "のか！？"]],
        )

    def test_translation_query_marks_geometric_clusters(self):
        region = TextBlock(
            lines=np.array([[[1, 1], [2, 1], [2, 2], [1, 2]]]),
            texts=["dummy"],
            font_size=20,
            direction="v",
        )
        region.render_segments = [
            segment("それなのに", (793, 1077, 832, 1248)),
            segment("思考だけが", (755, 1072, 792, 1247)),
            segment("こんなにも", (101, 1083, 147, 1266)),
            segment("ハッキリして…", (65, 1086, 101, 1315)),
        ]

        query, _ = _translation_query(region)

        self.assertEqual(query, "<1>それなのに思考だけが<2>こんなにもハッキリして…")


if __name__ == "__main__":
    unittest.main()
