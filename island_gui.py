"""Island review GUI launcher (M2-B)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from manga_localizer.island_gui import IslandReviewWindow
from manga_localizer.ocr_adapter import DEFAULT_UPSTREAM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="高亮岛核对页（M2-B）")
    parser.add_argument("--m2-dir", required=True, help="M2 artifacts dir (var/island_m1/m2)")
    parser.add_argument("--dataset-root", default=str(DEFAULT_UPSTREAM / "outputs" / "exam" / "dataset"))
    parser.add_argument("--pages", default="26-35")
    parser.add_argument("--out-dir", help="dispositions output dir (default: m2-dir)")
    parser.add_argument("--no-lazy", action="store_true", help="start with the lazy switch off")
    args = parser.parse_args(argv)

    pages = []
    for raw in args.pages.split(","):
        item = raw.strip()
        if "-" in item:
            left, right = item.split("-", 1)
            pages.extend(str(n) for n in range(int(left), int(right) + 1))
        else:
            pages.append(item)

    application = QApplication(sys.argv[:1])
    window = IslandReviewWindow(
        Path(args.m2_dir),
        Path(args.dataset_root),
        pages=pages,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        lazy_default=not args.no_lazy,
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
