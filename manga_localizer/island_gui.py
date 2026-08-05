"""Island review GUI (M2-B): REVIEW dispositions with a lazy switch.

Loads the M2 artifacts (cross-validation verdicts + final texts + original
images), lists REVIEW islands, shows the original crop with the thick yellow
stroke, and lets the user approve / correct / reject / mark SFX.  The lazy
switch (default ON) auto-accepts every REVIEW island with the manga reading;
turning it off enables item-by-item review.  Dark mode is the default and can
be toggled from the sidebar.  Saves per-page ``{page}.dispositions.json``
for ``scripts/apply_dispositions.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from .island_contract import load_island_dataset
from .island_dry_run import OcrLine, render_island_dry_run
from .theme import apply_theme, load_ui_config, save_ui_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ui.json"
ISLAND_DARK_KEY = "island_dark"

DISPOSITIONS_APPROVE = "approve"
DISPOSITIONS_CORRECT = "correct"
DISPOSITIONS_REJECT = "reject"
DISPOSITIONS_SFX = "sfx"
_DISPOSITIONS = (DISPOSITIONS_APPROVE, DISPOSITIONS_CORRECT, DISPOSITIONS_REJECT, DISPOSITIONS_SFX)

_HEADERS = ["岛", "manga(权威)", "Paddle 列", "分歧原因", "处置", "最终文本"]


class ZoomPanView(QGraphicsView):
    """Image preview with wheel zoom and drag pan."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.Antialiasing
        )
        self._item: Optional[QGraphicsPixmapItem] = None

    def set_image(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())
        self.reset_view()

    def reset_view(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        event.accept()


class IslandReviewWindow(QMainWindow):
    def __init__(
        self,
        m2_dir: Path,
        dataset_root: Path,
        *,
        pages: Sequence[str] = ("26", "27", "28", "29", "30", "31", "32", "33", "34", "35"),
        out_dir: Optional[Path] = None,
        lazy_default: bool = True,
    ) -> None:
        super().__init__()
        self.m2_dir = Path(m2_dir).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.pages = list(pages)
        self.out_dir = Path(out_dir or self.m2_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._page_index = 0
        self._verdicts: list[dict] = []
        self._final_by_id: Mapping[str, str] = {}
        self._reviews: list[dict] = []
        self._row_index = 0
        self._lazy = bool(lazy_default)
        self.dark = bool(load_ui_config(CONFIG_PATH).get(ISLAND_DARK_KEY, True))

        self.setWindowTitle("高亮岛核对（M2-B）")
        self.resize(1280, 820)
        self.setMinimumSize(1180, 760)
        self._build_ui()
        self._apply_theme()
        self._load_page(0)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)

        self.page_combo = QComboBox()
        for page in self.pages:
            self.page_combo.addItem(f"第 {page} 页")
        self.page_combo.currentIndexChanged.connect(self._load_page)

        self.lazy_check = QCheckBox("懒人开关（默认承认 manga 结果）")
        self.lazy_check.setChecked(self._lazy)
        self.lazy_check.toggled.connect(self._on_lazy_toggled)

        self.dark_check = QCheckBox("深色模式")
        self.dark_check.setChecked(self.dark)
        self.dark_check.toggled.connect(self._toggle_theme)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)

        self.save_button = QPushButton("保存处置")
        self.save_button.clicked.connect(self._save_dispositions)

        sidebar_layout.addWidget(QLabel("页面"))
        sidebar_layout.addWidget(self.page_combo)
        sidebar_layout.addWidget(self.lazy_check)
        sidebar_layout.addWidget(self.dark_check)
        sidebar_layout.addWidget(self.summary_label)
        sidebar_layout.addStretch(1)
        sidebar_layout.addWidget(self.save_button)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        header = self.table.horizontalHeader()
        header.setMinimumHeight(32)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for column in (1, 2, 3, 5):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setMinimumWidth(430)
        self.table.setMinimumHeight(180)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.view = ZoomPanView()
        self.view.setMinimumSize(640, 300)

        self.hint_label = QLabel("原图（黄色描边=岛 bbox，蓝色框=Paddle 行；滚轮缩放，拖动平移）")

        self.correct_edit = QLineEdit()
        self.correct_edit.setPlaceholderText("改字：输入修正后的原文（配合'改字'按钮）")
        approve_button = QPushButton("批准(manga)")
        correct_button = QPushButton("改字")
        reject_button = QPushButton("拒绝")
        sfx_button = QPushButton("标SFX")
        approve_button.clicked.connect(lambda: self._dispose(DISPOSITIONS_APPROVE))
        correct_button.clicked.connect(lambda: self._dispose(DISPOSITIONS_CORRECT))
        reject_button.clicked.connect(lambda: self._dispose(DISPOSITIONS_REJECT))
        sfx_button.clicked.connect(lambda: self._dispose(DISPOSITIONS_SFX))
        edit_row = QWidget()
        edit_layout = QHBoxLayout(edit_row)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.addWidget(self.correct_edit, 1)
        edit_layout.addWidget(approve_button)
        edit_layout.addWidget(correct_button)
        edit_layout.addWidget(reject_button)
        edit_layout.addWidget(sfx_button)

        image_area = QWidget()
        image_layout = QVBoxLayout(image_area)
        image_layout.setContentsMargins(8, 8, 8, 8)
        image_layout.addWidget(self.hint_label)
        image_layout.addWidget(self.view, 1)
        image_layout.addWidget(edit_row)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(image_area)
        splitter.setSizes([320, 500])
        splitter.setChildrenCollapsible(False)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(sidebar)
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    # ------------------------------------------------------------ data
    def _load_page(self, index: int) -> None:
        if not (0 <= index < len(self.pages)):
            return
        self._page_index = index
        page = self.pages[index]
        cv_path = self.m2_dir / f"{page}.m2.cv.json"
        final_path = self.m2_dir / f"{page}.final.json"
        if not cv_path.is_file() or not final_path.is_file():
            QMessageBox.warning(self, "缺数据", f"缺少 {page} 页 M2 产物")
            return
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8"))
        self._verdicts = cv["verdicts"]
        self._final_by_id = {item["island_id"]: item["final_text"] for item in final["islands"]}
        self._reviews = [v for v in self._verdicts if v["status"] == "review"]
        self._row_index = 0
        self._refresh_table()
        self._update_image()

    def _item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setForeground(QColor("#e2e8f0" if self.dark else "#111827"))
        return item

    def _refresh_table(self) -> None:
        rows = self._reviews if not self._lazy else []
        self.table.setRowCount(len(rows))
        for row_index, verdict in enumerate(rows):
            island_id = verdict["island_id"]
            self.table.setItem(row_index, 0, self._item(island_id))
            self.table.setItem(row_index, 1, self._item(verdict.get("manga_text", "")))
            self.table.setItem(
                row_index, 2, self._item(" | ".join(verdict.get("paddle_columns", [])))
            )
            self.table.setItem(
                row_index, 3, self._item("、".join(verdict.get("review_reasons", [])))
            )
            self.table.setItem(row_index, 4, self._item("—"))
            self.table.setItem(
                row_index, 5, self._item(self._final_by_id.get(island_id, ""))
            )
        summary = (
            f"REVIEW {len(self._reviews)} 项"
            + ("（懒人开关：全部承认 manga）" if self._lazy else "（请逐项处置）")
        )
        self.summary_label.setText(summary)

    def _update_image(self) -> None:
        page = self.pages[self._page_index]
        dataset_dir = self.dataset_root / page
        image_path = dataset_dir / "original.jpg"
        if not image_path.is_file():
            self.view._scene.clear()
            self.view.set_image(QPixmap())
            return
        dataset = load_island_dataset(dataset_dir / "islands.json")
        paddle_path = self.m2_dir / f"{page}.paddle.lines.json"
        paddle_lines: Sequence[OcrLine] = ()
        if paddle_path.is_file():
            payload = json.loads(paddle_path.read_text(encoding="utf-8"))
            paddle_lines = [
                OcrLine(
                    polygon=tuple((float(p[0]), float(p[1])) for p in item["polygon"]),
                    text=str(item.get("text") or ""),
                )
                for item in payload.get("lines", [])
            ]
        with Image.open(image_path) as image:
            canvas = render_island_dry_run(
                image,
                dataset,
                ocr_lines=paddle_lines,
                stroke_color=(255, 215, 0),
                stroke_width=6,
            )
        self.view.set_image(_pixmap_from_image(canvas))

    def _on_selection_changed(self) -> None:
        self._row_index = max(0, self.table.currentRow())

    # ------------------------------------------------------ disposition
    def _dispose(self, decision: str) -> None:
        if self._lazy or not self._reviews:
            return
        row = self._row_index
        if not (0 <= row < len(self._reviews)):
            return
        verdict = self._reviews[row]
        corrected = self.correct_edit.text().strip()
        if decision == DISPOSITIONS_CORRECT and not corrected:
            QMessageBox.warning(self, "改字", "请先在输入框填写修正后的原文")
            return
        self.table.setItem(row, 4, self._item(decision))
        if decision == DISPOSITIONS_CORRECT:
            self.table.setItem(row, 5, self._item(corrected))
            verdict["_corrected"] = corrected
        else:
            self.table.setItem(
                row, 5, self._item(self._final_by_id.get(verdict["island_id"], ""))
            )
        verdict["_decision"] = decision

    def _on_lazy_toggled(self, checked: bool) -> None:
        self._lazy = checked
        self._refresh_table()
        self._update_image()

    def _dispositions_payload(self, page: str) -> dict:
        payload = {"page": page, "lazy": self._lazy, "items": []}
        for verdict in self._reviews:
            decision = getattr(verdict, "_decision", None)
            if self._lazy and decision is None:
                decision = DISPOSITIONS_APPROVE
            payload["items"].append(
                {
                    "island_id": verdict["island_id"],
                    "decision": decision,
                    "corrected_text": getattr(verdict, "_corrected", None),
                }
            )
        return payload

    def _save_dispositions(self) -> None:
        page = self.pages[self._page_index]
        payload = self._dispositions_payload(page)
        target = self.out_dir / f"{page}.dispositions.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "已保存", f"已保存 {len(payload['items'])} 项处置 -> {target}")

    # -------------------------------------------------------------- theme
    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.dark, CONFIG_PATH)
        config = load_ui_config(CONFIG_PATH)
        config[ISLAND_DARK_KEY] = self.dark
        save_ui_config(config, CONFIG_PATH)
        self._refresh_table()

    def _toggle_theme(self, dark: bool) -> None:
        self.dark = bool(dark)
        self._apply_theme()


def _pixmap_from_image(image) -> QPixmap:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.read())
    return pixmap
