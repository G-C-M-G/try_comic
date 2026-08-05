from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .errors import MangaLocalizerError, ReviewValidationError
from .ocr_contract import OCRContract, load_ocr_contract, resolve_image_path
from .ocr_adapter import (
    DEFAULT_UPSTREAM,
    OcrCorrectionRequest,
    build_apply_review_command,
    build_background_ocr_command,
    build_photos_capture_command,
    build_prepare_review_command,
    correction_status,
)
from .review_workspace import ReviewWorkspace
from .theme import add_shadow, apply_theme, is_dark_mode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RENDER_SCRIPT = PROJECT_ROOT / "render.py"
UI_CONFIG = PROJECT_ROOT / "config" / "ui.json"


@dataclass(frozen=True)
class PipelineLaunch:
    mode: str
    image: Path
    ocr: Path
    review: Path
    output: Path
    max_layout_overflow: int = 10
    manual_review: bool = True
    preserve_unit_center: bool = True
    translator: str = "sakura"
    translation_base_url: str = "http://127.0.0.1:8080/v1"
    mock_translations: Optional[Path] = None
    state: Optional[Path] = None


def build_render_command(
    request: PipelineLaunch,
    *,
    python_executable: Optional[str] = None,
    render_script: Path = RENDER_SCRIPT,
) -> Tuple[str, ...]:
    if request.mode not in {"dry-run", "full", "state"}:
        raise ValueError(f"unsupported GUI pipeline mode: {request.mode}")
    if request.max_layout_overflow not in {10, 20}:
        raise ValueError("max_layout_overflow must be 10 or 20")
    command = [
        python_executable or sys.executable,
        str(Path(render_script).resolve()),
        "--image",
        str(Path(request.image).resolve()),
        "--ocr",
        str(Path(request.ocr).resolve()),
        "--review",
        str(Path(request.review).resolve()),
        "--output",
        str(Path(request.output).resolve()),
        "--max-layout-overflow",
        str(request.max_layout_overflow),
    ]
    if not request.manual_review:
        command.append("--no-manual-review")
    if not request.preserve_unit_center:
        command.append("--no-preserve-unit-center")
    if request.mode == "dry-run":
        command.append("--dry-run")
    elif request.mode == "state":
        if request.state is None:
            raise ValueError("state mode requires a state JSON")
        command.extend(("--state", str(Path(request.state).resolve())))
    else:
        if request.translator not in {"sakura", "openai-compatible", "mock"}:
            raise ValueError(f"unsupported translator: {request.translator}")
        command.extend(
            (
                "--translator",
                request.translator,
                "--translation-base-url",
                request.translation_base_url,
            )
        )
        if request.translator == "mock":
            if request.mock_translations is None:
                raise ValueError("mock translator requires a translation mapping")
            command.extend(
                ("--mock-translations", str(Path(request.mock_translations).resolve()))
            )
    return tuple(command)


class ReviewCanvas(QGraphicsView):
    box_selected = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("reviewCanvas")
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap = QPixmap()
        self._fit_mode = True
        self._box_edit = False
        self._drag_origin = None
        self._drag_item = None

    def set_document(
        self,
        image_path: Optional[Path],
        items: Sequence[Mapping],
        selected_column_id: Optional[str],
    ) -> None:
        self._scene.clear()
        self._pixmap = QPixmap(str(image_path)) if image_path is not None else QPixmap()
        if self._pixmap.isNull():
            self._scene.setSceneRect(QRectF(0, 0, 640, 480))
            label = self._scene.addText("打开 OCR 或图片")
            label.setDefaultTextColor(QColor("#9ca3af"))
            label.setPos(250, 220)
            return
        pixmap_item = QGraphicsPixmapItem(self._pixmap)
        pixmap_item.setZValue(0)
        self._scene.addItem(pixmap_item)
        self._scene.setSceneRect(QRectF(self._pixmap.rect()))
        for item in items:
            self._add_review_overlay(item, item.get("column_id") == selected_column_id)
        if self._fit_mode:
            self.fit_image()

    def _add_review_overlay(self, item: Mapping, selected: bool) -> None:
        decision = str(item.get("decision", "pending"))
        colors = {
            "pending": QColor("#f59e0b"),
            "approved": QColor("#10b981"),
            "corrected": QColor("#14b8a6"),
            "rejected": QColor("#ef4444"),
        }
        color = QColor("#f59e0b") if selected else colors.get(decision, QColor("#f59e0b"))
        bbox = item.get("corrected_image_bbox") or item.get("image_bbox")
        if bbox:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            pen = QPen(QColor("#ffffff") if selected else color, 3 if selected else 2)
            if not selected:
                pen.setStyle(Qt.PenStyle.DashLine)
            rectangle = self._scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            rectangle.setZValue(10)
            label = self._scene.addSimpleText(str(item.get("column_id", "")))
            label.setBrush(QColor("#ffffff"))
            label.setPos(x1 + 2, max(0, y1 - 18))
            label.setZValue(12)
        if selected and not item.get("corrected_image_bbox"):
            polygon = item.get("corrected_polygon") or item.get("polygon")
            if polygon and len(polygon) >= 2:
                pen = QPen(QColor("#37d5ff"), 2)
                for first, second in zip(polygon, list(polygon[1:]) + [polygon[0]]):
                    line = self._scene.addLine(
                        float(first[0]),
                        float(first[1]),
                        float(second[0]),
                        float(second[1]),
                        pen,
                    )
                    line.setZValue(11)

    def fit_image(self) -> None:
        if self._pixmap.isNull():
            return
        self._fit_mode = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        if self._pixmap.isNull():
            return
        current = self.transform().m11()
        target = current * factor
        if 0.05 <= target <= 6.0:
            self._fit_mode = False
            self.scale(factor, factor)

    def set_box_edit(self, enabled: bool) -> None:
        self._box_edit = bool(enabled)
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_image()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if (
            self._box_edit
            and not self._pixmap.isNull()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag_origin = self._bounded_scene_point(event.position().toPoint())
            pen = QPen(QColor("#14b8a6"), 3)
            self._drag_item = self._scene.addRect(QRectF(self._drag_origin, self._drag_origin), pen)
            self._drag_item.setZValue(20)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None and self._drag_item is not None:
            end = self._bounded_scene_point(event.position().toPoint())
            self._drag_item.setRect(QRectF(self._drag_origin, end).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_origin is not None and self._drag_item is not None:
            rectangle = self._drag_item.rect().normalized()
            self._scene.removeItem(self._drag_item)
            self._drag_item = None
            self._drag_origin = None
            if rectangle.width() >= 2 and rectangle.height() >= 2:
                self.box_selected.emit(
                    (
                        round(rectangle.left()),
                        round(rectangle.top()),
                        round(rectangle.right()),
                        round(rectangle.bottom()),
                    )
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _bounded_scene_point(self, viewport_point):
        point = self.mapToScene(viewport_point)
        bounds = self._scene.sceneRect()
        point.setX(max(bounds.left(), min(bounds.right(), point.x())))
        point.setY(max(bounds.top(), min(bounds.bottom(), point.y())))
        return point


class MangaLocalizerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Manga Localizer")
        self.resize(1440, 860)
        self.setMinimumSize(1080, 680)
        self.contract: Optional[OCRContract] = None
        self.workspace: Optional[ReviewWorkspace] = None
        self.image_path: Optional[Path] = None
        self.selected_column_id: Optional[str] = None
        self.process: Optional[QProcess] = None
        self.process_output: Optional[Path] = None
        self.correction_process: Optional[QProcess] = None
        self.correction_action: Optional[str] = None
        self.correction_review_document: Optional[dict] = None
        self.correction_review_path: Optional[Path] = None
        self.dark = is_dark_mode(UI_CONFIG)
        self.cards = []
        self.nav_buttons = []
        self._build_window()
        self._apply_theme()
        self._update_review_summary()

    def _build_window(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_review_page())
        self.pages.addWidget(self._build_ocr_correction_page())
        self.pages.addWidget(self._build_run_page())
        layout.addWidget(self.pages, 1)
        self.statusBar().showMessage("就绪")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(8)
        brand = QLabel("Manga\nLocalizer")
        brand.setObjectName("title")
        layout.addWidget(brand)
        subtitle = QLabel("漫画本地化工作台")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(22)
        review_nav = self._nav_button(
            "人工核对",
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
            0,
        )
        run_nav = self._nav_button(
            "运行任务",
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            2,
        )
        ocr_nav = self._nav_button(
            "OCR订正",
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            1,
        )
        layout.addWidget(review_nav)
        layout.addWidget(ocr_nav)
        layout.addWidget(run_nav)
        layout.addStretch(1)
        self.sidebar_status = QLabel("尚未加载 OCR")
        self.sidebar_status.setObjectName("badgeIdle")
        self.sidebar_status.setWordWrap(True)
        layout.addWidget(self.sidebar_status)
        self.dark_toggle = QCheckBox("深色模式")
        self.dark_toggle.setChecked(self.dark)
        self.dark_toggle.toggled.connect(self._toggle_theme)
        layout.addWidget(self.dark_toggle)
        self._select_page(0)
        return sidebar

    def _nav_button(self, text: str, icon: QIcon, page: int) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navItem")
        button.setIcon(icon)
        button.setProperty("pageIndex", page)
        button.clicked.connect(lambda _checked=False, index=page: self._select_page(index))
        self.nav_buttons.append(button)
        return button

    def _select_page(self, index: int) -> None:
        if hasattr(self, "pages"):
            self.pages.setCurrentIndex(index)
        for button in self.nav_buttons:
            button.setProperty("active", int(button.property("pageIndex")) == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(20, 18, 20, 18)
        page_layout.setSpacing(14)
        header, actions = self._page_header(
            "人工核对",
            "逐列确认 OCR 文字与完整框，所有决定绑定源文件哈希",
        )
        self.review_badge = QLabel("待定 0")
        self.review_badge.setObjectName("badgeIdle")
        actions.addWidget(self.review_badge)
        actions.addSpacing(8)
        open_ocr = QPushButton("打开 OCR")
        open_ocr.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        open_ocr.clicked.connect(self.choose_ocr)
        actions.addWidget(open_ocr)
        open_image = QPushButton("打开图片")
        open_image.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        open_image.clicked.connect(self.choose_image)
        actions.addWidget(open_image)
        save = QPushButton("保存核对")
        save.setObjectName("primary")
        save.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        save.clicked.connect(self.save_review)
        actions.addWidget(save)
        page_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_canvas_card())
        splitter.addWidget(self._build_review_editor_card())
        splitter.setSizes([780, 480])
        page_layout.addWidget(splitter, 1)
        return page

    def _build_canvas_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        tools = QHBoxLayout()
        title = QLabel("原图与 OCR 几何")
        title.setObjectName("h2")
        tools.addWidget(title)
        tools.addStretch(1)
        zoom_out = QPushButton("−")
        zoom_out.setObjectName("ghost")
        zoom_out.setFixedWidth(38)
        zoom_out.setToolTip("缩小")
        zoom_out.clicked.connect(lambda: self.canvas.zoom(0.8))
        tools.addWidget(zoom_out)
        zoom_in = QPushButton("+")
        zoom_in.setObjectName("ghost")
        zoom_in.setFixedWidth(38)
        zoom_in.setToolTip("放大")
        zoom_in.clicked.connect(lambda: self.canvas.zoom(1.25))
        tools.addWidget(zoom_in)
        fit = QPushButton("适应")
        fit.setObjectName("ghost")
        fit.clicked.connect(self._fit_canvas)
        tools.addWidget(fit)
        self.box_edit_button = QPushButton("框选修正")
        self.box_edit_button.setCheckable(True)
        self.box_edit_button.toggled.connect(self._toggle_box_edit)
        tools.addWidget(self.box_edit_button)
        layout.addLayout(tools)
        self.canvas = ReviewCanvas()
        self.canvas.box_selected.connect(self._canvas_box_selected)
        layout.addWidget(self.canvas, 1)
        return card

    def _build_review_editor_card(self) -> QFrame:
        card = self._card()
        card.setMinimumWidth(420)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_row = QHBoxLayout()
        title = QLabel("风险队列")
        title.setObjectName("h2")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.queue_count = QLabel("0 项")
        self.queue_count.setObjectName("muted")
        title_row.addWidget(self.queue_count)
        layout.addLayout(title_row)

        self.review_table = QTableWidget(0, 3)
        self.review_table.setHorizontalHeaderLabels(("列", "决定", "当前文字"))
        self.review_table.verticalHeader().setVisible(False)
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.review_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.review_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.review_table.setShowGrid(False)
        self.review_table.setMinimumHeight(170)
        self.review_table.setMaximumHeight(220)
        header = self.review_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.review_table.itemSelectionChanged.connect(self._table_selection_changed)
        layout.addWidget(self.review_table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor = QWidget()
        editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 2, 4, 2)
        editor_layout.setSpacing(10)
        candidates = QGridLayout()
        candidates.setHorizontalSpacing(10)
        candidates.setVerticalSpacing(6)
        self.current_value = self._candidate_row(candidates, 0, "当前")
        self.manga_value = self._candidate_row(candidates, 1, "MangaOCR")
        self.paddle_value = self._candidate_row(candidates, 2, "PaddleOCR")
        self.reason_value = self._candidate_row(candidates, 3, "原因")
        candidates.setColumnStretch(1, 1)
        editor_layout.addLayout(candidates)

        self.decision_group = QButtonGroup(self)
        decision_row = QHBoxLayout()
        self.decision_buttons = {}
        for index, (label, decision) in enumerate(
            (("批准", "approved"), ("修正", "corrected"), ("拒绝", "rejected"), ("待定", "pending"))
        ):
            radio = QRadioButton(label)
            self.decision_group.addButton(radio, index)
            radio.setProperty("decision", decision)
            radio.toggled.connect(self._decision_changed)
            self.decision_buttons[decision] = radio
            decision_row.addWidget(radio)
        decision_row.addStretch(1)
        editor_layout.addLayout(decision_row)

        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.corrected_text = QLineEdit()
        form.addRow("修正文字", self.corrected_text)
        bbox_widget = QWidget()
        bbox_layout = QGridLayout(bbox_widget)
        bbox_layout.setContentsMargins(0, 0, 0, 0)
        bbox_layout.setHorizontalSpacing(8)
        bbox_layout.setVerticalSpacing(6)
        self.bbox_spins = []
        for index, label in enumerate(("X1", "Y1", "X2", "Y2")):
            spin = QSpinBox()
            spin.setPrefix(f"{label} ")
            spin.setRange(0, 99999)
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            bbox_layout.addWidget(spin, index // 2, index % 2)
            self.bbox_spins.append(spin)
        form.addRow("完整框", bbox_widget)
        self.note_edit = QLineEdit()
        form.addRow("备注", self.note_edit)
        editor_layout.addLayout(form)

        action_row = QHBoxLayout()
        apply_button = QPushButton("应用当前决定")
        apply_button.setObjectName("primary")
        apply_button.clicked.connect(self.apply_current_decision)
        action_row.addWidget(apply_button)
        action_row.addStretch(1)
        approve_all = QPushButton("批准全部待定")
        approve_all.clicked.connect(self.approve_all_pending)
        action_row.addWidget(approve_all)
        editor_layout.addLayout(action_row)
        editor_layout.addStretch(1)
        scroll.setWidget(editor)
        layout.addWidget(scroll, 1)
        return card

    def _build_ocr_correction_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        header, actions = self._page_header(
            "OCR订正",
            "Windows照片前台快采，上游后台融合，schema 4保持只读消费",
        )
        self.correction_badge = QLabel("就绪")
        self.correction_badge.setObjectName("badgeIdle")
        actions.addWidget(self.correction_badge)
        layout.addWidget(header)

        config_card = self._card()
        config_layout = QGridLayout(config_card)
        config_layout.setContentsMargins(14, 14, 14, 14)
        config_layout.setHorizontalSpacing(8)
        config_layout.setVerticalSpacing(8)
        title = QLabel("订正批次")
        title.setObjectName("h2")
        config_layout.addWidget(title, 0, 0, 1, 3)
        self.correction_upstream_edit = self._path_row(
            config_layout, 1, "上游", lambda: self._choose_correction_directory(self.correction_upstream_edit)
        )
        self.correction_images_edit = self._path_row(
            config_layout, 2, "原图目录", lambda: self._choose_correction_directory(self.correction_images_edit)
        )
        self.correction_capture_edit = self._path_row(
            config_layout, 3, "采集目录", lambda: self._choose_correction_directory(self.correction_capture_edit)
        )
        self.correction_ocr_edit = self._path_row(
            config_layout, 4, "OCR目录", lambda: self._choose_correction_directory(self.correction_ocr_edit)
        )
        config_layout.addWidget(QLabel("页号"), 5, 0)
        self.correction_pages_edit = QLineEdit("1,2,6,9,12,23,28,29,31,33,35")
        config_layout.addWidget(self.correction_pages_edit, 5, 1, 1, 2)
        self.correction_upstream_edit.setText(str(DEFAULT_UPSTREAM))
        work_root = PROJECT_ROOT / "output" / "ocr-correction"
        self.correction_capture_edit.setText(str(work_root / "photos-capture"))
        self.correction_ocr_edit.setText(str(work_root / "ocr"))
        controls = QHBoxLayout()
        refresh = QPushButton("刷新状态")
        refresh.clicked.connect(self._refresh_correction_status)
        controls.addWidget(refresh)
        controls.addStretch(1)
        self.capture_button = QPushButton("前台快速采集")
        self.capture_button.clicked.connect(self.run_photos_quick_capture)
        controls.addWidget(self.capture_button)
        self.analyze_ocr_button = QPushButton("后台分析")
        self.analyze_ocr_button.setObjectName("primary")
        self.analyze_ocr_button.clicked.connect(self.run_ocr_analysis)
        controls.addWidget(self.analyze_ocr_button)
        config_layout.addLayout(controls, 6, 0, 1, 3)
        config_layout.setColumnStretch(1, 1)
        layout.addWidget(config_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        status_card = self._card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(14, 14, 14, 14)
        status_title = QLabel("页面状态")
        status_title.setObjectName("h2")
        status_layout.addWidget(status_title)
        self.correction_status_table = QTableWidget(0, 4)
        self.correction_status_table.setHorizontalHeaderLabels(("页", "照片采集", "OCR", "核对"))
        self.correction_status_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.correction_status_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.correction_status_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        status_layout.addWidget(self.correction_status_table, 1)
        splitter.addWidget(status_card)

        review_card = self._card()
        review_layout = QVBoxLayout(review_card)
        review_layout.setContentsMargins(14, 14, 14, 14)
        review_header = QHBoxLayout()
        review_title = QLabel("Schema 2核对")
        review_title.setObjectName("h2")
        review_header.addWidget(review_title)
        review_header.addStretch(1)
        prepare = QPushButton("生成")
        prepare.clicked.connect(self.prepare_selected_correction_review)
        review_header.addWidget(prepare)
        load = QPushButton("加载")
        load.clicked.connect(self.load_correction_review)
        review_header.addWidget(load)
        save = QPushButton("保存")
        save.clicked.connect(self.save_correction_review)
        review_header.addWidget(save)
        apply_button = QPushButton("应用")
        apply_button.setObjectName("primary")
        apply_button.clicked.connect(self.apply_correction_review)
        review_header.addWidget(apply_button)
        review_layout.addLayout(review_header)
        self.correction_review_edit = QLineEdit()
        self.correction_review_edit.setPlaceholderText("选择页面后生成或加载 *.ocr.review.json")
        review_layout.addWidget(self.correction_review_edit)
        self.correction_items_table = QTableWidget(0, 9)
        self.correction_items_table.setHorizontalHeaderLabels(
            ("ID", "决定", "文字", "角色", "气泡", "顺序", "断句", "续句", "完整框")
        )
        self.correction_items_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.correction_items_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        review_layout.addWidget(self.correction_items_table, 1)
        add_row = QPushButton("新增文字框")
        add_row.clicked.connect(self.add_correction_row)
        review_layout.addWidget(add_row)
        splitter.addWidget(review_card)
        splitter.setSizes((360, 800))
        layout.addWidget(splitter, 1)

        self.correction_log = QPlainTextEdit()
        self.correction_log.setReadOnly(True)
        self.correction_log.setMaximumHeight(120)
        layout.addWidget(self.correction_log)
        return page

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        header, actions = self._page_header(
            "运行任务",
            "调用同一套无窗口 CLI 核心，运行日志与状态完整保留",
        )
        self.run_badge = QLabel("就绪")
        self.run_badge.setObjectName("badgeIdle")
        actions.addWidget(self.run_badge)
        layout.addWidget(header)

        path_card = self._card()
        path_layout = QGridLayout(path_card)
        path_layout.setContentsMargins(14, 14, 14, 14)
        path_layout.setHorizontalSpacing(8)
        path_layout.setVerticalSpacing(8)
        title = QLabel("输入与输出")
        title.setObjectName("h2")
        path_layout.addWidget(title, 0, 0, 1, 3)
        self.ocr_edit = self._path_row(path_layout, 1, "OCR", self.choose_ocr)
        self.image_edit = self._path_row(path_layout, 2, "原图", self.choose_image)
        self.review_edit = self._path_row(path_layout, 3, "Review", self.choose_review)
        self.output_edit = self._path_row(path_layout, 4, "输出", self.choose_output)
        path_layout.setColumnStretch(1, 1)
        layout.addWidget(path_card)

        middle = QHBoxLayout()
        middle.setSpacing(14)
        settings_card = self._card()
        settings_layout = QGridLayout(settings_card)
        settings_layout.setContentsMargins(14, 14, 14, 14)
        settings_layout.setHorizontalSpacing(12)
        settings_layout.setVerticalSpacing(10)
        settings_title = QLabel("处理策略")
        settings_title.setObjectName("h2")
        settings_layout.addWidget(settings_title, 0, 0, 1, 2)
        self.manual_review_check = QCheckBox("人工核对")
        self.manual_review_check.setChecked(True)
        self.manual_review_check.toggled.connect(self._update_review_summary)
        settings_layout.addWidget(self.manual_review_check, 1, 0)
        self.center_check = QCheckBox("整句中心锁定")
        self.center_check.setChecked(True)
        settings_layout.addWidget(self.center_check, 1, 1)
        settings_layout.addWidget(QLabel("排版边界"), 2, 0)
        self.overflow_combo = QComboBox()
        self.overflow_combo.addItems(("10", "20"))
        settings_layout.addWidget(self.overflow_combo, 2, 1)
        self.risk_label = QLabel("")
        self.risk_label.setObjectName("badgeWarn")
        self.risk_label.setWordWrap(True)
        settings_layout.addWidget(self.risk_label, 3, 0, 1, 2)
        settings_layout.setColumnStretch(1, 1)
        middle.addWidget(settings_card, 1)

        translator_card = self._card()
        translator_layout = QGridLayout(translator_card)
        translator_layout.setContentsMargins(14, 14, 14, 14)
        translator_layout.setHorizontalSpacing(12)
        translator_layout.setVerticalSpacing(10)
        translator_title = QLabel("翻译器")
        translator_title.setObjectName("h2")
        translator_layout.addWidget(translator_title, 0, 0, 1, 3)
        translator_layout.addWidget(QLabel("后端"), 1, 0)
        self.translator_combo = QComboBox()
        self.translator_combo.addItems(("sakura", "openai-compatible", "mock"))
        self.translator_combo.currentTextChanged.connect(self._translator_changed)
        translator_layout.addWidget(self.translator_combo, 1, 1, 1, 2)
        translator_layout.addWidget(QLabel("接口"), 2, 0)
        self.base_url_edit = QLineEdit("http://127.0.0.1:8080/v1")
        translator_layout.addWidget(self.base_url_edit, 2, 1, 1, 2)
        translator_layout.addWidget(QLabel("Mock JSON"), 3, 0)
        self.mock_edit = QLineEdit()
        translator_layout.addWidget(self.mock_edit, 3, 1)
        self.mock_button = QPushButton()
        self.mock_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.mock_button.setToolTip("选择 mock 整句译文")
        self.mock_button.setFixedWidth(42)
        self.mock_button.clicked.connect(self.choose_mock)
        translator_layout.addWidget(self.mock_button, 3, 2)
        translator_layout.setColumnStretch(1, 1)
        middle.addWidget(translator_card, 1)
        layout.addLayout(middle)

        log_card = self._card()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_layout.setSpacing(10)
        command_row = QHBoxLayout()
        log_title = QLabel("运行日志")
        log_title.setObjectName("h2")
        command_row.addWidget(log_title)
        command_row.addStretch(1)
        self.dry_run_button = QPushButton("Dry-run")
        self.dry_run_button.clicked.connect(self.run_dry_run)
        command_row.addWidget(self.dry_run_button)
        self.state_button = QPushButton("重嵌 State")
        self.state_button.clicked.connect(self.run_state)
        command_row.addWidget(self.state_button)
        self.full_button = QPushButton("运行完整流程")
        self.full_button.setObjectName("primary")
        self.full_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.full_button.clicked.connect(self.run_full)
        command_row.addWidget(self.full_button)
        log_layout.addLayout(command_row)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        log_layout.addWidget(self.log_edit, 1)
        layout.addWidget(log_card, 1)
        self._translator_changed(self.translator_combo.currentText())
        return page

    def _page_header(self, title: str, subtitle: str):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        labels = QVBoxLayout()
        labels.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("title")
        labels.addWidget(title_label)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("subtitle")
        labels.addWidget(subtitle_label)
        layout.addLayout(labels)
        layout.addStretch(1)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        layout.addLayout(actions)
        return header, actions

    def _card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        self.cards.append(card)
        return card

    def _candidate_row(self, layout: QGridLayout, row: int, label: str) -> QLabel:
        key = QLabel(label)
        key.setObjectName("muted")
        key.setAlignment(Qt.AlignmentFlag.AlignTop)
        value = QLabel("-")
        value.setWordWrap(True)
        value.setMinimumWidth(0)
        value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(key, row, 0)
        layout.addWidget(value, row, 1)
        return value

    def _path_row(self, layout: QGridLayout, row: int, label: str, callback) -> QLineEdit:
        layout.addWidget(QLabel(label), row, 0)
        edit = QLineEdit()
        layout.addWidget(edit, row, 1)
        button = QPushButton()
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        button.setToolTip(f"选择{label}")
        button.setFixedWidth(42)
        button.clicked.connect(callback)
        layout.addWidget(button, row, 2)
        return edit

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.dark, UI_CONFIG)
        for card in self.cards:
            add_shadow(card, self.dark)

    def _toggle_theme(self, dark: bool) -> None:
        self.dark = bool(dark)
        self._apply_theme()

    def _choose_correction_directory(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择目录",
            target.text() or str(PROJECT_ROOT),
        )
        if path:
            target.setText(str(Path(path).resolve()))
            self._refresh_correction_status()

    def _correction_pages(self) -> list[str]:
        values = []
        for raw in self.correction_pages_edit.text().split(","):
            item = raw.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                values.extend(str(number) for number in range(int(left), int(right) + 1))
            else:
                values.append(str(int(item)) if item.isdigit() else item)
        return list(dict.fromkeys(values))

    def _correction_request(self) -> OcrCorrectionRequest:
        return OcrCorrectionRequest(
            images=Path(self.correction_images_edit.text()),
            pages=self.correction_pages_edit.text().strip(),
            capture_dir=Path(self.correction_capture_edit.text()),
            ocr_dir=Path(self.correction_ocr_edit.text()),
            upstream=Path(self.correction_upstream_edit.text()),
        )

    def _refresh_correction_status(self) -> None:
        if not hasattr(self, "correction_status_table"):
            return
        try:
            pages = self._correction_pages()
        except ValueError:
            pages = []
        capture = Path(self.correction_capture_edit.text() or PROJECT_ROOT)
        ocr = Path(self.correction_ocr_edit.text() or PROJECT_ROOT)
        rows = correction_status(pages, capture, ocr)
        self.correction_status_table.setRowCount(0)
        for row, value in enumerate(rows):
            self.correction_status_table.insertRow(row)
            for column, key in enumerate(("page", "capture", "ocr", "review")):
                item = QTableWidgetItem(str(value[key]))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(value["page"]))
                self.correction_status_table.setItem(row, column, item)
        if rows and self.correction_status_table.currentRow() < 0:
            self.correction_status_table.selectRow(0)

    def _selected_correction_page(self) -> Optional[str]:
        row = self.correction_status_table.currentRow()
        if row < 0:
            return None
        item = self.correction_status_table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole) or item.text()) if item else None

    def run_photos_quick_capture(self) -> None:
        try:
            command = build_photos_capture_command(self._correction_request())
        except ValueError as error:
            QMessageBox.critical(self, "采集参数无效", str(error))
            return
        self._start_correction_process(command, "capture")

    def run_ocr_analysis(self) -> None:
        try:
            command = build_background_ocr_command(self._correction_request())
        except ValueError as error:
            QMessageBox.critical(self, "OCR参数无效", str(error))
            return
        self._start_correction_process(command, "analyze")

    def prepare_selected_correction_review(self) -> None:
        page = self._selected_correction_page()
        if not page:
            QMessageBox.warning(self, "缺少页面", "请先选择需要核对的页面。")
            return
        ocr = Path(self.correction_ocr_edit.text()) / f"{page}.ocr.json"
        if not ocr.is_file():
            QMessageBox.warning(self, "缺少 OCR", str(ocr))
            return
        review = Path(self.correction_ocr_edit.text()) / f"{page}.ocr.review.json"
        command = build_prepare_review_command(
            ocr,
            upstream=Path(self.correction_upstream_edit.text()),
            review_path=review,
        )
        self.correction_review_edit.setText(str(review))
        self._start_correction_process(command, "prepare-review")

    def _start_correction_process(self, command: Sequence[str], action: str) -> None:
        if self.correction_process is not None or self.process is not None:
            QMessageBox.warning(self, "任务运行中", "请等待当前任务完成。")
            return
        self.correction_action = action
        self.correction_process = QProcess(self)
        self.correction_process.setWorkingDirectory(self.correction_upstream_edit.text())
        self.correction_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.correction_process.readyReadStandardOutput.connect(self._read_correction_output)
        self.correction_process.finished.connect(self._correction_process_finished)
        self.correction_log.appendPlainText("\n$ " + self._display_command(command))
        self.correction_badge.setText("运行中")
        self.correction_badge.setObjectName("badgeWarn")
        for button in (self.capture_button, self.analyze_ocr_button):
            button.setEnabled(False)
        self.correction_process.start(command[0], list(command[1:]))

    def _read_correction_output(self) -> None:
        if self.correction_process is None:
            return
        raw = bytes(self.correction_process.readAllStandardOutput())
        if raw:
            self.correction_log.appendPlainText(raw.decode("utf-8", errors="replace").rstrip())

    def _correction_process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_correction_output()
        action = self.correction_action
        if exit_code == 0:
            self.correction_badge.setText("完成")
            self.correction_badge.setObjectName("badgeOk")
            if action == "prepare-review" and self.correction_review_edit.text():
                self._load_correction_review_path(Path(self.correction_review_edit.text()))
        else:
            self.correction_badge.setText("需处理" if exit_code == 2 else "失败")
            self.correction_badge.setObjectName("badgeWarn" if exit_code == 2 else "badgeErr")
        self.correction_badge.style().unpolish(self.correction_badge)
        self.correction_badge.style().polish(self.correction_badge)
        if self.correction_process is not None:
            self.correction_process.deleteLater()
        self.correction_process = None
        self.correction_action = None
        for button in (self.capture_button, self.analyze_ocr_button):
            button.setEnabled(True)
        self._refresh_correction_status()

    def load_correction_review(self) -> None:
        path_text = self.correction_review_edit.text().strip()
        if not path_text:
            path_text, _ = QFileDialog.getOpenFileName(
                self,
                "加载 Schema 2 OCR 核对",
                self.correction_ocr_edit.text() or str(PROJECT_ROOT),
                "Review JSON (*.review.json);;JSON (*.json)",
            )
        if path_text:
            self._load_correction_review_path(Path(path_text))

    def _load_correction_review_path(self, path: Path) -> None:
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if document.get("schema_version") != 2:
                raise ValueError("核对文件不是 schema 2")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            QMessageBox.critical(self, "无法加载核对文件", str(error))
            return
        self.correction_review_document = document
        self.correction_review_path = path.resolve()
        self.correction_review_edit.setText(str(self.correction_review_path))
        self._populate_correction_items()

    @staticmethod
    def _bbox_text(value) -> str:
        return ",".join(str(int(number)) for number in value) if value else ""

    @staticmethod
    def _parse_bbox_text(value: str) -> list[int]:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError("完整框必须为 x1,y1,x2,y2")
        result = [int(part) for part in parts]
        if not (result[0] < result[2] and result[1] < result[3]):
            raise ValueError("完整框坐标顺序无效")
        return result

    def _populate_correction_items(self) -> None:
        table = self.correction_items_table
        table.setRowCount(0)
        if self.correction_review_document is None:
            return
        rows = []
        rows.extend(("item", index, item) for index, item in enumerate(self.correction_review_document.get("items", [])))
        rows.extend(("recovery", index, item) for index, item in enumerate(self.correction_review_document.get("recoveries", [])))
        rows.extend(("addition", index, item) for index, item in enumerate(self.correction_review_document.get("additions", [])))
        for row, (kind, index, item) in enumerate(rows):
            table.insertRow(row)
            column_id = item.get("column_id") or item.get("candidate_id") or f"addition_{index + 1}"
            values = (
                column_id,
                item.get("decision", "pending"),
                item.get("corrected_text") or item.get("text") or item.get("current_text", ""),
                item.get("role", "uncertain"),
                item.get("container_id") or "",
                item.get("selection_order") if item.get("selection_order") is not None else "",
                "是" if item.get("break_after") else "否",
                "是" if item.get("join_after") else "否",
                self._bbox_text(
                    item.get("corrected_image_bbox")
                    or item.get("image_bbox")
                    or item.get("current_image_bbox")
                ),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, (kind, index))
                table.setItem(row, column, cell)

    def add_correction_row(self) -> None:
        if self.correction_review_document is None:
            QMessageBox.warning(self, "缺少核对文件", "请先生成或加载 schema 2 核对文件。")
            return
        additions = self.correction_review_document.setdefault("additions", [])
        additions.append({
            "column_id": f"column_added_{len(additions) + 1:03d}",
            "decision": "added",
            "text": "",
            "role": "dialogue",
            "container_id": None,
            "break_after": False,
            "join_after": False,
            "selection_order": 0,
            "image_bbox": None,
        })
        self._populate_correction_items()

    def save_correction_review(self) -> bool:
        if self.correction_review_document is None or self.correction_review_path is None:
            QMessageBox.warning(self, "缺少核对文件", "请先生成或加载 schema 2 核对文件。")
            return False
        try:
            for row in range(self.correction_items_table.rowCount()):
                identity = self.correction_items_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                kind, index = identity
                collection = {
                    "item": "items",
                    "recovery": "recoveries",
                    "addition": "additions",
                }[kind]
                item = self.correction_review_document[collection][index]
                decision = self.correction_items_table.item(row, 1).text().strip()
                text = self.correction_items_table.item(row, 2).text().strip()
                role = self.correction_items_table.item(row, 3).text().strip()
                container = self.correction_items_table.item(row, 4).text().strip() or None
                order_text = self.correction_items_table.item(row, 5).text().strip()
                break_after = self.correction_items_table.item(row, 6).text().strip() in {"是", "true", "1"}
                join_after = self.correction_items_table.item(row, 7).text().strip() in {"是", "true", "1"}
                if break_after and join_after:
                    raise ValueError(f"{self.correction_items_table.item(row, 0).text()} 不能同时断句和续句")
                bbox_text = self.correction_items_table.item(row, 8).text().strip()
                bbox = (
                    self._parse_bbox_text(bbox_text)
                    if bbox_text
                    else None
                )
                if bbox is None and (kind == "item" or decision in {"added", "approved", "corrected"}):
                    raise ValueError(f"{self.correction_items_table.item(row, 0).text()} 缺少完整框")
                item["decision"] = decision
                item["role"] = role
                item["container_id"] = container
                item["break_after"] = break_after
                item["join_after"] = join_after
                item["selection_order"] = int(order_text) if order_text else row
                if kind == "item":
                    changed = text != str(item.get("current_text", "")) or bbox != item.get("current_image_bbox")
                    item["corrected_text"] = text if text != str(item.get("current_text", "")) else None
                    item["corrected_image_bbox"] = bbox if bbox != item.get("current_image_bbox") else None
                    if changed and decision == "approved":
                        item["decision"] = "corrected"
                else:
                    item["text"] = text
                    if bbox is not None:
                        item["image_bbox"] = bbox
            temporary = self.correction_review_path.with_name(self.correction_review_path.name + ".tmp")
            temporary.write_text(
                json.dumps(self.correction_review_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.correction_review_path)
        except (OSError, TypeError, ValueError, AttributeError) as error:
            QMessageBox.critical(self, "无法保存核对文件", str(error))
            return False
        self.statusBar().showMessage(f"已保存 {self.correction_review_path.name}")
        return True

    def apply_correction_review(self) -> None:
        if not self.save_correction_review() or self.correction_review_path is None:
            return
        page = self._selected_correction_page()
        if not page:
            QMessageBox.warning(self, "缺少页面", "请选择核对文件对应页面。")
            return
        ocr = Path(self.correction_ocr_edit.text()) / f"{page}.ocr.json"
        output = Path(self.correction_ocr_edit.text()) / f"{page}.reviewed.ocr.json"
        command = build_apply_review_command(
            ocr,
            self.correction_review_path,
            output,
            upstream=Path(self.correction_upstream_edit.text()),
        )
        self._start_correction_process(command, "apply-review")
        self._refresh_canvas()

    def choose_ocr(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 schema 4 OCR JSON",
            str(PROJECT_ROOT / "samples" / "ocr"),
            "OCR JSON (*.ocr.json);;JSON (*.json)",
        )
        if path:
            self.load_ocr(Path(path))

    def load_ocr(self, path: Path) -> bool:
        if self.workspace is not None and self.workspace.dirty:
            answer = QMessageBox.question(
                self,
                "未保存核对",
                "放弃当前未保存的核对决定？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        try:
            contract = load_ocr_contract(path)
            review_path = path.with_suffix(".review.json")
            workspace = ReviewWorkspace.load_or_create(contract, review_path)
            image_path = resolve_image_path(contract)
        except MangaLocalizerError as error:
            QMessageBox.critical(self, "无法加载 OCR", str(error))
            return False
        self.contract = contract
        self.workspace = workspace
        self.image_path = image_path
        self.ocr_edit.setText(str(path.resolve()))
        self.review_edit.setText(str(review_path.resolve()))
        self.image_edit.setText(str(image_path.resolve()))
        stem = path.name.removesuffix(".ocr.json")
        self.output_edit.setText(str(PROJECT_ROOT / "output" / f"{stem}.translated.png"))
        mock_path = PROJECT_ROOT / "config" / "sample_translations" / f"{stem}.json"
        self.mock_edit.setText(str(mock_path) if mock_path.is_file() else "")
        self._configure_bbox_ranges()
        self._populate_review_table()
        self.sidebar_status.setText(path.name)
        self.statusBar().showMessage(f"已加载 {path.name}")
        return True

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择原图",
            str(self.image_path.parent if self.image_path else PROJECT_ROOT),
            "图片 (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.critical(self, "无法加载图片", "图片格式无效或文件已损坏。")
            return
        if self.contract is not None and (
            pixmap.width(),
            pixmap.height(),
        ) != self.contract.image_size:
            QMessageBox.critical(
                self,
                "图片尺寸不匹配",
                f"图片尺寸 {(pixmap.width(), pixmap.height())} 与 OCR {self.contract.image_size} 不一致。",
            )
            return
        self.image_path = Path(path).resolve()
        self.image_edit.setText(str(self.image_path))
        self._refresh_canvas()

    def choose_review(self) -> None:
        if self.contract is None:
            QMessageBox.warning(self, "缺少 OCR", "请先加载 OCR JSON。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择核对决定",
            str(Path(self.review_edit.text()).parent if self.review_edit.text() else PROJECT_ROOT),
            "Review JSON (*.review.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            self.workspace = ReviewWorkspace.load(self.contract, Path(path))
        except MangaLocalizerError as error:
            QMessageBox.critical(self, "无法加载核对决定", str(error))
            return
        self.review_edit.setText(str(Path(path).resolve()))
        self._populate_review_table()

    def choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择最终图片",
            self.output_edit.text() or str(PROJECT_ROOT / "output" / "translated.png"),
            "PNG (*.png)",
        )
        if path:
            self.output_edit.setText(str(Path(path).resolve()))

    def choose_mock(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 mock 整句译文",
            str(PROJECT_ROOT / "config" / "sample_translations"),
            "JSON (*.json)",
        )
        if path:
            self.mock_edit.setText(str(Path(path).resolve()))

    def _populate_review_table(self) -> None:
        self.review_table.blockSignals(True)
        self.review_table.setRowCount(0)
        if self.workspace is not None:
            for row, item in enumerate(self.workspace.items):
                self.review_table.insertRow(row)
                column = QTableWidgetItem(str(item["column_id"]))
                column.setData(Qt.ItemDataRole.UserRole, str(item["column_id"]))
                self.review_table.setItem(row, 0, column)
                self.review_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(self._decision_label(item.get("decision"))),
                )
                self.review_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(str(item.get("current_text", ""))),
                )
        self.review_table.blockSignals(False)
        if self.review_table.rowCount():
            self.review_table.selectRow(0)
            self._load_editor(self._row_column_id(0))
        else:
            self.selected_column_id = None
            self._clear_editor()
        self._update_review_summary()
        self._refresh_canvas()

    def _table_selection_changed(self) -> None:
        row = self.review_table.currentRow()
        if row >= 0:
            self._load_editor(self._row_column_id(row))

    def _row_column_id(self, row: int) -> str:
        item = self.review_table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _load_editor(self, column_id: str) -> None:
        if self.workspace is None:
            return
        item = self.workspace.item(column_id)
        self.selected_column_id = column_id
        decision = str(item.get("decision", "pending"))
        self.decision_buttons.get(decision, self.decision_buttons["pending"]).setChecked(True)
        self.current_value.setText(str(item.get("current_text") or "-"))
        readings = " / ".join(str(value) for value in item.get("mangaocr_readings", [])) or "-"
        paddle = str(item.get("paddleocr_text") or "-")
        self.manga_value.setText(readings + self._confidence(item.get("mangaocr_confidence")))
        self.paddle_value.setText(paddle + self._confidence(item.get("paddleocr_confidence")))
        self.reason_value.setText(", ".join(item.get("reason_codes", [])) or "-")
        self.corrected_text.setText(str(item.get("corrected_text") or item.get("current_text") or ""))
        bbox = item.get("corrected_image_bbox") or item.get("image_bbox") or (0, 0, 0, 0)
        for spin, value in zip(self.bbox_spins, bbox):
            spin.setValue(int(value))
        self.note_edit.setText(str(item.get("note", "")))
        self._decision_changed()
        self._refresh_canvas()

    def _clear_editor(self) -> None:
        for label in (self.current_value, self.manga_value, self.paddle_value, self.reason_value):
            label.setText("-")
        self.corrected_text.clear()
        self.note_edit.clear()
        for spin in self.bbox_spins:
            spin.setValue(0)

    def _checked_decision(self) -> str:
        button = self.decision_group.checkedButton()
        return str(button.property("decision")) if button is not None else "pending"

    def _decision_changed(self) -> None:
        corrected = self._checked_decision() == "corrected"
        self.corrected_text.setEnabled(corrected)
        for spin in self.bbox_spins:
            spin.setEnabled(corrected)

    def _configure_bbox_ranges(self) -> None:
        if self.contract is None:
            return
        width, height = self.contract.image_size
        for index, spin in enumerate(self.bbox_spins):
            spin.setRange(0, width if index % 2 == 0 else height)

    def apply_current_decision(self) -> None:
        if self.workspace is None or self.selected_column_id is None:
            return
        try:
            self.workspace.update(
                self.selected_column_id,
                self._checked_decision(),
                corrected_text=self.corrected_text.text(),
                corrected_image_bbox=tuple(spin.value() for spin in self.bbox_spins),
                note=self.note_edit.text(),
            )
        except MangaLocalizerError as error:
            QMessageBox.critical(self, "决定无效", str(error))
            return
        self._refresh_review_row(self.selected_column_id)
        self._update_review_summary()
        self._refresh_canvas()
        self.statusBar().showMessage(f"已应用 {self.selected_column_id}，尚未保存")

    def approve_all_pending(self) -> None:
        if self.workspace is None or self.workspace.counts.pending == 0:
            return
        answer = QMessageBox.question(
            self,
            "批准全部待定",
            "确认保留所有待定框的当前文字与几何？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for item in self.workspace.items:
            if item.get("decision") == "pending":
                self.workspace.update(str(item["column_id"]), "approved")
        self._populate_review_table()
        self.statusBar().showMessage("已批准全部待定项，尚未保存")

    def _refresh_review_row(self, column_id: str) -> None:
        if self.workspace is None:
            return
        updated = self.workspace.item(column_id)
        for row in range(self.review_table.rowCount()):
            if self._row_column_id(row) == column_id:
                self.review_table.item(row, 1).setText(
                    self._decision_label(updated.get("decision"))
                )
                break

    def save_review(self) -> bool:
        if self.workspace is None:
            QMessageBox.warning(self, "缺少 OCR", "请先加载 OCR JSON。")
            return False
        path_text = self.review_edit.text().strip()
        if not path_text:
            path_text, _ = QFileDialog.getSaveFileName(
                self,
                "保存核对决定",
                str(PROJECT_ROOT / "output" / "page.ocr.review.json"),
                "Review JSON (*.review.json);;JSON (*.json)",
            )
            if not path_text:
                return False
        try:
            application = self.workspace.save(Path(path_text))
        except MangaLocalizerError as error:
            QMessageBox.critical(self, "无法保存核对决定", str(error))
            return False
        self.review_edit.setText(str(self.workspace.path))
        self.statusBar().showMessage(
            f"核对已保存：待定 {application.pending_count}，拒绝 {application.rejected_count}"
        )
        return True

    def _update_review_summary(self, *_args) -> None:
        pending = self.workspace.counts.pending if self.workspace is not None else 0
        total = len(self.workspace.items) if self.workspace is not None else 0
        if hasattr(self, "queue_count"):
            self.queue_count.setText(f"{total} 项")
        if hasattr(self, "review_badge"):
            self.review_badge.setText(f"待定 {pending}")
            self.review_badge.setObjectName("badgeWarn" if pending else "badgeOk")
            self.review_badge.style().unpolish(self.review_badge)
            self.review_badge.style().polish(self.review_badge)
        if hasattr(self, "risk_label"):
            manual = self.manual_review_check.isChecked()
            if pending and manual:
                self.risk_label.setText(f"翻译阻断：仍有 {pending} 项 pending")
                self.risk_label.show()
            elif pending:
                self.risk_label.setText(f"人工核对已关闭：将 bypass {pending} 项风险")
                self.risk_label.show()
            else:
                self.risk_label.hide()

    def _refresh_canvas(self) -> None:
        if not hasattr(self, "canvas"):
            return
        items = self.workspace.items if self.workspace is not None else ()
        self.canvas.set_document(self.image_path, items, self.selected_column_id)

    def _fit_canvas(self) -> None:
        self.canvas.fit_image()

    def _toggle_box_edit(self, enabled: bool) -> None:
        if enabled and self.selected_column_id is None:
            self.box_edit_button.setChecked(False)
            return
        self.canvas.set_box_edit(enabled)
        self.box_edit_button.setText("完成框选" if enabled else "框选修正")

    def _canvas_box_selected(self, bbox: tuple) -> None:
        if self.selected_column_id is None:
            return
        self.decision_buttons["corrected"].setChecked(True)
        for spin, value in zip(self.bbox_spins, bbox):
            spin.setValue(int(value))
        self.statusBar().showMessage("新框已填入，点击“应用当前决定”确认")

    def _translator_changed(self, translator: str) -> None:
        mock = translator == "mock"
        self.mock_edit.setEnabled(mock)
        self.mock_button.setEnabled(mock)
        self.base_url_edit.setEnabled(not mock)

    def run_dry_run(self) -> None:
        if not self._ready_to_run(allow_pending=True):
            return
        ocr = Path(self.ocr_edit.text())
        output = Path(self.output_edit.text()).with_name(
            f"{ocr.name.removesuffix('.ocr.json')}.dry-run.png"
        )
        self._launch("dry-run", output)

    def run_full(self) -> None:
        if self._ready_to_run():
            self._launch("full", Path(self.output_edit.text()))

    def run_state(self) -> None:
        if not self._ready_to_run():
            return
        state, _ = QFileDialog.getOpenFileName(
            self,
            "选择可编辑 state",
            str(PROJECT_ROOT / "output"),
            "State JSON (*.state.json);;JSON (*.json)",
        )
        if not state:
            return
        output, _ = QFileDialog.getSaveFileName(
            self,
            "保存重嵌图片",
            str(Path(state).with_name(f"{Path(state).stem.removesuffix('.state')}.reembedded.png")),
            "PNG (*.png)",
        )
        if output:
            self._launch("state", Path(output), state=Path(state))

    def _ready_to_run(self, *, allow_pending: bool = False) -> bool:
        if self.process is not None:
            return False
        if self.workspace is None or self.contract is None:
            QMessageBox.warning(self, "缺少 OCR", "请先加载 OCR JSON。")
            return False
        if not self.image_edit.text().strip() or not self.output_edit.text().strip():
            QMessageBox.warning(self, "路径不完整", "请选择原图和输出路径。")
            return False
        if not self.save_review():
            return False
        if (
            not allow_pending
            and self.manual_review_check.isChecked()
            and self.workspace.counts.pending
        ):
            QMessageBox.warning(
                self,
                "人工核对阻断",
                f"仍有 {self.workspace.counts.pending} 项 pending，无法开始翻译。",
            )
            return False
        return True

    def _launch(self, mode: str, output: Path, *, state: Optional[Path] = None) -> None:
        try:
            request = PipelineLaunch(
                mode=mode,
                image=Path(self.image_edit.text()),
                ocr=Path(self.ocr_edit.text()),
                review=Path(self.review_edit.text()),
                output=output,
                max_layout_overflow=int(self.overflow_combo.currentText()),
                manual_review=self.manual_review_check.isChecked(),
                preserve_unit_center=self.center_check.isChecked(),
                translator=self.translator_combo.currentText(),
                translation_base_url=self.base_url_edit.text().strip(),
                mock_translations=(Path(self.mock_edit.text()) if self.mock_edit.text().strip() else None),
                state=state,
            )
            command = build_render_command(request)
        except (TypeError, ValueError) as error:
            QMessageBox.critical(self, "运行参数无效", str(error))
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        self.process_output = output
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self._set_run_buttons(False)
        self.log_edit.appendPlainText("\n$ " + self._display_command(command))
        self._set_run_status("运行中", "badgeWarn")
        self.process.start(command[0], list(command[1:]))

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput())
        if raw:
            self.log_edit.appendPlainText(raw.decode("utf-8", errors="replace").rstrip())

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        output = self.process_output
        self._read_process_output()
        if exit_code == 0 and output is not None and output.is_file():
            self.image_path = output
            self._refresh_canvas()
            self._set_run_status("完成", "badgeOk")
            self.statusBar().showMessage(f"完成：{output.name}")
        elif exit_code == 2 and output is not None and output.is_file():
            self.image_path = output
            self._refresh_canvas()
            self._set_run_status("需人工处理", "badgeWarn")
            self.statusBar().showMessage("流程需要人工处理，详情见运行日志")
        else:
            self._set_run_status("失败", "badgeErr")
            self.statusBar().showMessage(f"运行失败，退出码 {exit_code}")
        if self.process is not None:
            self.process.deleteLater()
        self.process = None
        self.process_output = None
        self._set_run_buttons(True)

    def _set_run_buttons(self, enabled: bool) -> None:
        for button in (self.dry_run_button, self.state_button, self.full_button):
            button.setEnabled(enabled)

    def _set_run_status(self, text: str, object_name: str) -> None:
        self.run_badge.setText(text)
        self.run_badge.setObjectName(object_name)
        self.run_badge.style().unpolish(self.run_badge)
        self.run_badge.style().polish(self.run_badge)

    @staticmethod
    def _display_command(command: Sequence[str]) -> str:
        return " ".join(f'"{part}"' if " " in part else part for part in command)

    @staticmethod
    def _decision_label(value) -> str:
        return {
            "approved": "批准",
            "corrected": "修正",
            "rejected": "拒绝",
            "pending": "待定",
        }.get(str(value), str(value))

    @staticmethod
    def _confidence(value) -> str:
        if value is None:
            return ""
        try:
            return f"  ({float(value):.3f})"
        except (TypeError, ValueError):
            return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manga Localizer PySide6 GUI")
    parser.add_argument("--ocr", type=Path, help="Open a schema 4 OCR JSON on launch.")
    return parser


def create_application(argv: Optional[Sequence[str]] = None) -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName("Manga Localizer")
    application.setOrganizationName("Manga Localizer")
    return application


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    application = create_application([sys.argv[0]])
    window = MangaLocalizerWindow()
    if args.ocr is not None:
        window.load_ocr(args.ocr)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
