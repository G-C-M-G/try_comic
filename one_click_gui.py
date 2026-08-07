"""漫画翻译软件 — one-click GUI.

按 docs/GUI设计规范.md 实现（teal 主色 #14b8a6、卡片化、侧边栏布局）。
后台调用 one_click_translate.py（venv python 子进程），输出实时日志。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

def _find_root() -> Path:
    """Project root containing venv/one_click_translate.py.

    In PyInstaller onefile mode ``__file__`` points at the temp extraction
    dir, so resolve from the exe location instead and walk up until the
    venv is found (exe sits in ``dist/``, venv one level up).
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        candidates = [base, base.parent]
    else:
        base = Path(__file__).resolve().parent
        candidates = [base]
    for candidate in candidates:
        if (candidate / "venv" / "Scripts" / "python.exe").is_file():
            return candidate
    return candidates[0]


ROOT = _find_root()


def label(text: str) -> QLabel:
    """Secondary-text label (design token)."""
    return QLabel(text)
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "one_click_translate.py"

# ---- design tokens (docs/GUI设计规范.md) ----
LIGHT = {
    "bg": "#f9fafb", "card": "#ffffff", "border": "#e5e7eb",
    "border_hover": "#d1d5db", "hover": "#f3f4f6",
    "text": "#111827", "text_sec": "#4b5563", "text_mut": "#9ca3af",
    "primary": "#14b8a6", "primary_hover": "#0d9488",
    "primary_soft": "#f0fdfa", "primary_text": "#ffffff",
    "danger": "#ef4444", "warning": "#f59e0b",
    "disabled_primary": "rgba(20,184,166,0.5)",
    "selected_text": "#042f2e",
    "nav_selected_bg": "#f0fdfa",
    "nav_selected_text": "#0d9488",
}
DARK = {
    "bg": "#0f172a", "card": "#1e293b", "border": "#334155",
    "border_hover": "#475569", "hover": "#243244",
    "text": "#e2e8f0", "text_sec": "#cbd5e1", "text_mut": "#94a3b8",
    "primary": "#2dd4bf", "primary_hover": "#14b8a6",
    "primary_soft": "rgba(19,78,74,0.45)", "primary_text": "#042f2e",
    "danger": "#f87171", "warning": "#fbbf24",
    "disabled_primary": "rgba(45,212,191,0.4)",
    "selected_text": "#042f2e",
    "nav_selected_bg": "#2dd4bf",
    "nav_selected_text": "#042f2e",
}


def build_qss(t: dict) -> str:
    """QSS for the given token set (light/dark share the same layout)."""
    return f"""
* {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 14px; color: {t['text']}; }}
QMainWindow {{ background: {t['bg']}; }}
#sidebar {{ background: {t['card']}; border-right: 1px solid {t['border']}; }}
#brand {{ color: {t['primary']}; font-size: 18px; font-weight: 700; }}
#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 16px; }}
#cardTitle {{ color: {t['text']}; font-size: 14px; font-weight: 600; }}
QLabel {{ color: {t['text_sec']}; font-size: 13px; }}
QLineEdit, QSpinBox, QComboBox {{
  background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px;
  padding: 8px 12px; selection-background-color: {t['primary']};
  color: {t['text']};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {t['primary']}; }}
QPushButton {{ border-radius: 12px; padding: 8px 16px; border: 1px solid {t['border']}; background: {t['card']}; }}
QPushButton:hover {{ background: {t['hover']}; border-color: {t['border_hover']}; }}
QPushButton[primary="true"] {{
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t['primary']}, stop:1 {t['primary_hover']});
  color: {t['primary_text']}; font-weight: 600; border: none;
}}
QPushButton[primary="true"]:hover {{ background: {t['primary_hover']}; }}
QPushButton[primary="true"]:disabled {{ background: {t['disabled_primary']}; }}
QPushButton[danger="true"] {{ background: {t['danger']}; color: white; border: none; font-weight: 600; }}
QPushButton[ghost="true"] {{ background: transparent; border: none; color: {t['text_sec']}; }}
QPushButton[ghost="true"]:hover {{ background: {t['hover']}; }}
QTextEdit#log {{
  font-family: Consolas, "Courier New", monospace; font-size: 12px;
  background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px;
  color: {t['text']};
}}
QListWidget {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; }}
QListWidget::item {{ padding: 6px 10px; border-radius: 8px; }}
QListWidget::item:selected {{ background: {t['nav_selected_bg']}; color: {t['nav_selected_text']}; font-weight: 600; }}
QTableWidget {{
  background: {t['card']}; color: {t['text']};
  border: 1px solid {t['border']}; border-radius: 12px;
  gridline-color: {t['border']};
}}
QTableWidget::item {{ padding: 6px 8px; color: {t['text']}; }}
QTableWidget::item:selected {{ background: {t['primary_soft']}; color: {t['selected_text']}; }}
QHeaderView::section {{
  background: {t['hover']}; color: {t['text']};
  border: none; border-bottom: 1px solid {t['border']};
  padding: 6px 8px; font-weight: 600;
}}
QLineEdit, QSpinBox, QComboBox {{ color: {t['text']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{
  background: #ffffff; color: #111827;
  border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 8px 10px; font-size: 13px;
}}
"""


CAPTURE_HELP = (
    "<b>采集规矩</b><br>"
    "1. 每页采三个帧：原图帧 / 扫描文本帧 / 全选帧；<br>"
    "   蓝框检测比较<b>扫描文本帧 vs 全选帧</b>（非原图）。<br>"
    "2. 窗口 = 页面 × 显示缩放%（上限，放不下自动缩小），<br>"
    "   6 窗 3×2 并行、紧贴四边对齐；<br>"
    "   请勿手动移动/遮挡窗口（程序会实时锁位）。<br>"
    "3. 检测双路径：<b>严格蓝色蒙版公式优先</b><br>"
    "   （keep×原图+(0,38,67)，容差12），无结果时<br>"
    "   两帧任一 RGB 通道差&gt;8 兜底。<br>"
    "4. 整页大框（面积&gt;帧20%）自动丢弃，不是正文单位。<br>"
    "5. 硬规则：三次 Ctrl+A 无蓝色像素 → 该页进入<br>"
    "   <b>待审核</b>（审核页确认无框后才可关闭）。<br>"
    "6. 采集期间勿动鼠标键盘（窗口自动抢焦点）。"
)


class Runner(QThread):
    """Run one_click_translate.py as a subprocess, streaming output."""

    line = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, args: list[str]) -> None:
        super().__init__()
        self.args = args

    def run(self) -> None:
        cmd = [str(PYTHON), str(SCRIPT), *self.args]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            self.line.emit("!! 找不到 venv Python，请检查安装")
            self.finished_ok.emit(False)
            return
        for text in proc.stdout:
            self.line.emit(text.rstrip())
        code = proc.wait()
        self.finished_ok.emit(code == 0)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("漫画翻译软件")
        self.resize(1180, 760)
        icon = ROOT / "app.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.runner: Runner | None = None
        self.dark = True
        self._pending_colors: dict[str, str] = {}
        self._build()
        self._apply_theme()

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(DARK if self.dark else LIGHT))
        self.theme_btn.setText("🌙 暗色模式" if not self.dark else "☀️ 亮色模式")

    def _toggle_theme(self) -> None:
        self.dark = not self.dark
        self._apply_theme()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- sidebar ----
        side = QFrame(objectName="sidebar")
        side.setFixedWidth(220)
        side_l = QVBoxLayout(side)
        side_l.setContentsMargins(20, 24, 20, 24)
        brand = QLabel("漫画翻译软件", objectName="brand")
        side_l.addWidget(brand)
        sub = label("Photos 蓝框流水线")
        side_l.addWidget(sub)
        side_l.addSpacing(24)
        nav = QListWidget()
        for item in ("采集", "处理", "审核", "预览", "术语", "设置"):
            nav.addItem(QListWidgetItem(item))
        nav.setCurrentRow(0)
        side_l.addWidget(nav, 1)
        self.nav = nav
        self.theme_btn = QPushButton()
        self.theme_btn.setProperty("ghost", True)
        self.theme_btn.clicked.connect(self._toggle_theme)
        side_l.addWidget(self.theme_btn)
        root.addWidget(side)

        # ---- content ----
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)
        root.addWidget(content, 1)

        # shared controls (capture + process)
        self.images_edit = QLineEdit()
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        self.pages_edit = QLineEdit()
        self.pages_edit.setPlaceholderText("留空=全部，如 1-30 或 1,3,5")
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 12)
        self.parallel_spin.setValue(6)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(10, 100)
        self.scale_spin.setValue(42)

        self.tab_capture = self._build_capture_tab(browse)
        self.tab_process = self._build_process_tab()
        self.tab_review = self._build_review_tab()
        self.tab_preview = self._build_preview_tab()
        self.tab_glossary = self._build_glossary_tab()
        self.tab_settings = self._build_settings_tab()

        self.log = QTextEdit(readOnly=True, objectName="log")
        self.log.setMinimumHeight(140)
        progress_card, progress_lay = self._card("运行进度")
        self.stage_label = label("阶段：—")
        self.item_label = label("当前：—")
        self.eta_label = label("耗时 —　预计剩余 —")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        for w in (self.stage_label, self.item_label, self.eta_label):
            progress_lay.addWidget(w)
        progress_lay.addWidget(self.progress_bar)
        self._style_progress()
        cl.addWidget(progress_card)
        cl.addWidget(self.tab_capture)
        cl.addWidget(self.tab_process)
        cl.addWidget(self.tab_review)
        cl.addWidget(self.tab_preview)
        cl.addWidget(self.tab_glossary)
        cl.addWidget(self.tab_settings)
        cl.addWidget(self.log, 1)

        self._current = self.tab_capture
        self._show_tab(0)
        nav.currentRowChanged.connect(self._show_tab)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll_progress)
        self._timer.start()
        self._log("就绪。选择原图目录后开始采集。")

    def _card(self, title: str, help_text: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(objectName="card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lab = QLabel(title, objectName="cardTitle")
        if help_text:
            row = QHBoxLayout()
            row.addWidget(lab)
            row.addStretch(1)
            help_btn = QPushButton("ⓘ")
            help_btn.setProperty("ghost", True)
            help_btn.setFixedSize(30, 30)
            help_btn.setCursor(Qt.PointingHandCursor)
            help_btn.setToolTip(help_text)
            help_btn.setStyleSheet(
                "QPushButton { border: 2px solid #2dd4bf; border-radius: 15px; "
                "color: #2dd4bf; font-size: 17px; font-weight: 700; background: transparent; }"
                "QPushButton:hover { background: rgba(45,212,191,0.25); }"
            )
            row.addWidget(help_btn)
            lay.addLayout(row)
        else:
            lay.addWidget(lab)
        return card, lay

    def _build_capture_tab(self, browse: QPushButton) -> QFrame:
        card, lay = self._card("采集（Photos 自动框选 → 蓝框检测）", help_text=CAPTURE_HELP)
        row1 = QHBoxLayout()
        row1.addWidget(label("原图目录"))
        row1.addWidget(self.images_edit, 1)
        row1.addWidget(browse)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(label("页码范围"))
        row2.addWidget(self.pages_edit, 1)
        row2.addSpacing(16)
        row2.addWidget(label("并行窗口"))
        row2.addWidget(self.parallel_spin)
        row2.addSpacing(16)
        row2.addWidget(label("显示缩放%"))
        row2.addWidget(self.scale_spin)
        row2.addStretch(1)
        lay.addLayout(row2)
        btn = QPushButton("开始采集")
        btn.setProperty("primary", True)
        btn.clicked.connect(self._start_capture)
        lay.addWidget(btn, alignment=Qt.AlignRight)
        self.capture_btn = btn
        return card

    def _build_process_tab(self) -> QFrame:
        card, lay = self._card("处理（OCR → 翻译 → 擦字 → 嵌字）")
        note = label("后台跑 stage2：OCR 读蓝框、翻译（AI 优先、本地 Sakura 兜底）、排版出 typeset.png。"
                      "待审核页会跳过，需先在“审核”页确认。")
        note.setWordWrap(True)
        lay.addWidget(note)
        row = QHBoxLayout()
        self.resume_check = QCheckBox("断点续跑（跳过已有 typeset 的页）")
        self.pdf_check = QCheckBox("生成 PDF（默认不生成，只出 typeset 图）")
        row.addWidget(self.resume_check)
        row.addSpacing(24)
        row.addWidget(self.pdf_check)
        row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(label("填黑填充色"))
        self.fill_combo = QComboBox()
        self.fill_combo.addItem("黑色 0,0,0", "0,0,0")
        self.fill_combo.addItem("白色 255,255,255", "255,255,255")
        self.fill_combo.addItem("蓝色 30,144,255", "30,144,255")
        row2.addWidget(self.fill_combo)
        row2.addSpacing(24)
        row2.addWidget(label("处理并行任务数"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 3)
        self.workers_spin.setValue(1)
        row2.addWidget(self.workers_spin)
        row2.addStretch(1)
        lay.addLayout(row2)
        btn = QPushButton("开始处理")
        btn.setProperty("primary", True)
        btn.clicked.connect(self._start_process)
        lay.addWidget(btn, alignment=Qt.AlignRight)
        self.process_btn = btn
        return card

    def _style_progress(self) -> None:
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 12px; background: #334155; "
            "height: 14px; text-align: center; font-size: 11px; color: #e2e8f0; }"
            "QProgressBar::chunk { border-radius: 12px; background: #2dd4bf; }"
        )

    def _build_preview_tab(self) -> QFrame:
        card, lay = self._card("预览（原图 / 带蓝框 check 图）")
        tool = QHBoxLayout()
        self.prev_page_btn = QPushButton("上一页")
        self.next_page_btn = QPushButton("下一页")
        self.page_edit = QLineEdit()
        self.page_edit.setFixedWidth(70)
        self.page_total_label = label(" / —")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("原图", "original")
        self.mode_combo.addItem("带蓝框", "check")
        self.mode_combo.addItem("译文（typeset 结果）", "typeset")
        self.fit_btn = QPushButton("适应窗口")
        for w in (self.prev_page_btn, self.next_page_btn):
            tool.addWidget(w)
        tool.addWidget(label("页"))
        tool.addWidget(self.page_edit)
        tool.addWidget(self.page_total_label)
        tool.addSpacing(12)
        tool.addWidget(self.mode_combo)
        tool.addWidget(self.fit_btn)
        tool.addStretch(1)
        lay.addLayout(tool)

        self._view = QGraphicsView()
        self._view.setMinimumSize(900, 640)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._scene = QGraphicsScene(self)
        self._view.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self._view.wheelEvent = self._zoom_wheel
        self._current_page = 1
        self._pages_list: list[str] = []
        lay.addWidget(self._view, 1)

        self.prev_page_btn.clicked.connect(lambda: self._goto_page(self._current_page - 1))
        self.next_page_btn.clicked.connect(lambda: self._goto_page(self._current_page + 1))
        self.page_edit.returnPressed.connect(self._goto_edit_page)
        self.mode_combo.currentIndexChanged.connect(self._reload_page)
        self.fit_btn.clicked.connect(self._fit_view)
        return card

    def _zoom_wheel(self, event) -> None:
        from PySide6.QtGui import QWheelEvent

        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._view.scale(factor, factor)
        event.accept()

    def _fit_view(self) -> None:
        self._view.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def _preview_images(self) -> list[str]:
        images = self.images_edit.text().strip()
        if not images:
            return []
        out = ROOT / "outputs" / "oneclick" / Path(images).name
        if not out.is_dir():
            return []
        import re as _re

        pages = set()
        for page in out.glob("capture/*/capture.json"):
            try:
                data = json.loads(page.read_text(encoding="utf-8-sig"))
                src = data.get("source", {}).get("path")
                if src:
                    pages.add(page.parent.name)
            except (OSError, ValueError):
                continue
        return sorted(pages, key=lambda p: int(p) if p.isdigit() else p)

    def _preview_path(self, page: str) -> str | None:
        images = self.images_edit.text().strip()
        if not images:
            return None
        out = ROOT / "outputs" / "oneclick" / Path(images).name
        if self.mode_combo.currentData() == "typeset":
            typeset = out / "typeset" / f"{page}.typeset.png"
            if typeset.is_file():
                return str(typeset)
            return None
        if self.mode_combo.currentData() == "check":
            check = out / "check" / f"{page}.jpg"
            if check.is_file():
                return str(check)
        manifest = out / "capture" / page / "capture.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
                src = data.get("source", {}).get("path")
                if src and Path(src).is_file():
                    return src
            except (OSError, ValueError):
                pass
        return None

    def _reload_page(self) -> None:
        if not self._pages_list:
            return
        page = self._pages_list[min(self._current_page - 1, len(self._pages_list) - 1)]
        path = self._preview_path(page)
        if path is None:
            self._scene.clear()
            self._pix_item = QGraphicsPixmapItem()
            self._scene.addItem(self._pix_item)
            self.page_total_label.setText(f" / {len(self._pages_list)}")
            return
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        self.page_edit.setText(page)
        self.page_total_label.setText(f" / {len(self._pages_list)}")
        self._fit_view()

    def _goto_page(self, page: int) -> None:
        if not self._pages_list:
            return
        self._current_page = max(1, min(page, len(self._pages_list)))
        self._reload_page()

    def _goto_edit_page(self) -> None:
        try:
            page = int(self.page_edit.text().strip())
        except ValueError:
            return
        if page in (int(p) if p.isdigit() else -1 for p in self._pages_list):
            self._current_page = self._pages_list.index(str(page)) + 1
            self._reload_page()
        else:
            self._goto_page(page)

    def _build_settings_tab(self) -> QFrame:
        card, lay = self._card("设置（AI / 采集 / 嵌字 / 蓝框检测）")
        note = label(
            "翻译：配置 DeepSeek API Key 后优先走 AI 整页翻译，失败自动回退本地 Sakura；"
            "AI OCR：配置 OpenAI 兼容视觉模型后蓝框文字改用视觉模型，留空则用本地 manga-ocr。"
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        self.ds_key_edit = QLineEdit()
        self.ds_key_edit.setEchoMode(QLineEdit.Password)
        self.ai_ocr_url_edit = QLineEdit()
        self.ai_ocr_url_edit.setPlaceholderText("https://api.siliconflow.cn/v1")
        self.ai_ocr_model_edit = QLineEdit()
        self.ai_ocr_model_edit.setPlaceholderText("Qwen/Qwen2.5-VL-72B-Instruct")
        self.ai_ocr_key_edit = QLineEdit()
        self.ai_ocr_key_edit.setEchoMode(QLineEdit.Password)

        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("2400x1400")
        self.min_font_spin = QSpinBox()
        self.min_font_spin.setRange(12, 80)
        self.min_font_spin.setValue(28)
        self.font_combo = QComboBox()
        for name, file in (
            ("微软雅黑 msyh.ttc", "msyh.ttc"),
            ("黑体 simhei.ttf", "simhei.ttf"),
            ("楷体 simkai.ttf", "simkai.ttf"),
            ("仿宋 simfang.ttf", "simfang.ttf"),
            ("宋体 simsun.ttc", "simsun.ttc"),
        ):
            self.font_combo.addItem(name, file)
        self.stroke_spin = QSpinBox()
        self.stroke_spin.setRange(2, 24)
        self.stroke_spin.setValue(10)
        self.row_gap_spin = QSpinBox()
        self.row_gap_spin.setRange(2, 12)
        self.row_gap_spin.setValue(4)
        self.col_gap_spin = QSpinBox()
        self.col_gap_spin.setRange(2, 12)
        self.col_gap_spin.setValue(4)
        self.tracking_spin = QSpinBox()
        self.tracking_spin.setRange(8, 32)
        self.tracking_spin.setValue(16)

        self.text_color_btn = QPushButton("黑 #000000")
        self.text_color_btn.clicked.connect(
            lambda: self._pick_color(self.text_color_btn, "text_color"))
        self.stroke_color_btn = QPushButton("白 #FFFFFF")
        self.stroke_color_btn.clicked.connect(
            lambda: self._pick_color(self.stroke_color_btn, "stroke_color"))
        self.bg_color_combo = QComboBox()
        self.bg_color_combo.addItem("无背景", "none")
        self.bg_color_combo.addItem("白色 255,255,255", "255,255,255")
        self.bg_color_combo.addItem("黑色 0,0,0", "0,0,0")
        self.bg_color_combo.addItem("自定义…", "custom")

        self.box_delta_spin = QSpinBox()
        self.box_delta_spin.setRange(2, 30)
        self.box_delta_spin.setValue(8)
        self.box_tol_spin = QSpinBox()
        self.box_tol_spin.setRange(1, 20)
        self.box_tol_spin.setValue(5)
        self.box_area_spin = QSpinBox()
        self.box_area_spin.setRange(5, 50)
        self.box_area_spin.setValue(20)

        def _row(text, widget):
            row = QHBoxLayout()
            row.addWidget(label(text))
            row.addWidget(widget, 1)
            lay.addLayout(row)
            return row

        ai_title = QLabel("AI 翻译", objectName="cardTitle")
        lay.addWidget(ai_title)
        _row("DeepSeek API Key", self.ds_key_edit)
        ocr_title = QLabel("AI OCR（可选，留空=本地 manga-ocr）", objectName="cardTitle")
        lay.addWidget(ocr_title)
        _row("AI OCR 接口", self.ai_ocr_url_edit)
        _row("AI OCR 模型", self.ai_ocr_model_edit)
        _row("AI OCR Key", self.ai_ocr_key_edit)
        cap_title = QLabel("采集", objectName="cardTitle")
        lay.addWidget(cap_title)
        region_row = _row("采集区域（宽×高，屏 2560×1600 留边界用 2400×1400）", self.region_edit)
        type_title = QLabel("嵌字", objectName="cardTitle")
        lay.addWidget(type_title)
        _row("字号下限（px）", self.min_font_spin)
        _row("字体", self.font_combo)
        _row("描边粗细（字号÷N，大=细，默认 10）", self.stroke_spin)
        _row("行间距（横排，字号÷N，默认 4）", self.row_gap_spin)
        _row("列间距（竖排，字号÷N，默认 4）", self.col_gap_spin)
        _row("字距（字号÷N，默认 16）", self.tracking_spin)
        _row("文字颜色（取色）", self.text_color_btn)
        _row("描边颜色（取色）", self.stroke_color_btn)
        _row("背景色（逐字色块）", self.bg_color_combo)
        scene_row = QHBoxLayout()
        scene_row.addWidget(label("场景预设"))
        scene_light = QPushButton("白底黑字")
        scene_dark = QPushButton("深底白字")
        scene_light.clicked.connect(lambda: self._apply_scene("light"))
        scene_dark.clicked.connect(lambda: self._apply_scene("dark"))
        scene_row.addWidget(scene_light)
        scene_row.addWidget(scene_dark)
        scene_row.addStretch(1)
        lay.addLayout(scene_row)
        box_title = QLabel("蓝框检测（默认值即可，排查时再调）", objectName="cardTitle")
        lay.addWidget(box_title)
        _row("delta 阈值（兜底灵敏度，默认 8）", self.box_delta_spin)
        _row("严格容差（公式匹配容差，默认 5）", self.box_tol_spin)
        _row("整页大框面积比 %（默认 20）", self.box_area_spin)

        save = QPushButton("保存设置")
        save.setProperty("primary", True)
        save.clicked.connect(self._save_settings)
        lay.addWidget(save, alignment=Qt.AlignRight)
        self._load_settings()
        return card

    def _settings_path(self) -> Path:
        return ROOT / "config" / "api_keys.json"

    def _load_settings(self) -> None:
        data = {}
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.ds_key_edit.setText(data.get("deepseek_api_key", ""))
        self.ai_ocr_url_edit.setText(data.get("ai_ocr_url", ""))
        self.ai_ocr_model_edit.setText(data.get("ai_ocr_model", ""))
        self.ai_ocr_key_edit.setText(data.get("ai_ocr_api_key", ""))
        self.region_edit.setText(str(data.get("layout_region", "2400x1400")))
        self.min_font_spin.setValue(int(data.get("min_font_size", 28)))
        font = data.get("font_name", "msyh.ttc")
        index = self.font_combo.findData(font)
        self.font_combo.setCurrentIndex(index if index >= 0 else 0)
        self.stroke_spin.setValue(int(data.get("stroke_divisor", 10)))
        self.row_gap_spin.setValue(int(data.get("row_gap_divisor", 4)))
        self.col_gap_spin.setValue(int(data.get("column_gap_divisor", 4)))
        self.tracking_spin.setValue(int(data.get("tracking_divisor", 16)))
        text_color = data.get("text_color", "0,0,0")
        stroke_color = data.get("stroke_color", "255,255,255")
        self._pending_colors["text_color"] = text_color
        self._pending_colors["stroke_color"] = stroke_color
        self._set_color_btn(self.text_color_btn, "text_color", "0,0,0")
        self._set_color_btn(self.stroke_color_btn, "stroke_color", "255,255,255")
        bg = str(data.get("background_color", "none"))
        index = self.bg_color_combo.findData(bg)
        if index < 0:
            self.bg_color_combo.addItem(f"自定义 {bg}", bg)
            index = self.bg_color_combo.count() - 1
        self.bg_color_combo.setCurrentIndex(index)
        self.box_delta_spin.setValue(int(data.get("box_delta", 8)))
        self.box_tol_spin.setValue(int(data.get("box_tolerance", 5)))
        self.box_area_spin.setValue(int(data.get("box_area_ratio", 20)))

    def _color_str(self, key: str, default: str) -> str:
        data = {}
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return str(data.get(key, default))

    def _set_color_btn(self, btn: QPushButton, key: str, default: str) -> None:
        value = self._color_str(key, default)
        btn.setText(f"{value.replace(',', ',')} #{''.join('%02X' % int(v) for v in value.split(','))}" if "," in value else value)

    def _pick_color(self, btn: QPushButton, key: str) -> None:
        from PySide6.QtWidgets import QColorDialog

        current = self._color_str(key, "0,0,0" if key == "text_color" else "255,255,255")
        parts = [int(v) for v in current.split(",")] if "," in current else [0, 0, 0]
        color = QColorDialog.getColor(QColor(*parts), self, "选择颜色")
        if not color.isValid():
            return
        value = f"{color.red()},{color.green()},{color.blue()}"
        btn.setText(f"{value} #{color.name().lstrip('#').upper()}")
        self._pending_colors[key] = value

    def _apply_scene(self, scene: str) -> None:
        if scene == "light":
            self._pending_colors["text_color"] = "0,0,0"
            self._pending_colors["stroke_color"] = "255,255,255"
            self.text_color_btn.setText("黑 #000000")
            self.stroke_color_btn.setText("白 #FFFFFF")
            self.bg_color_combo.setCurrentIndex(0)  # 无背景
        else:
            self._pending_colors["text_color"] = "255,255,255"
            self._pending_colors["stroke_color"] = "0,0,0"
            self.text_color_btn.setText("白 #FFFFFF")
            self.stroke_color_btn.setText("黑 #000000")
            index = self.bg_color_combo.findData("0,0,0")
            self.bg_color_combo.setCurrentIndex(index if index >= 0 else 0)

    def _save_settings(self) -> None:
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "deepseek_api_key": self.ds_key_edit.text().strip(),
            "ai_ocr_url": self.ai_ocr_url_edit.text().strip(),
            "ai_ocr_model": self.ai_ocr_model_edit.text().strip(),
            "ai_ocr_api_key": self.ai_ocr_key_edit.text().strip(),
            "layout_region": self.region_edit.text().strip() or "2400x1400",
            "min_font_size": self.min_font_spin.value(),
            "font_name": self.font_combo.currentData() or "msyh.ttc",
            "stroke_divisor": self.stroke_spin.value(),
            "row_gap_divisor": self.row_gap_spin.value(),
            "column_gap_divisor": self.col_gap_spin.value(),
            "tracking_divisor": self.tracking_spin.value(),
            "text_color": self._pending_colors.get(
                "text_color", self._color_str("text_color", "0,0,0")),
            "stroke_color": self._pending_colors.get(
                "stroke_color", self._color_str("stroke_color", "255,255,255")),
            "background_color": self.bg_color_combo.currentData(),
            "box_delta": self.box_delta_spin.value(),
            "box_tolerance": self.box_tol_spin.value(),
            "box_area_ratio": self.box_area_spin.value(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log("设置已保存（config/api_keys.json）")

    def _build_glossary_tab(self) -> QFrame:
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        card, lay = self._card("术语库（人名/术语固定译名，翻译即时生效）")
        self.term_table = QTableWidget(0, 3)
        self.term_table.setHorizontalHeaderLabels(["原文（日文）", "译文", "备注"])
        self.term_table.horizontalHeader().setStretchLastSection(True)
        self.term_table.setColumnWidth(0, 300)
        self.term_table.setColumnWidth(1, 200)
        lay.addWidget(self.term_table, 1)

        edit_row = QHBoxLayout()
        edit_row.addWidget(label("原文"))
        self.term_src_edit = QLineEdit()
        edit_row.addWidget(self.term_src_edit, 1)
        edit_row.addWidget(label("译文"))
        self.term_dst_edit = QLineEdit()
        edit_row.addWidget(self.term_dst_edit, 1)
        edit_row.addWidget(label("备注"))
        self.term_note_edit = QLineEdit()
        edit_row.addWidget(self.term_note_edit, 1)
        lay.addLayout(edit_row)

        btn_row = QHBoxLayout()
        add = QPushButton("新增")
        add.setProperty("primary", True)
        add.clicked.connect(self._term_add)
        update = QPushButton("更新选中")
        update.clicked.connect(self._term_update)
        remove = QPushButton("删除选中")
        remove.setProperty("danger", True)
        remove.clicked.connect(self._term_remove)
        reload = QPushButton("重新加载")
        reload.clicked.connect(self._term_reload)
        for b in (add, update, remove, reload):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        self.term_table.itemSelectionChanged.connect(self._term_select)
        return card

    def _glossary_path(self) -> Path:
        return ROOT / "config" / "glossary.json"

    def _glossary_data(self) -> dict:
        try:
            return json.loads(self._glossary_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "terms": {}, "names": {}, "notes": {}}

    def _term_reload(self) -> None:
        from PySide6.QtWidgets import QTableWidgetItem

        self.term_table.setRowCount(0)
        terms = self._glossary_data().get("terms", {})
        for index, (source, entry) in enumerate(sorted(terms.items())):
            self.term_table.insertRow(index)
            self.term_table.setItem(index, 0, QTableWidgetItem(source))
            self.term_table.setItem(index, 1, QTableWidgetItem(str(entry.get("translation", ""))))
            self.term_table.setItem(index, 2, QTableWidgetItem(str(entry.get("note", ""))))
        self._log(f"术语库已加载：{len(terms)} 条")

    def _term_select(self) -> None:
        row = self.term_table.currentRow()
        if row < 0:
            return
        self.term_src_edit.setText(self.term_table.item(row, 0).text())
        self.term_dst_edit.setText(self.term_table.item(row, 1).text())
        self.term_note_edit.setText(self.term_table.item(row, 2).text())

    def _term_save(self, terms: dict) -> None:
        data = self._glossary_data()
        data["terms"] = terms
        self._glossary_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _term_add(self) -> None:
        source = self.term_src_edit.text().strip()
        translation = self.term_dst_edit.text().strip()
        if not source or not translation:
            QMessageBox.warning(self, "提示", "原文和译文不能为空")
            return
        terms = self._glossary_data().get("terms", {})
        terms[source] = {
            "translation": translation,
            "note": self.term_note_edit.text().strip(),
        }
        self._term_save(terms)
        self._term_reload()

    def _term_update(self) -> None:
        row = self.term_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在表格中选择要更新的条目")
            return
        source = self.term_table.item(row, 0).text()
        terms = self._glossary_data().get("terms", {})
        if source not in terms:
            return
        terms[source] = {
            "translation": self.term_dst_edit.text().strip(),
            "note": self.term_note_edit.text().strip(),
        }
        self._term_save(terms)
        self._term_reload()

    def _term_remove(self) -> None:
        row = self.term_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在表格中选择要删除的条目")
            return
        source = self.term_table.item(row, 0).text()
        answer = QMessageBox.question(self, "删除术语", f"删除「{source}」？")
        if answer != QMessageBox.Yes:
            return
        terms = self._glossary_data().get("terms", {})
        terms.pop(source, None)
        self._term_save(terms)
        self._term_reload()

    def _build_review_tab(self) -> QFrame:
        card, lay = self._card("审核（三次采集无蓝色像素的页面）")
        self.review_list = QListWidget()
        lay.addWidget(self.review_list, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_review)
        confirm = QPushButton("确认这些页无框")
        confirm.setProperty("primary", True)
        confirm.clicked.connect(self._confirm_review)
        h = QHBoxLayout()
        h.addWidget(refresh)
        h.addWidget(confirm)
        h.addStretch(1)
        lay.addLayout(h)
        return card

    # ---- tab switching ----
    def _show_tab(self, index: int) -> None:
        for w, i in (
            (self.tab_capture, 0),
            (self.tab_process, 1),
            (self.tab_review, 2),
            (self.tab_preview, 3),
            (self.tab_glossary, 4),
            (self.tab_settings, 5),
        ):
            w.setVisible(i == index)
        if index == 2:
            self._refresh_review()
        if index == 4:
            self._term_reload()
        if index == 3:
            self._pages_list = self._preview_images()
            if self._pages_list:
                self._goto_page(1)
            else:
                self.page_total_label.setText(" / 0（先采集或选择目录）")

    # ---- progress polling (run.json) ----
    def _poll_progress(self) -> None:
        images = self.images_edit.text().strip()
        if not images:
            return
        run_file = (
            ROOT / "outputs" / "oneclick" / Path(images).name / "run.json"
        )
        if not run_file.is_file():
            return
        try:
            status = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        stage = status.get("stage", "idle")
        total = status.get("total", 0) or 0
        done = status.get("done", 0) or 0
        item = status.get("current_item", "")
        elapsed = status.get("elapsed", 0)
        eta = status.get("eta", 0)
        finished = status.get("finished", False)
        self.stage_label.setText(f"阶段：{stage}（{status.get('stage_index', 0) + 1}/{status.get('total_stages', 6)}）")
        self.item_label.setText(f"当前：{item or '—'}")
        ratio = done / total if total else 0
        self.progress_bar.setValue(int(ratio * 1000))
        self.progress_bar.setFormat(f"{done}/{total} · {ratio * 100:.0f}%")
        self.eta_label.setText(
            f"耗时 {int(elapsed)}s　预计剩余 {int(eta)}s" if not finished
            else f"已结束（{status.get('ok', True) and '成功' or '失败'}）"
        )

    # ---- helpers ----
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择原图目录")
        if path:
            self.images_edit.setText(path)

    def _log(self, text: str) -> None:
        self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _images(self) -> str | None:
        value = self.images_edit.text().strip()
        if not value:
            QMessageBox.warning(self, "提示", "请先选择原图目录")
            return None
        return value

    def _base_args(self) -> list[str]:
        args = ["--images", self._images()]
        pages = self.pages_edit.text().strip()
        if pages:
            args += ["--pages", pages]
        args += ["--parallel", str(self.parallel_spin.value())]
        args += ["--display-scale", str(self.scale_spin.value())]
        return args

    def _detection_args(self) -> list[str]:
        args = []
        region = self.region_edit.text().strip()
        if region:
            args += ["--layout-region", region]
        args += ["--box-delta", str(self.box_delta_spin.value())]
        args += ["--box-tolerance", str(self.box_tol_spin.value())]
        args += ["--box-area-ratio", str(self.box_area_spin.value() / 100.0)]
        args += ["--min-font-size", str(self.min_font_spin.value())]
        args += ["--font", self.font_combo.currentData() or "msyh.ttc"]
        args += ["--stroke-divisor", str(self.stroke_spin.value())]
        args += ["--row-gap-divisor", str(self.row_gap_spin.value())]
        args += ["--column-gap-divisor", str(self.col_gap_spin.value())]
        args += ["--tracking-divisor", str(self.tracking_spin.value())]
        args += ["--text-color", self._pending_colors.get(
            "text_color", self._color_str("text_color", "0,0,0"))]
        args += ["--stroke-color", self._pending_colors.get(
            "stroke_color", self._color_str("stroke_color", "255,255,255"))]
        bg = self.bg_color_combo.currentData()
        if bg and bg != "none":
            args += ["--background-color", bg]
        args += ["--max-workers", str(self.workers_spin.value())]
        return args

    def _ai_args(self) -> list[str]:
        args = []
        key = self.ds_key_edit.text().strip()
        if key:
            args += ["--ai-api-key", key]
        ocr_url = self.ai_ocr_url_edit.text().strip()
        ocr_model = self.ai_ocr_model_edit.text().strip()
        ocr_key = self.ai_ocr_key_edit.text().strip()
        if ocr_url and ocr_model and ocr_key:
            args += ["--ai-ocr-url", ocr_url, "--ai-ocr-model", ocr_model,
                     "--ai-ocr-api-key", ocr_key]
        return args

    def _start_capture(self) -> None:
        if not self._images():
            return
        self._run([*self._base_args(), *self._detection_args()])

    def _start_process(self) -> None:
        if not self._images():
            return
        args = [*self._base_args(), "--process", *self._ai_args(),
                *self._detection_args()]
        if self.resume_check.isChecked():
            args.append("--resume")
        if not self.pdf_check.isChecked():
            args.append("--no-pdf")
        fill = self.fill_combo.currentData()
        if fill:
            args += ["--bluebox-fill", fill]
        self._run(args)

    def _run(self, args: list[str]) -> None:
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self, "提示", "已有任务在运行")
            return
        self.capture_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self._log("$ python one_click_translate.py " + " ".join(args))
        self.runner = Runner(args)
        self.runner.line.connect(self._log)
        self.runner.finished_ok.connect(self._on_done)
        self.runner.start()

    def _on_done(self, ok: bool) -> None:
        self.capture_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self._log("== 任务结束（%s）==" % ("成功" if ok else "失败"))
        self._refresh_review()

    # ---- review ----
    def _review_pages(self) -> list[str]:
        images = self.images_edit.text().strip()
        if not images:
            return []
        out = ROOT / "outputs" / "oneclick" / Path(images).name / "capture"
        if not out.is_dir():
            return []
        review = []
        for page in sorted((p for p in out.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            manifest = page / "capture.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            cap = data.get("capture", {})
            if cap.get("selection_review_required") and not cap.get("selection_review_confirmed"):
                review.append(page.name)
        return review

    def _refresh_review(self) -> None:
        self.review_list.clear()
        for page in self._review_pages():
            self.review_list.addItem(QListWidgetItem(page))

    def _confirm_review(self) -> None:
        images = self._images()
        if not images:
            return
        pages = self._review_pages()
        if not pages:
            QMessageBox.information(self, "提示", "当前没有待审核页")
            return
        answer = QMessageBox.question(
            self, "确认无框",
            "已人工确认这些页确实没有蓝色选区框？（确认后 stage2 将静默跳过）\n" + ", ".join(pages),
        )
        if answer != QMessageBox.Yes:
            return
        self._run(["--images", images, "--confirm-no-boxes", ",".join(pages)])
        self._refresh_review()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
