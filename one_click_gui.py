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
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

def _find_root() -> Path:
    """Project root containing venv/（exe 在哪一层都能找到）。

    打包运行时 ``__file__`` 指向临时解压目录，所以从 **exe 位置**往上找：
    onefile 放在 ``dist/``（上一层就是项目根），onedir 放在
    ``dist/<名字>/``（要上溯两层）—— 统一按最多 4 层祖先扫描，比写死层数稳。
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    candidates = [base, *(base.parents[:4])]
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
    "bg": "#000000", "card": "#121212", "border": "#262626",
    "border_hover": "#3a3a3a", "hover": "#1a1a1a",
    "text": "#e5e5e5", "text_sec": "#a3a3a3", "text_mut": "#737373",
    "primary": "#2dd4bf", "primary_hover": "#14b8a6",
    "primary_soft": "rgba(45,212,191,0.18)", "primary_text": "#042f2e",
    "danger": "#f87171", "warning": "#fbbf24",
    "disabled_primary": "rgba(45,212,191,0.35)",
    "selected_text": "#042f2e",
    "nav_selected_bg": "#2dd4bf",
    "nav_selected_text": "#042f2e",
}


def build_qss(t: dict) -> str:
    """QSS for the given token set (light/dark share the same layout)."""
    return f"""
* {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 14px; color: {t['text']}; }}
QMainWindow {{ background: {t['bg']}; }}
QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
  background: {t['bg']}; border: none;
}}
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
  background: {t['card']}; color: {t['text']};
  border: 1px solid {t['border']}; border-radius: 8px;
  padding: 8px 10px; font-size: 13px;
}}
"""


CAPTURE_HELP = (
    "<b>采集规矩</b><br>"
    "1. 每页采三个帧：原图帧 / 扫描文本帧 / 全选帧；<br>"
    "   蓝框检测用 <b>扫描文本帧 + 全选帧 + 原图帧</b> 三帧。<br>"
    "2. 窗口 = 页面 × 显示缩放%（上限，放不下自动缩小），<br>"
    "   6 窗 3×2 并行、紧贴四边对齐；<br>"
    "   请勿手动移动/遮挡窗口（程序会实时锁位）。<br>"
    "3. 检测<b>单路径</b>：严格蓝色蒙版公式<br>"
    "   <code>|selected - (keep×base + (0,38,67))| ≤ 5</code>，<br>"
    "   keep=174/255，paint-delta 守卫排除页面固有纯蓝。<br>"
    "4. 整页大框（面积&gt;帧20%）自动丢弃，不是正文单位。<br>"
    "5. 硬规则：缺 base 帧 / 三次 Ctrl+A 无蓝色像素 /<br>"
    "   采集异常 → 该页进入 <b>待审核</b>（审核页确认无框或全部重采）。<br>"
    "6. 采集期间勿动鼠标键盘（窗口自动抢焦点）。"
)


class Runner(QThread):
    """Run one_click_translate.py as a subprocess, streaming output."""

    line = Signal(str)
    finished_ok = Signal(bool)
    cancelled = Signal()

    def __init__(self, args: list[str]) -> None:
        super().__init__()
        self.args = args
        self.proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self) -> None:
        """Terminate the whole child process tree; safe from the GUI thread.

        ``proc.terminate()`` only kills one_click_translate.py itself; its
        children (photos_capture.py, the Photos windows it opens) would
        survive as orphans and keep stacking windows on the desktop.
        ``taskkill /T /F`` kills the entire tree rooted at this process,
        and the residual-window sweep closes any Photos window whose
        title is a page name we were capturing.
        """
        self._cancelled = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                import subprocess as _sp

                _sp.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                try:
                    proc.terminate()
                except OSError:
                    pass
        # Sweep orphan Photos windows the killed tree may have opened:
        # any window whose title is one of the page file names from the
        # --images directory.
        try:
            import os as _os
            from photos_capture import _close_residual_photos_windows

            images_arg = None
            for index, item in enumerate(self.args):
                if item == "--images" and index + 1 < len(self.args):
                    images_arg = self.args[index + 1]
                    break
            if images_arg:
                names = {
                    name
                    for name in _os.listdir(images_arg)
                    if name.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                    )
                }
                _close_residual_photos_windows(names)
        except Exception:
            pass

    def run(self) -> None:
        cmd = [str(PYTHON), str(SCRIPT), *self.args]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            self.line.emit("!! 找不到 venv Python，请检查安装")
            self.finished_ok.emit(False)
            return
        proc = self.proc
        for text in proc.stdout:
            self.line.emit(text.rstrip())
        code = proc.wait()
        if self._cancelled:
            self.line.emit("!! 任务已取消")
            self.cancelled.emit()
        else:
            self.finished_ok.emit(code == 0)


# 框来源（顺序 = 读取优先级：paddle 优先）。
# 采集结果按来源分开存放：capture/{source}/ 与 check/{source}/，
# 所以界面上取图/取采集都要按来源逐个找。dataset/typeset 只有一套。
_BOX_SOURCES = (("paddle", "PaddleOCR"), ("photos", "Photos"))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("漫画翻译软件")
        self.resize(1180, 860)
        icon = ROOT / "app.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.runner: Runner | None = None
        self.dark = self._load_theme(self._theme_path())
        self._pending_colors: dict[str, str] = {}
        self._build()
        self._apply_theme()

    @staticmethod
    def _theme_path() -> Path:
        return ROOT / "config" / "ui.json"

    @staticmethod
    def _load_theme(path: Path) -> bool:
        """Read the persisted dark-mode flag; default to dark."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return bool(data.get("dark", True))
        except (OSError, ValueError):
            return True

    @staticmethod
    def _save_theme(path: Path, dark: bool) -> None:
        """Persist the dark flag, preserving any other keys in ui.json."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
            data["dark"] = bool(dark)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError:
            pass

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(DARK if self.dark else LIGHT))
        self.theme_btn.setText("🌙 暗色模式" if not self.dark else "☀️ 亮色模式")
        self._style_progress()
        tokens = DARK if self.dark else LIGHT
        for btn in self.findChildren(QPushButton, "helpBtn"):
            btn.setStyleSheet(
                "QPushButton { border: 2px solid %s; border-radius: 15px; "
                "color: %s; font-size: 17px; font-weight: 700; background: transparent; }"
                "QPushButton:hover { background: %s; }"
                % (tokens["primary"], tokens["primary"], tokens["primary_soft"])
            )

    def _toggle_theme(self) -> None:
        self.dark = not self.dark
        self._apply_theme()
        self._save_theme(self._theme_path(), self.dark)

    def closeEvent(self, event) -> None:
        runner = self.runner
        if runner is not None and runner.isRunning():
            answer = QMessageBox.question(
                self, "退出确认",
                "后台任务仍在运行，退出将终止处理进程。确定退出？",
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._log("== 正在取消后台任务… ==")
            runner.cancel()
            if not runner.wait(5000):
                # Last resort: the subprocess did not exit within 5 s.
                runner.terminate()
                runner.wait(1000)
        event.accept()

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
        sub = label("框: PaddleOCR 检测 / Photos 蓝框 双路径 · AI 翻译嵌字")
        side_l.addWidget(sub)
        side_l.addSpacing(24)
        nav = QListWidget()
        for item in ("采集", "审核", "处理", "预览", "术语", "设置"):
            nav.addItem(QListWidgetItem(item))
        nav.setCurrentRow(0)
        side_l.addWidget(nav, 1)
        self.nav = nav
        self.theme_btn = QPushButton()
        self.theme_btn.setProperty("ghost", True)
        self.theme_btn.clicked.connect(self._toggle_theme)
        side_l.addWidget(self.theme_btn)
        root.addWidget(side)

        # ---- content (scrollable so nothing is cut off on small windows) ----
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

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
        self.row_spacing_spin = QSpinBox()
        self.row_spacing_spin.setRange(0, 600)
        self.row_spacing_spin.setValue(200)
        self.row_spacing_spin.setToolTip(
            "上下两排窗口的垂直间距（px），0 = 不传参（CLI 默认 200）"
        )
        self.right_gap_spin = QSpinBox()
        self.right_gap_spin.setRange(0, 600)
        self.right_gap_spin.setValue(100)
        self.right_gap_spin.setToolTip(
            "最右侧窗口距右边缘的间隙（px），0 = 贴边"
        )

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
        # Keep the capture and process directory rows in sync: editing or
        # browsing in either tab updates both, so the process tab never
        # asks for a directory the user already picked on the capture tab.
        self.images_edit.textChanged.connect(
            lambda text: self.process_images_edit.setText(text)
        )
        self.process_images_edit.textChanged.connect(
            lambda text: self.images_edit.setText(text)
        )
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
            help_btn.setObjectName("helpBtn")
            help_btn.setProperty("ghost", True)
            help_btn.setFixedSize(30, 30)
            help_btn.setCursor(Qt.PointingHandCursor)
            help_btn.setToolTip(help_text)
            row.addWidget(help_btn)
            lay.addLayout(row)
        else:
            lay.addWidget(lab)
        return card, lay

    def _build_capture_tab(self, browse: QPushButton) -> QFrame:
        card, lay = self._card("采集", help_text=CAPTURE_HELP)
        self.capture_card = card
        row0 = QHBoxLayout()
        row0.addWidget(label("框来源"))
        self.box_source_capture_combo = QComboBox()
        self.box_source_capture_combo.addItem("PaddleOCR 全页检测（默认，离线快）", "paddle")
        self.box_source_capture_combo.addItem("Photos 蓝框采集（窗口自动化）", "photos")
        self.box_source_capture_combo.setToolTip(
            "与「设置」页同一选项（双向同步）。\n"
            "paddle：离线检测、不开 Photos 窗口，下面一排窗口参数不生效；\n"
            "photos：原流程，需窗口自动化，下面的窗口参数才生效。"
        )
        row0.addWidget(self.box_source_capture_combo)
        row0.addStretch(1)
        self.capture_hint = label("")
        row0.addWidget(self.capture_hint)
        lay.addLayout(row0)
        # 双向同步：采集页与设置页的下拉保持一致
        # （设置页那个下拉在本页之后才构建，它的信号在设置页里连）
        self.box_source_capture_combo.currentIndexChanged.connect(
            lambda _index: self._sync_box_source(self.box_source_capture_combo)
        )
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
        # Photos 专属参数（窗口布局）：Paddle 路径下禁用并说明
        self.capture_photos_only = [
            self.parallel_spin, self.scale_spin,
            self.row_spacing_spin, self.right_gap_spin,
        ]
        row2b = QHBoxLayout()
        row2b.addWidget(label("上下行距"))
        row2b.addWidget(self.row_spacing_spin)
        row2b.addSpacing(16)
        row2b.addWidget(label("右边界间隙"))
        row2b.addWidget(self.right_gap_spin)
        row2b.addSpacing(16)
        self.capture_overwrite_check = QCheckBox("覆盖旧采集（重采选中页）")
        self.capture_overwrite_check.setChecked(True)
        row2b.addWidget(self.capture_overwrite_check)
        row2b.addStretch(1)
        lay.addLayout(row2b)
        btn = QPushButton("开始采集")
        btn.setProperty("primary", True)
        btn.clicked.connect(self._start_capture)
        stop_btn = QPushButton("停止采集")
        stop_btn.setProperty("danger", True)
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(self._stop_task)
        row_btns = QHBoxLayout()
        row_btns.addStretch(1)
        row_btns.addWidget(stop_btn)
        row_btns.addWidget(btn)
        lay.addLayout(row_btns)
        self.capture_btn = btn
        self.stop_btn = stop_btn
        return card

    def _sync_box_source(self, source_widget=None) -> None:
        """采集页与设置页的框来源双向同步，并按来源启用/禁用窗口参数。

        Paddle 路径不需要窗口（不开 Photos），并行窗口/显示缩放/行距/间隙
        这些参数完全不生效——留着可点会让人误以为改了有用。
        """
        capture_combo = getattr(self, "box_source_capture_combo", None)
        settings_combo = getattr(self, "box_source_combo", None)
        if capture_combo is None or settings_combo is None:
            # 两个页签都建好之前不做同步（采集页先于设置页构建）
            return
        target = settings_combo if source_widget is capture_combo else capture_combo
        if source_widget is not None and target.currentData() != source_widget.currentData():
            target.blockSignals(True)
            index = target.findData(source_widget.currentData())
            if index >= 0:
                target.setCurrentIndex(index)
            target.blockSignals(False)
        source = (source_widget or settings_combo).currentData() or "paddle"
        photos = source == "photos"
        for widget in getattr(self, "capture_photos_only", []):
            widget.setEnabled(photos)
        hint = getattr(self, "capture_hint", None)
        if hint is not None:
            hint.setText(
                "" if photos
                else "Paddle 检测不需要窗口，下列窗口参数不生效"
            )
        card = getattr(self, "capture_card", None)
        if card is not None:
            titles = card.findChildren(QLabel, "cardTitle")
            if titles:
                titles[0].setText(
                    "采集（当前框来源：{}）".format(
                        "Photos 蓝框采集" if photos else "PaddleOCR 全页检测"
                    )
                )

    def _build_process_tab(self) -> QFrame:
        card, lay = self._card("处理（OCR → 翻译 → 擦字 → 嵌字）")
        note = label("后台处理：默认全链（OCR → 翻译 → 擦字 → 嵌字 → PDF），"
                     "也可只跑某一阶段（例如改完翻译提示词后只重翻译选中页）。"
                     "待审核页会跳过，需先在“审核”页确认。")
        note.setWordWrap(True)
        lay.addWidget(note)
        # The process tab needs its own directory row: a QWidget cannot
        # live in two layouts, so this is a second edit box that stays in
        # sync with the capture tab's (either one updates both).
        row0 = QHBoxLayout()
        row0.addWidget(label("原图目录"))
        self.process_images_edit = QLineEdit()
        self.process_images_edit.setPlaceholderText(
            "与采集页同步；处理前需先选择原图目录"
        )
        row0.addWidget(self.process_images_edit, 1)
        process_browse = QPushButton("浏览…")
        process_browse.clicked.connect(self._browse_process)
        row0.addWidget(process_browse)
        lay.addLayout(row0)
        row = QHBoxLayout()
        row.addWidget(label("处理范围"))
        self.stage_combo = QComboBox()
        self.stage_combo.addItem("全部处理（OCR→翻译→嵌字）", "all")
        self.stage_combo.addItem("仅 OCR（蓝框读字）", "ocr")
        self.stage_combo.addItem("仅翻译", "translate")
        self.stage_combo.addItem("仅嵌字（擦字+嵌字+PDF）", "typeset")
        row.addWidget(self.stage_combo)
        row.addSpacing(16)
        self.overwrite_check = QCheckBox("覆盖旧结果（重跑即重做）")
        self.overwrite_check.setChecked(True)
        self.overwrite_check.toggled.connect(
            lambda checked: self.resume_check.setChecked(False) if checked else None)
        self.resume_check = QCheckBox("断点续跑（跳过已有 typeset 的页）")
        self.resume_check.toggled.connect(
            lambda checked: self.overwrite_check.setChecked(False) if checked else None)
        row.addWidget(self.overwrite_check)
        row.addSpacing(16)
        row.addWidget(self.resume_check)
        row.addStretch(1)
        lay.addLayout(row)
        row_pdf = QHBoxLayout()
        self.pdf_check = QCheckBox("生成 PDF（默认不生成，只出 typeset 图）")
        row_pdf.addWidget(self.pdf_check)
        row_pdf.addStretch(1)
        lay.addLayout(row_pdf)
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
        self.workers_spin.setValue(3)
        self.workers_spin.setToolTip(
            "并行处理（翻译+擦字+嵌字）的页数，上限 3。\n"
            "擦字是最重的一步（实测 ~12s/页，FSR 修复占八成），3 路约 3× 提速。"
        )
        row2.addWidget(self.workers_spin)
        row2.addStretch(1)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(label("页码范围"))
        self.process_pages_edit = QLineEdit()
        self.process_pages_edit.setPlaceholderText(
            "留空=全部，如 41,55,58 或 110-160（只翻译嵌字这些页）"
        )
        row3.addWidget(self.process_pages_edit, 1)
        row3.addStretch(1)
        lay.addLayout(row3)
        btn = QPushButton("开始处理")
        btn.setProperty("primary", True)
        btn.clicked.connect(self._start_process)
        stop_btn = QPushButton("停止处理")
        stop_btn.setProperty("danger", True)
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(self._stop_task)
        row_btns = QHBoxLayout()
        row_btns.addStretch(1)
        row_btns.addWidget(stop_btn)
        row_btns.addWidget(btn)
        lay.addLayout(row_btns)
        self.process_btn = btn
        self.process_stop_btn = stop_btn
        return card

    def _style_progress(self) -> None:
        tokens = DARK if self.dark else LIGHT
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 12px; background: %s; "
            "height: 14px; text-align: center; font-size: 11px; color: %s; }"
            "QProgressBar::chunk { border-radius: 12px; background: %s; }"
            % (tokens["border"], tokens["text"], tokens["primary"])
        )

    def _build_preview_tab(self) -> QFrame:
        card, lay = self._card("预览（原图 / 检出框 check 图 / 译文）")
        tool = QHBoxLayout()
        self.prev_page_btn = QPushButton("上一页")
        self.next_page_btn = QPushButton("下一页")
        self.page_edit = QLineEdit()
        self.page_edit.setFixedWidth(70)
        self.page_total_label = label(" / —")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("原图", "original")
        self.mode_combo.addItem("检出框（Paddle 红 / Photos 蓝）", "check")
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
        pages = set()
        for page in out.glob("capture/*/capture.json"):
            try:
                data = json.loads(page.read_text(encoding="utf-8-sig"))
                src = data.get("source", {}).get("path")
                if src:
                    pages.add(page.parent.name)
            except (OSError, ValueError):
                continue
        # Numeric page names sort by numeric value, then non-numeric names
        # alphabetically.  The previous mixed int/str key raised TypeError
        # when a book had both ("cover", "1").
        def _preview_page_key(name: str) -> tuple[int, int | str]:
            if name.isdigit():
                return (0, int(name))
            return (1, name.lower())

        return sorted(pages, key=_preview_page_key)

    def _preview_path(self, page: str) -> str | None:
        images = self.images_edit.text().strip()
        if not images:
            return None
        out = ROOT / "outputs" / "oneclick" / Path(images).name
        if self.mode_combo.currentData() == "typeset":
            typeset = out / "typeset" / f"{page}.typeset.png"
            if typeset.is_file():
                return str(typeset)
            # No typeset output for this page (untouched / not processed):
            # fall through to the original image below instead of showing
            # a blank preview.
        if self.mode_combo.currentData() == "check":
            for source, _label in _BOX_SOURCES:
                check = out / "check" / source / f"{page}.bluebox_check.png"
                if check.is_file():
                    return str(check)
            for name in (f"{page}.bluebox_check.png", f"{page}.jpg"):
                check = out / "check" / name
                if check.is_file():
                    return str(check)
        manifest = None
        for source, _label in _BOX_SOURCES:
            candidate = out / "capture" / source / page / "capture.json"
            if candidate.is_file():
                manifest = candidate
                break
        if manifest is not None:
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
        text = self.page_edit.text().strip()
        if not text:
            return
        # Exact page-name match first (supports non-numeric names like
        # "cover" or zero-padded "001").
        if text in self._pages_list:
            self._current_page = self._pages_list.index(text) + 1
            self._reload_page()
            return
        try:
            number = int(text)
        except ValueError:
            return
        for index, name in enumerate(self._pages_list):
            if name.isdigit() and int(name) == number:
                self._current_page = index + 1
                self._reload_page()
                return
        self._goto_page(number)

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
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItem("自动（默认，Windows OCR → Paddle 兜底）", "auto")
        self.ocr_engine_combo.addItem("Windows OCR", "windows-ocr")
        self.ocr_engine_combo.addItem("PaddleOCR", "paddle")
        self.stage2_engine_combo = QComboBox()
        self.stage2_engine_combo.addItem("manga-ocr（本地，默认）", "manga-ocr")
        self.stage2_engine_combo.addItem("AI 视觉 OCR", "ai-ocr")
        self.box_source_combo = QComboBox()
        self.box_source_combo.addItem("PaddleOCR 全页检测（默认，离线快）", "paddle")
        self.box_source_combo.addItem("Photos 蓝框采集（窗口自动化）", "photos")
        self.box_source_combo.setToolTip(
            "采集期的框来源。两者产出同一份 capture.json 契约，下游完全一致。\n"
            "paddle：离线检测、无窗口自动化、GPU 下 32 页约 7 秒；\n"
            "photos：原流程，需开 Photos 窗口并做焦点/时序控制。"
        )
        self.drop_large_boxes_check = QCheckBox("剔除大拟音/效果字框（Paddle 路径）")
        self.drop_large_boxes_check.setChecked(True)
        self.drop_large_boxes_check.setToolTip(
            "剔除大号拟音/效果字框（如 ドッ/ズズッ 这类风格化大字）：\n"
            "判据 = 框短边 > 2×同方向基准 且 > 页短边 10%。\n"
            "按方向分组比较（横比横、竖比竖），横排文字行不会被误删。"
        )
        self.paddle_device_combo = QComboBox()
        self.paddle_device_combo.addItem("自动（有 GPU 就用，默认）", "auto")
        self.paddle_device_combo.addItem("GPU", "gpu")
        self.paddle_device_combo.addItem("CPU", "cpu")
        self.paddle_device_combo.setToolTip(
            "PaddleOCR（采集期）运行设备。GPU 实测比 CPU 快约 7.6 倍；\n"
            "注意与本地 Sakura 抢占显存（8GB 卡同时跑可能 OOM，此时选 CPU）。"
        )

        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("2400x1400")
        self.min_font_spin = QSpinBox()
        self.min_font_spin.setRange(12, 80)
        self.min_font_spin.setValue(28)
        self.sfx_font_spin = QSpinBox()
        self.sfx_font_spin.setRange(12, 80)
        self.sfx_font_spin.setValue(30)
        self.sfx_font_spin.setToolTip(
            "拟声词嵌字字号上限（默认 30）。\n"
            "拟声框在原图里又大又粗，字号若跟着框走会嵌出一个盖过对白的巨型"
            "中文拟声；\n上限只封顶不放大（框小仍按框自适应）。"
        )
        self.ai_ocr_workers_spin = QSpinBox()
        self.ai_ocr_workers_spin.setRange(1, 32)
        self.ai_ocr_workers_spin.setValue(12)
        self.ai_ocr_workers_spin.setToolTip(
            "AI OCR 并发请求数（默认 12，1 = 串行，护栏 32）。\n"
            "实测 1 路 1.91s/框 → 8 路 0.20s/框（9×）→ 16 路 0.10s/框（19×），"
            "24 路反而退化。\n机器或账号更给力可以调高，撞服务端限流（429）就调小。"
        )
        self.sfx_ratio_spin = QDoubleSpinBox()
        self.sfx_ratio_spin.setRange(1.0, 5.0)
        self.sfx_ratio_spin.setSingleStep(0.1)
        self.sfx_ratio_spin.setValue(2.0)
        self.sfx_ratio_spin.setToolTip(
            "过大单框判定倍数（默认 2.0）：未合并的单框，短边 > N×同方向平均 →\n"
            "按拟声词把字号封到「拟声词字号上限」。这个倍数同时用于采集期剔除\n"
            "超大拟音框（--sfx-min-side-ratio）。"
        )
        self.page_ceiling_spin = QDoubleSpinBox()
        self.page_ceiling_spin.setRange(0.0, 4.0)
        self.page_ceiling_spin.setSingleStep(0.1)
        self.page_ceiling_spin.setValue(1.5)
        self.page_ceiling_spin.setToolTip(
            "整页字号上限（默认 1.5，0 = 关闭）：译文字号不超过该页「常规字号」的 N 倍。\n"
            "常规字号 = 该页蓝框短边中位数（页面主体文字的尺寸）。\n"
            "各岛各自尽量撑满会让同一页出现 2 倍级差（实测页 7：131px vs 基准 66px）。"
        )
        self.short_source_spin = QSpinBox()
        self.short_source_spin.setRange(0, 20)
        self.short_source_spin.setValue(0)
        self.short_source_spin.setToolTip(
            "短原文封顶门槛（默认 0 = 关闭）：原文（日文）≤ N 字时字号也封顶。\n"
            "注意实测：开启 6 会让 127 个岛面积比掉到 0.02~0.06（短原文的框本来\n"
            "就大，字号跟框走才对），所以默认关闭。"
        )
        self.sfx_glossary_spin = QSpinBox()
        self.sfx_glossary_spin.setRange(0, 20)
        self.sfx_glossary_spin.setValue(2)
        self.sfx_glossary_spin.setToolTip(
            "拟声词自动进术语库的门槛（默认 2，0 = 关闭）。\n"
            "同一原文 + 同一译法出现 N 次才收录，收录后全书统一译法；\n"
            "同一原文出现两种译法时自动撤出（不锁死可能错的译法）。"
        )
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

        self.scan_wait_spin = QDoubleSpinBox()
        self.scan_wait_spin.setRange(0.1, 5.0)
        self.scan_wait_spin.setSingleStep(0.1)
        self.scan_wait_spin.setDecimals(1)
        self.scan_wait_spin.setValue(0.5)
        self.scan_settle_spin = QDoubleSpinBox()
        self.scan_settle_spin.setRange(0.3, 8.0)
        self.scan_settle_spin.setSingleStep(0.1)
        self.scan_settle_spin.setDecimals(1)
        self.scan_settle_spin.setValue(1.3)
        self.select_wait_spin = QDoubleSpinBox()
        self.select_wait_spin.setRange(0.1, 5.0)
        self.select_wait_spin.setSingleStep(0.1)
        self.select_wait_spin.setDecimals(1)
        self.select_wait_spin.setValue(0.45)
        self.regrab_wait_spin = QDoubleSpinBox()
        self.regrab_wait_spin.setRange(0.1, 5.0)
        self.regrab_wait_spin.setSingleStep(0.1)
        self.regrab_wait_spin.setDecimals(1)
        self.regrab_wait_spin.setValue(0.6)
        self.focus_clear_pct_spin = QDoubleSpinBox()
        self.focus_clear_pct_spin.setRange(0.0, 50.0)
        self.focus_clear_pct_spin.setSingleStep(1.0)
        self.focus_clear_pct_spin.setDecimals(1)
        self.focus_clear_pct_spin.setValue(4.0)
        self.focus_clear_pct_spin.setToolTip(
            "移焦点用的标题栏点击点 = 扫描按钮中心向右偏移窗口宽度的百分比。\n"
            "默认 4（约 1/25 窗宽）；点击仍落在按钮上/弹窗上时调大（如 8、12）。"
        )

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
        _row("框来源（采集）", self.box_source_combo)
        self.box_source_combo.currentIndexChanged.connect(
            lambda _index: self._sync_box_source(self.box_source_combo)
        )
        _row("OCR 引擎（采集，Photos 路径）", self.ocr_engine_combo)
        _row("读字引擎（嵌字前）", self.stage2_engine_combo)
        _row("OCR 设备（采集，Paddle）", self.paddle_device_combo)
        _row("大拟音过滤", self.drop_large_boxes_check)
        cap_title = QLabel("采集", objectName="cardTitle")
        lay.addWidget(cap_title)
        region_row = _row("采集区域（宽×高，屏 2560×1600 留边界用 2400×1400）", self.region_edit)
        wait_title = QLabel("采集等待（秒，默认值即可，采集异常时再调）", objectName="cardTitle")
        lay.addWidget(wait_title)
        _row("点扫描文本后等待（默认 0.5）", self.scan_wait_spin)
        _row("扫描文本稳定上限（默认 1.3）", self.scan_settle_spin)
        _row("Ctrl+A 后等待（默认 0.45）", self.select_wait_spin)
        _row("选区重抓间隔（默认 0.6）", self.regrab_wait_spin)
        _row("焦点挪开偏移（窗口宽 %，默认 4，点不上标题栏就调大）", self.focus_clear_pct_spin)
        type_title = QLabel("嵌字", objectName="cardTitle")
        lay.addWidget(type_title)
        _row("字号下限（px）", self.min_font_spin)
        _row("拟声词字号上限（px）", self.sfx_font_spin)
        _row("拟声词进术语库门槛（次）", self.sfx_glossary_spin)
        _row("过大单框判定倍数（默认 2.0）", self.sfx_ratio_spin)
        _row("整页字号上限（×常规字号）", self.page_ceiling_spin)
        _row("短原文封顶门槛（0=关）", self.short_source_spin)
        _row("AI OCR 并发（默认 12）", self.ai_ocr_workers_spin)
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
        engine = data.get("ocr_engine")
        index = self.ocr_engine_combo.findData(engine) if engine else -1
        self.ocr_engine_combo.setCurrentIndex(index if index >= 0 else 0)
        box_source = data.get("box_source")
        index = self.box_source_combo.findData(box_source) if box_source else -1
        self.box_source_combo.setCurrentIndex(index if index >= 0 else 0)
        self._sync_box_source(self.box_source_combo)
        self.drop_large_boxes_check.setChecked(
            bool(data.get("drop_large_boxes", True)))
        stage2_engine = data.get("stage2_engine")
        index = self.stage2_engine_combo.findData(stage2_engine) if stage2_engine else -1
        self.stage2_engine_combo.setCurrentIndex(index if index >= 0 else 0)
        device = data.get("paddle_device")
        index = self.paddle_device_combo.findData(device) if device else -1
        self.paddle_device_combo.setCurrentIndex(index if index >= 0 else 0)
        self.min_font_spin.setValue(self._int_setting(data, "min_font_size", 28))
        self.sfx_font_spin.setValue(self._int_setting(data, "sfx_max_font_size", 30))
        self.sfx_glossary_spin.setValue(
            self._int_setting(data, "sfx_glossary_min_count", 2)
        )
        self.ai_ocr_workers_spin.setValue(
            self._int_setting(data, "ai_ocr_workers", 12)
        )
        try:
            self.sfx_ratio_spin.setValue(
                float(data.get("sfx_min_side_ratio", 2.0))
            )
        except (TypeError, ValueError):
            self.sfx_ratio_spin.setValue(2.0)
        self.short_source_spin.setValue(
            self._int_setting(data, "short_source_max_chars", 0)
        )
        try:
            self.page_ceiling_spin.setValue(
                float(data.get("page_font_ceiling_ratio", 1.5))
            )
        except (TypeError, ValueError):
            self.page_ceiling_spin.setValue(1.5)
        font = data.get("font_name", "msyh.ttc")
        index = self.font_combo.findData(font)
        self.font_combo.setCurrentIndex(index if index >= 0 else 0)
        self.stroke_spin.setValue(self._int_setting(data, "stroke_divisor", 10))
        self.row_gap_spin.setValue(self._int_setting(data, "row_gap_divisor", 4))
        self.col_gap_spin.setValue(self._int_setting(data, "column_gap_divisor", 4))
        self.tracking_spin.setValue(self._int_setting(data, "tracking_divisor", 16))
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
        self.box_delta_spin.setValue(self._int_setting(data, "box_delta", 8))
        self.box_tol_spin.setValue(self._int_setting(data, "box_tolerance", 5))
        self.box_area_spin.setValue(self._int_setting(data, "box_area_ratio", 20))
        self.scan_wait_spin.setValue(self._float_setting(data, "scan_wait", 0.5))
        self.scan_settle_spin.setValue(self._float_setting(data, "scan_settle", 1.3))
        self.select_wait_spin.setValue(self._float_setting(data, "select_wait", 0.45))
        self.regrab_wait_spin.setValue(self._float_setting(data, "regrab_wait", 0.6))

    @staticmethod
    def _float_setting(data: dict, key: str, default: float) -> float:
        """Read a float setting without letting a malformed config crash."""
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_setting(data: dict, key: str, default: int) -> int:
        """Read an int setting without letting a malformed config crash the GUI."""
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _color_str(self, key: str, default: str) -> str:
        data = {}
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return str(data.get(key, default))

    def _set_color_btn(self, btn: QPushButton, key: str, default: str) -> None:
        value = self._color_str(key, default)
        if "," not in value:
            btn.setText(value)
            return
        try:
            parts = [int(v) for v in value.split(",")]
            if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
                raise ValueError
        except (TypeError, ValueError):
            value = default
            parts = [int(v) for v in value.split(",")]
        btn.setText(f"{value} #{''.join('%02X' % v for v in parts)}")

    def _pick_color(self, btn: QPushButton, key: str) -> None:
        from PySide6.QtWidgets import QColorDialog

        current = self._color_str(key, "0,0,0" if key == "text_color" else "255,255,255")
        parts = [int(v) for v in current.split(",")] if "," in current else [0, 0, 0]
        if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
            parts = [0, 0, 0]
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
            "box_source": self.box_source_combo.currentData(),
            "drop_large_boxes": self.drop_large_boxes_check.isChecked(),
            "ocr_engine": self.ocr_engine_combo.currentData(),
            "paddle_device": self.paddle_device_combo.currentData(),
            "stage2_engine": self.stage2_engine_combo.currentData(),
            "layout_region": self.region_edit.text().strip() or "2400x1400",
            "min_font_size": self.min_font_spin.value(),
            "sfx_max_font_size": self.sfx_font_spin.value(),
            "sfx_glossary_min_count": self.sfx_glossary_spin.value(),
            "ai_ocr_workers": self.ai_ocr_workers_spin.value(),
            "sfx_min_side_ratio": self.sfx_ratio_spin.value(),
            "short_source_max_chars": self.short_source_spin.value(),
            "page_font_ceiling_ratio": self.page_ceiling_spin.value(),
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
            "scan_wait": self.scan_wait_spin.value(),
            "scan_settle": self.scan_settle_spin.value(),
            "select_wait": self.select_wait_spin.value(),
            "regrab_wait": self.regrab_wait_spin.value(),
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
        card, lay = self._card("审核（失败/无蓝像素/无文本页，默认全部进审核）")
        split = QHBoxLayout()
        split.setSpacing(16)

        left = QVBoxLayout()
        left.addWidget(label("待审核页（多选）"))
        self.review_list = QListWidget()
        self.review_list.setSelectionMode(QListWidget.ExtendedSelection)
        left.addWidget(self.review_list, 1)
        split.addLayout(left, 2)

        right = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(label("预览帧"))
        self.review_frame_combo = QComboBox()
        self.review_frame_combo.addItem("扫描文本帧（Photos，有无文字）", "photos_ocr")
        self.review_frame_combo.addItem("全选帧（Photos，有无蓝框）", "photos_selected")
        self.review_frame_combo.addItem(
            "检出框 check 图（Paddle 红框 / Photos 蓝框）", "check")
        top.addWidget(self.review_frame_combo)
        top.addStretch(1)
        right.addLayout(top)
        self.review_view = QGraphicsView()
        self.review_view.setMinimumSize(520, 420)
        self.review_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.review_scene = QGraphicsScene(self)
        self.review_view.setScene(self.review_scene)
        self.review_pix = QGraphicsPixmapItem()
        self.review_scene.addItem(self.review_pix)
        self.review_view.wheelEvent = self._review_zoom
        right.addWidget(self.review_view, 1)
        split.addLayout(right, 3)
        lay.addLayout(split, 1)

        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_review)
        confirm = QPushButton("确认无框（跳过）")
        confirm.setProperty("primary", True)
        confirm.clicked.connect(self._confirm_review)
        manga = QPushButton("改用 manga 保底")
        manga.clicked.connect(self._review_manga_fallback)
        reidentify = QPushButton("重新识别（Paddle）")
        reidentify.setProperty("danger", True)
        reidentify.setToolTip(
            "对所选页用 Paddle 检测重新识别框（离线、不开 Photos 窗口、快）。"
            "适用于：筛框后无框、检测框不准、或从 Photos 路径切过来。"
        )
        reidentify.clicked.connect(self._review_reidentify)
        recapture = QPushButton("重采所选（Photos）")
        recapture.setProperty("danger", True)
        recapture.setToolTip("对所选页开 Photos 窗口重新采集（原流程，慢但框紧贴文字）")
        recapture.clicked.connect(self._review_recapture)
        recapture_all = QPushButton("全部重采")
        recapture_all.setProperty("danger", True)
        recapture_all.setToolTip(
            "清除所有待审核页的 capture 目录并按当前框来源重新采集"
        )
        recapture_all.clicked.connect(self._review_recapture_all)
        h = QHBoxLayout()
        for b in (refresh, confirm, manga, reidentify, recapture, recapture_all):
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.review_list.currentRowChanged.connect(self._review_show_preview)
        self.review_frame_combo.currentIndexChanged.connect(self._review_show_preview)
        return card

    def _review_zoom(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.review_view.scale(factor, factor)
        event.accept()

    def _review_selected_pages(self) -> list[str]:
        pages = []
        for item in self.review_list.selectedItems():
            page = item.data(Qt.UserRole)
            if page:
                pages.append(page)
        return pages

    def _review_current_preview_path(self, page: str) -> str | None:
        """审核页预览图路径：选中帧 → photos_base → check 图 → 原始源图。

        只找 ``photos_*`` 会让 **Paddle 采集的页预览全空**（它的 capture/ 下
        只有 capture.json），所以 check 图（Paddle 红框 / Photos 蓝框）是必经
        一环——审核要看的正是"当时检出了什么"。
        """
        images = self.images_edit.text().strip()
        if not images:
            return None
        out = ROOT / "outputs" / "oneclick" / Path(images).name
        page_dir = None
        manifest = None
        for source, _label in _BOX_SOURCES:
            candidate = out / "capture" / source / page
            if (candidate / "capture.json").is_file():
                page_dir, manifest = candidate, candidate / "capture.json"
                break
        frame = self.review_frame_combo.currentData()
        if frame == "check":
            for source, _label in _BOX_SOURCES:
                candidate = out / "check" / source / f"{page}.bluebox_check.png"
                if candidate.is_file():
                    return str(candidate)
            return None
        if page_dir is not None:
            path = page_dir / f"{frame}.png"
            if path.is_file():
                return str(path)
        for source, _label in _BOX_SOURCES:
            candidate = out / "check" / source / f"{page}.bluebox_check.png"
            if candidate.is_file():
                return str(candidate)
        if page_dir is not None and (page_dir / "photos_base.png").is_file():
            return str(page_dir / "photos_base.png")
        if manifest is not None:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
                src = data.get("source", {}).get("path")
                if src and Path(src).is_file():
                    return str(src)
            except (OSError, ValueError):
                pass
        return None

    def _review_show_preview(self) -> None:
        # Always preview the CURRENT row, not the first selected item:
        # ``currentRowChanged`` fires before ``itemSelectionChanged`` updates
        # the selection set, so ``selectedItems()[0]`` can return the
        # previously-clicked page and show the wrong frame.  ``currentItem``
        # is updated atomically with the row change.
        item = self.review_list.currentItem()
        if item is None or not self.review_list.count():
            return
        page = item.data(Qt.UserRole)
        if not page:
            return
        images = self.images_edit.text().strip()
        if not images:
            return
        path = self._review_current_preview_path(page)
        if path is None:
            self.review_scene.clear()
            self.review_pix = QGraphicsPixmapItem()
            self.review_scene.addItem(self.review_pix)
            return
        path = Path(path)
        if frame == "check":
            # 显式选"检出框"时直接给 check 图（Paddle 页只有它）
            path = out / "check" / f"{page}.bluebox_check.png"
        if not path.is_file():
            self.review_scene.clear()
            self.review_pix = QGraphicsPixmapItem()
            self.review_scene.addItem(self.review_pix)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self.review_scene.clear()
        self.review_pix = QGraphicsPixmapItem(pixmap)
        self.review_scene.addItem(self.review_pix)
        self.review_scene.setSceneRect(self.review_pix.boundingRect())
        self.review_view.fitInView(self.review_pix, Qt.KeepAspectRatio)

    def _review_manga_fallback(self) -> None:
        images = self._images()
        if not images:
            return
        pages = self._review_selected_pages()
        if not pages:
            QMessageBox.information(self, "提示", "请先选择要改用的页面")
            return
        out = ROOT / "outputs" / "oneclick" / Path(images).name / "capture"
        for page in pages:
            manifest = out / page / "capture.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            cap = data.setdefault("capture", {})
            cap["fallback_manga"] = True
            cap["selection_review_required"] = False
            manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"已改用 manga 保底：{', '.join(pages)}")
        self._refresh_review()

    def _review_reidentify(self) -> None:
        """重新识别（Paddle 检测路径）：对所选页重跑框识别。

        与"重采"的区别是**不开 Photos 窗口**：走 PaddleOCR 全页检测，离线、
        快速、无焦点/时序问题。审核页里"筛框后无框"的页多半是框被过滤光了，
        走这条最快。
        """
        images = self._images()
        if not images:
            return
        pages = self._review_selected_pages()
        if not pages:
            QMessageBox.information(self, "提示", "请先选择要重新识别的页面")
            return
        self._recapture_pages(pages, box_source="paddle")

    def _review_recapture(self) -> None:
        images = self._images()
        if not images:
            return
        pages = self._review_selected_pages()
        if not pages:
            QMessageBox.information(self, "提示", "请先选择要重采的页面")
            return
        self._recapture_pages(pages, box_source="photos")

    def _review_recapture_all(self) -> None:
        """Re-capture every page currently in the review queue.

        Equivalent to clicking "重采所选" for every pending page in one
        shot: the single-path detector (strict formula only) means that
        pages with no blue pixels after three Ctrl+A attempts are now
        expected to be the majority of the review queue, so a bulk
        re-capture is the most efficient way to clear it.
        """
        images = self._images()
        if not images:
            return
        pages = [name for name, _ in self._review_pages()]
        if not pages:
            QMessageBox.information(self, "提示", "当前没有待审核页")
            return
        self._recapture_pages(pages)

    def _recapture_pages(self, pages: list[str], *, box_source: str = "") -> None:
        """Shared recapture pipeline: confirm, clear capture dirs, start stage1.

        ``box_source`` 强制指定框来源（"paddle" = 检测重识别，"photos" = 开窗口
        重采）；留空则用界面当前设置。
        """
        if not pages:
            return
        label = {"paddle": "Paddle 重新识别", "photos": "Photos 窗口重采"}.get(
            box_source, "重新采集"
        )
        answer = QMessageBox.question(
            self, "重采确认",
            f"删除并{label}这 {len(pages)} 页？\n{', '.join(pages)}"
        )
        if answer != QMessageBox.Yes:
            return
        images = self._images()
        # Close orphan Photos windows first: a stale window can hold the
        # page's files open (so rmtree fails silently) or steal the
        # capture focus; then stage1 runs with --overwrite, which clears
        # the old artifacts itself (the GUI delete below is best-effort).
        try:
            import os as _os

            from photos_capture import _close_residual_photos_windows

            names = {
                name for name in _os.listdir(images)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            }
            _close_residual_photos_windows(names)
        except Exception:
            pass
        out = ROOT / "outputs" / "oneclick" / Path(images) / "capture"
        import shutil

        for page in pages:
          for _source, _label in _BOX_SOURCES:
            page_dir = out / _source / page
            if page_dir.is_dir():
                try:
                    shutil.rmtree(page_dir)
                except OSError as error:
                    # The --overwrite flag on the stage1 command below
                    # purges the same artifacts, so a locked file here
                    # must not abort the recapture.
                    self._log(f"警告: 删除 {page_dir} 失败（{error}），由 --overwrite 兜底")
        self._log(f"已清除 {len(pages)} 页数据，开始{label}…")
        args = [*self._base_args(), "--pages", ",".join(pages), "--overwrite"]
        if box_source:
            # 显式覆盖界面设置：审核页两条路各管一种来源，互不干扰
            args += ["--box-source", box_source]
        self._run(args)

    # ---- tab switching ----
    def _show_tab(self, index: int) -> None:
        for w, i in (
            (self.tab_capture, 0),
            (self.tab_review, 1),
            (self.tab_process, 2),
            (self.tab_preview, 3),
            (self.tab_glossary, 4),
            (self.tab_settings, 5),
        ):
            w.setVisible(i == index)
        if index == 1:
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
        ratio = done / total if total else 0
        running = self.runner is not None and self.runner.isRunning()
        if not finished and not running and elapsed > 0:
            # No active task but run.json says unfinished: the previous run
            # was interrupted (closed window / killed process).
            self.stage_label.setText(f"阶段：{stage}（上次运行中断）")
            self.item_label.setText("当前：—")
            self.progress_bar.setFormat(f"{done}/{total} · {ratio * 100:.0f}%")
            self.eta_label.setText(f"上次运行未正常结束（耗时 {int(elapsed)}s）")
            return
        self.stage_label.setText(f"阶段：{stage}（{status.get('stage_index', 0) + 1}/{status.get('total_stages', 6)}）")
        self.item_label.setText(f"当前：{item or '—'}")
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

    def _browse_process(self) -> None:
        """Browse for the source directory from the process tab.  The two
        directory rows are kept in sync, so writing here also updates the
        capture tab's field."""
        path = QFileDialog.getExistingDirectory(self, "选择原图目录")
        if path:
            self.process_images_edit.setText(path)

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
        box_source = self.box_source_combo.currentData()
        if box_source and box_source != "paddle":
            args += ["--box-source", box_source]
        if not self.drop_large_boxes_check.isChecked():
            args.append("--keep-large-boxes")
        engine = self.ocr_engine_combo.currentData()
        if engine and engine != "auto":
            args += ["--ocr-engine", engine]
        device = self.paddle_device_combo.currentData()
        if device and device != "auto":
            args += ["--paddle-device", device]
        stage2_engine = self.stage2_engine_combo.currentData()
        if stage2_engine and stage2_engine != "manga-ocr":
            args += ["--stage2-engine", stage2_engine]
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
        args += ["--sfx-max-font-size", str(self.sfx_font_spin.value())]
        args += ["--sfx-min-side-ratio", f"{self.sfx_ratio_spin.value():.2f}"]
        args += ["--short-source-max-chars", str(self.short_source_spin.value())]
        args += [
            "--page-font-ceiling-ratio", f"{self.page_ceiling_spin.value():.2f}",
        ]
        args += [
            "--sfx-glossary-min-count", str(self.sfx_glossary_spin.value()),
        ]
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
        args += ["--ai-ocr-workers", str(self.ai_ocr_workers_spin.value())]
        return args

    def _start_capture(self) -> None:
        if not self._images():
            return
        args = [*self._base_args(), *self._detection_args()]
        spacing = self.row_spacing_spin.value()
        if spacing > 0:
            args += ["--layout-row-spacing", str(spacing)]
        right_gap = self.right_gap_spin.value()
        if right_gap > 0:
            args += ["--layout-right-gap", str(right_gap)]
        args += ["--scan-wait", str(self.scan_wait_spin.value())]
        args += ["--scan-settle", str(self.scan_settle_spin.value())]
        args += ["--select-wait", str(self.select_wait_spin.value())]
        args += ["--regrab-wait", str(self.regrab_wait_spin.value())]
        args += ["--focus-clear-pct", str(self.focus_clear_pct_spin.value())]
        if self.capture_overwrite_check.isChecked():
            args.append("--overwrite")
        self._run(args)

    def _start_process(self) -> None:
        if not self._images():
            return
        args = [*self._base_args(), *self._ai_args(), *self._detection_args()]
        # The process tab has its own page range (independent of the
        # capture tab's): fill it to translate/typeset only those pages.
        process_pages = self.process_pages_edit.text().strip()
        if process_pages:
            # Strip any capture-tab "--pages <value>" pair from _base_args,
            # then append the process-tab range.  Appending keeps the
            # --images value intact (inserting at index 1 put --pages
            # BEFORE the image path, which made --images swallow it).
            cleaned = []
            skip_next = False
            for item in args:
                if skip_next:
                    skip_next = False
                    continue
                if item == "--pages":
                    skip_next = True
                    continue
                cleaned.append(item)
            args = cleaned
            args += ["--pages", process_pages]
        stage = self.stage_combo.currentData()
        if stage == "all":
            args.append("--process")
        else:
            args += ["--stage", stage]
        if self.overwrite_check.isChecked():
            args.append("--overwrite")
        elif self.resume_check.isChecked():
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
        self.stop_btn.setEnabled(True)
        self.process_stop_btn.setEnabled(True)
        self._log("$ python one_click_translate.py " + " ".join(args))
        self.runner = Runner(args)
        self.runner.line.connect(self._log)
        self.runner.finished_ok.connect(self._on_done)
        # A cancel outside closeEvent (e.g. the "stop" button) must still
        # re-enable the action buttons; route it through the same done
        # handler with ok=False so the UI never gets stuck.
        self.runner.cancelled.connect(lambda: self._on_done(False))
        self.runner.start()

    def _stop_task(self) -> None:
        """Stop the running capture/process task (one-click kill switch).

        Terminates the whole child process tree (one_click_translate.py
        -> photos_capture.py -> Photos windows) and sweeps orphan Photos
        windows, so no window is left hanging on the desktop.  Safe to
        click at any time while a task is running.
        """
        if self.runner is None or not self.runner.isRunning():
            return
        answer = QMessageBox.question(
            self, "停止任务",
            "停止当前任务？\n将终止处理进程并关闭其打开的 Photos 窗口。",
        )
        if answer != QMessageBox.Yes:
            return
        self._log("== 正在停止任务… ==")
        self.runner.cancel()
        # Runner.cancel() emits ``cancelled`` when the process ends,
        # which routes to _on_done(False) and re-enables the buttons.
        # If the process tree is already gone the signal may not fire;
        # restore the UI right away so the buttons never stay locked.
        self._on_done(False)

    def _on_done(self, ok: bool) -> None:
        self.capture_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process_stop_btn.setEnabled(False)
        self._log("== 任务结束（%s）==" % ("成功" if ok else "失败"))
        self._refresh_review()

    # ---- review ----
    def _review_pages(self) -> list[tuple[str, str]]:
        """Review-gated pages with their failure reason."""
        images = self.images_edit.text().strip()
        if not images:
            return []
        out = ROOT / "outputs" / "oneclick" / Path(images).name / "capture"
        if not out.is_dir():
            return []
        review = []

        def _page_key(path: Path) -> tuple[int, str]:
            name = path.name
            return (0, name) if name.isdigit() else (1, name.lower())

        entries = []
        for source, _label in _BOX_SOURCES:
            base = out / source
            if not base.is_dir():
                continue
            for page in sorted((p for p in base.iterdir() if p.is_dir()), key=_page_key):
                entries.append((page.name, page))
        for page_name, page in entries:
            manifest = page / "capture.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            cap = data.get("capture", {})
            if cap.get("selection_review_required") and not cap.get("selection_review_confirmed"):
                if cap.get("selection_empty_after_filter"):
                    reason = "筛框后无框"
                elif cap.get("no_text"):
                    reason = "无文本"
                elif cap.get("selection_missing"):
                    reason = "无蓝色像素"
                else:
                    reason = "采集失败"
                review.append((page_name, reason))
        return review

    def _refresh_review(self) -> None:
        self.review_list.clear()
        for page, reason in self._review_pages():
            item = QListWidgetItem(f"{page}  [{reason}]")
            item.setData(Qt.UserRole, page)
            self.review_list.addItem(item)
        if self.review_list.count():
            self.review_list.setCurrentRow(0)
            self._review_show_preview()
        else:
            self.review_scene.clear()
            self.review_pix = QGraphicsPixmapItem()
            self.review_scene.addItem(self.review_pix)

    def _confirm_review(self) -> None:
        images = self._images()
        if not images:
            return
        pages = self._review_selected_pages()
        if not pages:
            pages = [p for p, _ in self._review_pages()]
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
