"""翻译调整工作台页 - 可嵌入统一界面 / 独立运行双模式

流程: 选图 → 翻译并导出状态 → 图形化调整文本框 → 重新嵌字 → 合并PDF
编辑器交互复用 adjust_gui 的 View/TextBox；重嵌通过子进程调 re_render.py
（避免 GUI 进程 import MITS 依赖链，兼容 exe 打包）。
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

from PySide6.QtCore import Qt, QRectF, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QPlainTextEdit, QCheckBox,
    QProgressBar, QComboBox, QFileDialog, QMessageBox, QGraphicsPixmapItem,
    QGraphicsScene, QFrame, QColorDialog, QDoubleSpinBox,
)

from PIL import Image  # 类型注解用（渲染逻辑见 text_paint）

from paths import PYTHON, TRANSLATE, RERENDER
from service_manager import SakuraManager
from adjust_gui import View, TextBox  # noqa: E402
from gui import make_card
from glossary import Glossary
from text_segments import align_translation_segments

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_images(folder: str):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")) and n.lower() != ".thumb":
                files.append(os.path.join(root, n))
    return sorted(files, key=lambda f: natural_key(os.path.basename(f)))


def _bounds_for_lines(lines):
    points = [point for point in (lines or []) if len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _scale_points(points, old_bounds, new_bounds):
    if not points or old_bounds is None:
        return []
    old_x1, old_y1, old_x2, old_y2 = old_bounds
    new_x1, new_y1, new_x2, new_y2 = new_bounds
    old_w = max(old_x2 - old_x1, 1.0)
    old_h = max(old_y2 - old_y1, 1.0)
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    return [
        [
            new_x1 + (float(point[0]) - old_x1) * new_w / old_w,
            new_y1 + (float(point[1]) - old_y1) * new_h / old_h,
        ]
        for point in points
    ]


def render_text_preview(text: str, font_size: int, vertical: bool,
                        max_w: int, max_h: int, fg=None, bg=None, rtl: bool = True,
                        **kw) -> Image.Image:
    """渲染译文预览图（与重新嵌字共用 text_paint，所见即所得）"""
    from text_paint import render_text_image
    return render_text_image(text, font_size, vertical, max_w, max_h, fg or (0, 0, 0), bg,
                             rtl=rtl, **kw)


class TranslateWorker(QThread):
    """逐页翻译并导出调整状态"""
    progress = Signal(int, int, str)   # done, total, filename
    log = Signal(str)
    finished_ok = Signal(str)          # state_dir

    def __init__(self, files, state_dir):
        super().__init__()
        self.files = files
        self.state_dir = state_dir
        self._tmp_out = tempfile.mkdtemp(prefix="wb_out_")
        self._last_error = ""

    def run(self):
        total = len(self.files)
        ok = 0
        for i, f in enumerate(self.files, 1):
            base = os.path.splitext(os.path.basename(f))[0]
            self.progress.emit(i, total, base)
            out = os.path.join(self._tmp_out, f"o{i}.png")
            try:
                r = subprocess.run(
                    [PYTHON, TRANSLATE, f, out, "--no-popup", "--overwrite",
                     "--manual-dir", self.state_dir],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    creationflags=CREATE_NO_WINDOW, timeout=600)
                if r.returncode == 0:
                    ok += 1
                    if os.path.exists(os.path.join(self.state_dir, base + ".json")):
                        self.log.emit(f"[{i}/{total}] {base} OK")
                    else:
                        self.log.emit(f"[{i}/{total}] {base} 完成(无文本，未生成状态)")
                else:
                    err = (r.stdout or r.stderr or "").strip()
                    tail = (err.splitlines()[-1] if err else "未知错误")[-300:]
                    self.log.emit(f"[{i}/{total}] {base} 失败: {tail}")
                    self._last_error = tail
            except Exception as e:
                self.log.emit(f"[{i}/{total}] {base} 异常: {e}")
        self.log.emit(f"翻译完成: {ok}/{total} 页导出状态")
        self.finished_ok.emit(self.state_dir)


class RerenderWorker(QThread):
    """逐页重新嵌字（子进程调 re_render.py，GUI 进程不 import MITS）"""
    progress = Signal(int, int, str)
    log = Signal(str)
    done = Signal(str, str)  # out_dir, err

    def __init__(self, jsons, out_dir, fmt="auto", direction="auto"):
        super().__init__()
        self.jsons = jsons
        self.out_dir = out_dir
        self.fmt = fmt
        self.direction = direction

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        errs = []
        for i, j in enumerate(self.jsons, 1):
            name = os.path.splitext(os.path.basename(j))[0]
            self.progress.emit(i, len(self.jsons), name)
            try:
                r = subprocess.run(
                    [PYTHON, RERENDER, j, os.path.join(self.out_dir, name + ".png"),
                     "--format", self.fmt, "--direction", self.direction],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    creationflags=CREATE_NO_WINDOW, timeout=300)
                if r.returncode != 0:
                    tail = (r.stdout or r.stderr or "").strip().splitlines()
                    errs.append(f"{name}: {(tail[-1] if tail else '重嵌失败')[-200:]}")
            except Exception as e:
                errs.append(f"{name}: {e}")
        self.done.emit(self.out_dir, "\n".join(errs))


class WorkbenchPage(QWidget):
    """调整工作台页面（卡片式布局，编辑器逻辑保持原样）"""

    def __init__(self, service: SakuraManager = None, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.jsons = []
        self.idx = 0
        self._boxes = []
        self._selected_idx = -1
        self._orig_overlay = None
        self.state_dir = ""
        self.out_dir = ""
        self._undo_stack = {}        # page路径 -> 快照栈
        self._panel_dirty = False    # 面板修改入栈节流
        self._last_geo_snap = 0.0    # 拖动入栈节流
        self.service = service if service is not None else SakuraManager(self)
        self.service.status_changed.connect(self._on_service_status)
        self._init_ui()
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence.Undo, self, activated=self._undo)
        self._on_service_status(self.service.is_running() and "running" or "stopped")

    # ---------------- UI ----------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        # 标题行
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("调整工作台")
        title.setObjectName("title")
        sub = QLabel("翻译 → 图形化调整文本框 → 重新嵌字 → 合并 PDF")
        sub.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        head.addLayout(title_box)
        head.addStretch(1)
        self.svc_badge = QLabel()
        self.svc_badge.setObjectName("badge")
        head.addWidget(self.svc_badge)
        self.svc_label = QLabel("")
        head.addWidget(self.svc_label)
        root.addLayout(head)

        # 行1: 输入 / 状态 / 输出（两行紧凑布局，给编辑器让出空间）
        c1, v1 = make_card()
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("输入:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("图片(可多选) / 文件夹 / zip / cbz")
        b1 = QPushButton("浏览…")
        b1.clicked.connect(self._pick_input)
        self.btn_translate = QPushButton("翻译并导出调整状态")
        self.btn_translate.setToolTip("对输入图片自动翻译+擦字，并导出可调整的文本框状态（需要 sakura 服务运行中）")
        self.btn_translate.clicked.connect(self._start_translate)
        r1.addWidget(self.input_edit, 3)
        r1.addWidget(b1)
        r1.addWidget(self.btn_translate)
        v1.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("状态:"))
        self.state_edit = QLineEdit()
        self.state_edit.setPlaceholderText("调整状态目录（翻译后自动填充，也可手动选）")
        b2 = QPushButton("浏览…")
        b2.clicked.connect(self._pick_state)
        self.btn_load = QPushButton("载入状态")
        self.btn_load.clicked.connect(self._load_states)
        r2.addWidget(self.state_edit, 3)
        r2.addWidget(b2)
        r2.addWidget(self.btn_load)
        r2.addSpacing(10)
        r2.addWidget(QLabel("输出:"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("重新嵌字输出目录（默认: 调整状态\\输出）")
        b3 = QPushButton("浏览…")
        b3.clicked.connect(self._pick_out)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItem("与原图相同", "auto")
        self.fmt_combo.addItem("png", "png")
        self.fmt_combo.addItem("jpg", "jpg")
        self.fmt_combo.addItem("webp", "webp")
        r2.addWidget(self.out_edit, 3)
        r2.addWidget(b3)
        r2.addWidget(QLabel("格式:"))
        r2.addWidget(self.fmt_combo)
        v1.addLayout(r2)
        self.card_settings = c1
        root.addWidget(c1)

        # 行3: 翻页导航
        c3, v3 = make_card()
        r3 = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一页")
        self.btn_prev.clicked.connect(lambda: self._goto(self.idx - 1))
        self.page_label = QLabel("未载入状态")
        self.page_label.setObjectName("muted")
        self.btn_next = QPushButton("下一页 ▶")
        self.btn_next.clicked.connect(lambda: self._goto(self.idx + 1))
        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self._fit_view)
        r3.addWidget(self.btn_prev)
        r3.addWidget(self.page_label, 1)
        r3.addWidget(self.btn_next)
        r3.addWidget(self.btn_fit)
        r3.addWidget(QLabel("字号:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 120)
        self.font_spin.valueChanged.connect(self._on_font_change)
        r3.addWidget(self.font_spin)
        v3.addLayout(r3)

        # 行3c: 操作（拆行，按钮不再被压缩）
        r3c = QHBoxLayout()
        self.btn_save = QPushButton("保存本页")
        self.btn_save.clicked.connect(self._save_page)
        self.btn_rerender = QPushButton("重新嵌字")
        self.btn_rerender.setObjectName("primary")
        self.btn_rerender.clicked.connect(self._rerender_all)
        self.rerange_combo = QComboBox()
        self.rerange_combo.addItem("全部页面", "all")
        self.rerange_combo.addItem("仅当前页", "page")
        self.rerange_combo.setToolTip("重新嵌字的范围：全部页面或仅当前页")
        self.btn_pdf = QPushButton("合并PDF")
        self.btn_pdf.clicked.connect(self._merge_pdf)
        self.btn_open_out = QPushButton("打开输出")
        self.btn_open_out.clicked.connect(self._open_out)
        self.btn_add_box = QPushButton("＋ 添加文本框")
        self.btn_add_box.setToolTip("添加一个空文本框，拖到目标位置并填写译文（用于补检漏的文字）")
        self.btn_add_box.clicked.connect(self._add_box)
        self.btn_del_box = QPushButton("－ 删除文本框")
        self.btn_del_box.setToolTip("删除当前选中的文本框（或按 Delete 键）")
        self.btn_del_box.clicked.connect(self._delete_box)
        self.btn_box_bg = QPushButton("文本颜色")
        self.btn_box_bg.setToolTip("修改选中文本框的文字颜色（默认黑字，可改其他颜色）")
        self.btn_box_bg.clicked.connect(self._pick_box_fg)
        self.btn_focus = QPushButton("全屏编辑")
        self.btn_focus.setToolTip("隐藏输入/输出设置与日志，图片操作区最大化；再次点击恢复")
        self.btn_focus.clicked.connect(self._toggle_focus)
        r3c.addWidget(self.btn_save)
        r3c.addWidget(self.btn_rerender)
        r3c.addWidget(self.rerange_combo)
        r3c.addWidget(self.btn_pdf)
        r3c.addWidget(self.btn_open_out)
        r3c.addWidget(self.btn_add_box)
        r3c.addWidget(self.btn_del_box)
        r3c.addWidget(self.btn_box_bg)
        r3c.addStretch(1)
        r3c.addWidget(self.btn_focus)
        v3.addLayout(r3c)
        root.addWidget(c3)

        # 行3b: 原图参照 / 译文编辑 / 角度 / 方向
        r3b = QHBoxLayout()
        self.orig_check = QCheckBox("显示原文参照")
        self.orig_check.setChecked(True)
        self.orig_check.toggled.connect(self._on_orig_toggle)
        r3b.addWidget(self.orig_check)
        self.fit_check = QCheckBox("拖框自动适配字号")
        self.fit_check.setChecked(True)
        self.fit_check.setToolTip("拖动/缩放文本框时按新框尺寸自动调整字号；取消勾选后只改框不改变字号")
        r3b.addWidget(self.fit_check)
        r3b.addWidget(QLabel("方向:"))
        self.dir_combo = QComboBox()
        self.dir_combo.addItem("跟随原文(竖排右→左)", "auto")
        self.dir_combo.addItem("从左到右", "ltr")
        self.dir_combo.addItem("从右到左", "rtl")
        self.dir_combo.currentIndexChanged.connect(self._on_dir_change)
        r3b.addWidget(self.dir_combo)
        r3b.addWidget(QLabel("译文:"))
        self.trans_edit = QLineEdit()
        self.trans_edit.setPlaceholderText("选中文本框后可直接修改译文")
        self.trans_edit.textEdited.connect(self._on_trans_edit)
        r3b.addWidget(self.trans_edit, 1)
        self.btn_remember = QPushButton("记住此译法")
        self.btn_remember.setToolTip("把当前框的 原文=>译文 保存到全局术语表（译名一致性）")
        self.btn_remember.clicked.connect(self._remember_translation)
        r3b.addWidget(self.btn_remember)
        r3b.addWidget(QLabel("译文:"))
        self.trans_edit = QLineEdit()
        self.trans_edit.setPlaceholderText("选中文本框后可直接修改译文")
        self.trans_edit.textEdited.connect(self._on_trans_edit)
        r3b.addWidget(self.trans_edit, 1)
        self.btn_remember = QPushButton("记住此译法")
        self.btn_remember.setToolTip("把当前框的 原文=>译文 保存到全局术语表（译名一致性）")
        self.btn_remember.clicked.connect(self._remember_translation)
        r3b.addWidget(self.btn_remember)
        v3.addLayout(r3b)
        root.addWidget(c3)

        # 编辑器 + 字符面板（PS 风格）
        edit_row = QHBoxLayout()
        self.scene = QGraphicsScene()
        self.view = View()
        self.view.setScene(self.scene)
        self.view.box_selected.connect(self._on_box_selected)
        self.view.box_delete_requested.connect(self._delete_box)
        edit_row.addWidget(self.view, 1)
        edit_row.addWidget(self._build_style_panel())
        root.addLayout(edit_row, 1)

        # 日志/信息
        log_card, log_v = make_card("进度 / 日志")
        self.card_log = log_card
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(72)
        log_v.addWidget(self.info)
        root.addWidget(log_card)
        self._hint("提示: ①选择输入→「翻译并导出调整状态」(需 sakura 服务) 或直接载入 gui 导出的状态  ②蓝色框拖动移动、角点缩放、可改字号/角度/译文  ③重新嵌字(全部或当前页)→合并PDF")

        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("就绪")
        head.addWidget(self.progress_label)
        head.addWidget(self.progress_bar)

    def _hint(self, text):
        self.info.appendPlainText(text)

    # ---------------- 字符面板（PS 风格） ----------------

    def _build_style_panel(self):
        from fonts import list_fonts, import_font
        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(280)
        v = QVBoxLayout(panel)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        t = QLabel("文字样式")
        t.setObjectName("h2")
        v.addWidget(t)

        # 字体
        row = QHBoxLayout()
        self.font_combo = QComboBox()
        for f in list_fonts():
            self.font_combo.addItem(f["display"], f["name"])
        self.font_combo.currentIndexChanged.connect(
            lambda _: self._panel_set("font", self.font_combo.currentData()))
        self.bold_check = QCheckBox("粗体")
        self.bold_check.toggled.connect(lambda c: self._panel_set("font_bold", bool(c)))
        row.addWidget(self.font_combo, 1)
        row.addWidget(self.bold_check)
        v.addLayout(row)

        # 颜色: 文字色 / 描边色 / 描边宽
        row = QHBoxLayout()
        self.btn_fg_color = self._color_button("文字色")
        self.btn_fg_color.clicked.connect(lambda: self._pick_color("fg", "选择文字颜色"))
        self.btn_stroke_color = self._color_button("描边色")
        self.btn_stroke_color.clicked.connect(lambda: self._pick_color("stroke", "选择描边颜色"))
        row.addWidget(self.btn_fg_color)
        row.addWidget(self.btn_stroke_color)
        row.addWidget(QLabel("描边宽:"))
        self.stroke_spin = QSpinBox()
        self.stroke_spin.setRange(0, 30)
        self.stroke_spin.valueChanged.connect(lambda v: self._panel_set("stroke_w", int(v) or None))
        row.addWidget(self.stroke_spin, 1)
        v.addLayout(row)

        # 实心底
        row = QHBoxLayout()
        self.fill_check = QCheckBox("实心底色")
        self.fill_check.toggled.connect(lambda c: self._panel_set("fill_bg", bool(c)))
        self.btn_fill_color = self._color_button("底色")
        self.btn_fill_color.clicked.connect(lambda: self._pick_color("fill", "选择实心底色"))
        row.addWidget(self.fill_check)
        row.addWidget(self.btn_fill_color)
        row.addStretch(1)
        v.addLayout(row)

        # 段落: 行距 / 字距
        row = QHBoxLayout()
        row.addWidget(QLabel("行距:"))
        self.line_spin = QDoubleSpinBox()
        self.line_spin.setRange(0.8, 2.0)
        self.line_spin.setSingleStep(0.05)
        self.line_spin.setSuffix("x")
        self.line_spin.valueChanged.connect(
            lambda val: self._panel_set("line_spacing", round(float(val), 2)))
        row.addWidget(self.line_spin)
        row.addWidget(QLabel("字距:"))
        self.char_spin = QSpinBox()
        self.char_spin.setRange(-5, 30)
        self.char_spin.valueChanged.connect(lambda val: self._panel_set("char_spacing", int(val)))
        row.addWidget(self.char_spin, 1)
        v.addLayout(row)

        # 对齐
        row = QHBoxLayout()
        row.addWidget(QLabel("横排:"))
        self.align_h_combo = QComboBox()
        self.align_h_combo.addItem("左", "left")
        self.align_h_combo.addItem("居中", "center")
        self.align_h_combo.addItem("右", "right")
        self.align_h_combo.currentIndexChanged.connect(
            lambda _: self._panel_set("align_h", self.align_h_combo.currentData()))
        row.addWidget(self.align_h_combo)
        row.addWidget(QLabel("竖排:"))
        self.align_v_combo = QComboBox()
        self.align_v_combo.addItem("上", "top")
        self.align_v_combo.addItem("居中", "center")
        self.align_v_combo.addItem("下", "bottom")
        self.align_v_combo.currentIndexChanged.connect(
            lambda _: self._panel_set("align_v", self.align_v_combo.currentData()))
        row.addWidget(self.align_v_combo, 1)
        v.addLayout(row)

        # 方向 / 角度
        row = QHBoxLayout()
        row.addWidget(QLabel("方向:"))
        self.force_combo = QComboBox()
        self.force_combo.addItem("跟随框", "auto")
        self.force_combo.addItem("强制横排", "h")
        self.force_combo.addItem("强制竖排", "v")
        self.force_combo.currentIndexChanged.connect(
            lambda _: self._panel_set("force_dir", self.force_combo.currentData()))
        row.addWidget(self.force_combo)
        row.addWidget(QLabel("角度:"))
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(-45, 45)
        self.angle_spin.valueChanged.connect(self._on_angle_change)
        row.addWidget(self.angle_spin, 1)
        v.addLayout(row)

        # 导入字体
        btn_import = QPushButton("导入字体…")
        btn_import.clicked.connect(self._import_font)
        v.addWidget(btn_import)

        v.addStretch(1)
        self.style_panel = panel
        self._style_widgets = [
            self.font_combo, self.bold_check, self.btn_fg_color, self.btn_stroke_color,
            self.stroke_spin, self.fill_check, self.btn_fill_color, self.line_spin,
            self.char_spin, self.align_h_combo, self.align_v_combo, self.force_combo,
            self.angle_spin,
        ]
        for w in self._style_widgets:
            w.setEnabled(False)
        return panel

    def _color_button(self, text: str) -> QPushButton:
        b = QPushButton(text)
        b.setMinimumWidth(72)
        return b

    def _set_color_style(self, btn: QPushButton, rgb):
        btn.setStyleSheet(
            f"QPushButton {{ background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
            f"border: 1px solid #94a3b8; border-radius: 8px; color: "
            f"{'#111827' if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]) > 140 else '#ffffff'}; }}")

    def _pick_color(self, kind: str, title: str):
        idx = self._selected_idx
        if not (0 <= idx < len(self._boxes)):
            return
        box = self._boxes[idx]
        if kind == "fg":
            cur = box.fg or [0, 0, 0]
        elif kind == "stroke":
            cur = box.bg or [255, 255, 255]
        else:
            cur = box.fill_bg_color or [255, 255, 255]
        color = QColorDialog.getColor(QColor(*cur[:3]), self, title)
        if not color.isValid():
            return
        rgb = [color.red(), color.green(), color.blue()]
        if not self._panel_dirty:
            self._snapshot()
            self._panel_dirty = True
        if kind == "fg":
            box.fg = rgb
        elif kind == "stroke":
            box.bg = rgb
        else:
            box.fill_bg_color = rgb
        self._sync_panel_colors(box)
        self._update_preview(box)

    def _sync_panel_colors(self, box):
        self._set_color_style(self.btn_fg_color, getattr(box, "fg", None) or [0, 0, 0])
        self._set_color_style(self.btn_stroke_color, getattr(box, "bg", None) or [255, 255, 255])
        self._set_color_style(self.btn_fill_color, getattr(box, "fill_bg_color", None) or [255, 255, 255])

    def _panel_set(self, attr, value):
        idx = self._selected_idx
        if not (0 <= idx < len(self._boxes)):
            return
        box = self._boxes[idx]
        if getattr(box, attr, None) == value:
            return
        if not self._panel_dirty:
            self._snapshot()
            self._panel_dirty = True
        setattr(box, attr, value)
        self._update_preview(box)

    def _import_font(self):
        from fonts import import_font, list_fonts
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字体文件", "", "字体 (*.ttf *.ttc *.otf)")
        if not path:
            return
        name = import_font(path)
        if not name:
            self._hint("字体导入失败")
            return
        self.font_combo.blockSignals(True)
        self.font_combo.clear()
        for f in list_fonts():
            self.font_combo.addItem(f["display"], f["name"])
        idx = self.font_combo.findData(name)
        if idx >= 0:
            self.font_combo.setCurrentIndex(idx)
        self.font_combo.blockSignals(False)
        self._hint(f"已导入字体: {name}")

    # ---------------- 服务状态 ----------------

    def _on_service_status(self, state: str):
        if state == "running":
            self.svc_badge.setObjectName("badgeOk")
            self.svc_badge.setText("服务运行中")
            self.svc_label.setText("可直接翻译")
        elif state == "loading":
            self.svc_badge.setObjectName("badgeWarn")
            self.svc_badge.setText("加载中")
            self.svc_label.setText("模型加载中…")
        else:
            self.svc_badge.setObjectName("badgeErr")
            self.svc_badge.setText("服务未运行")
            self.svc_label.setText("翻译前请先启动服务")
        self.svc_badge.style().unpolish(self.svc_badge)
        self.svc_badge.style().polish(self.svc_badge)

    # ---------------- 输入 ----------------

    def _pick_input(self):
        """浏览按钮：弹出菜单选择 文件/压缩包 或 文件夹（取消不会连锁弹出第二个对话框）"""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        a_files = menu.addAction("选择图片 / 压缩包（可多选）")
        a_dir = menu.addAction("选择文件夹")
        a = menu.exec(self.input_edit.mapToGlobal(
            self.input_edit.rect().bottomLeft()))
        if a is a_files:
            files, _ = QFileDialog.getOpenFileNames(self, "选择漫画图片（可多选）", "",
                                                    "图片 (*.jpg *.jpeg *.png *.webp *.bmp);;压缩包 (*.zip *.cbz)")
            if files:
                self.input_edit.setText(";".join(files))
        elif a is a_dir:
            d = QFileDialog.getExistingDirectory(self, "选择漫画文件夹")
            if d:
                self.input_edit.setText(d)

    def _pick_state(self):
        d = QFileDialog.getExistingDirectory(self, "选择调整状态目录")
        if d:
            self.state_edit.setText(d)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.out_edit.setText(d)

    def _open_out(self):
        if self.out_dir and os.path.isdir(self.out_dir):
            os.startfile(self.out_dir)
        else:
            QMessageBox.information(self, "提示", "请先执行「重新嵌字全部」")

    # ---------------- 翻译 ----------------

    def _start_translate(self):
        src = self.input_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "提示", "请先选择输入")
            return

        parts = [p.strip() for p in src.split(";") if p.strip()]
        files = []
        work = None
        if len(parts) > 1 or (len(parts) == 1 and os.path.isfile(parts[0])
                              and not parts[0].lower().endswith((".zip", ".cbz"))):
            for p in parts:
                if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    files.append(p)
            if not files:
                QMessageBox.warning(self, "提示", "没有找到图片文件")
                return
        else:
            src_path = parts[0]
            if not os.path.exists(src_path):
                QMessageBox.warning(self, "提示", "输入路径不存在")
                return
            if os.path.isfile(src_path) and src_path.lower().endswith((".zip", ".cbz")):
                self.info.clear()
                self._hint("正在解压压缩包…")
                work = tempfile.mkdtemp(prefix="wb_zip_")
                with zipfile.ZipFile(src_path) as z:
                    z.extractall(work)
                files = collect_images(work)
            else:
                files = collect_images(src_path) if os.path.isdir(src_path) else [src_path]
        if not files:
            QMessageBox.warning(self, "提示", "没有找到图片")
            return
        files = sorted(files, key=lambda f: natural_key(os.path.basename(f)))

        # 状态目录（固定路径，新任务前清空）
        from paths import ROOT
        state_dir = os.path.join(ROOT, "调整状态")
        os.makedirs(state_dir, exist_ok=True)
        for old in glob.glob(os.path.join(state_dir, "*")):
            try:
                os.remove(old)
            except Exception:
                pass
        self.state_dir = state_dir
        self.state_edit.setText(state_dir)

        self.info.clear()
        self._hint(f"开始逐页翻译: {len(files)} 页 → 状态目录 {state_dir}")
        self._set_busy(True)
        self.progress_bar.setRange(0, len(files))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("翻译中 %v/%m")

        self.worker = TranslateWorker(files, state_dir)
        self.worker.progress.connect(self._on_translate_progress)
        self.worker.log.connect(self._hint)
        self.worker.finished_ok.connect(self._on_translate_done)
        self.worker.start()

    def _on_translate_progress(self, d, t, n):
        self.progress_label.setText(f"{d}/{t} {n}")
        self.progress_bar.setRange(0, t)
        self.progress_bar.setValue(d)

    def _on_translate_done(self, state_dir):
        self._set_busy(False)
        self.progress_label.setText("")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("完成")
        try:
            self.state_edit.setText(state_dir)
            n = len(self._find_jsons(state_dir))
            if n > 0:
                self._load_states(silent=True)
                self._hint(f"已导出 {n} 页调整状态并自动载入，可直接开始调整")
            else:
                last_err = getattr(self.worker, "_last_error", "")
                self._hint(f"翻译完成但未生成状态文件，请查看上方日志。{last_err}")
        except Exception as e:
            self._hint(f"自动载入状态失败: {e}")

    def _set_busy(self, busy: bool):
        """翻译/重嵌期间禁用编辑控件，避免修改被覆盖"""
        for w in (self.btn_translate, self.btn_load, self.btn_prev, self.btn_next,
                  self.btn_save, self.btn_rerender, self.btn_pdf, self.btn_add_box,
                  self.btn_del_box, self.btn_remember, self.btn_fit, self.btn_box_bg,
                  self.btn_focus, self.rerange_combo, self.dir_combo,
                  self.input_edit, self.state_edit, self.out_edit,
                  self.font_spin, self.trans_edit, self.angle_spin,
                  self.orig_check, self.fit_check):
            w.setEnabled(not busy)
        if not busy and not (0 <= self._selected_idx < len(self._boxes)):
            for w in self._style_widgets:
                w.setEnabled(False)
        self._hint("任务进行中，请等候完成…" if busy else "任务完成，可继续编辑")

    # ---------------- 状态载入 ----------------

    def _find_jsons(self, d):
        """查找目录下的 json（含子目录）"""
        jsons = glob.glob(os.path.join(d, "*.json"))
        for sub in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(sub):
                jsons += glob.glob(os.path.join(sub, "*.json"))
        return sorted(jsons, key=lambda j: natural_key(os.path.basename(j)))

    def _load_states(self, silent=False):
        from paths import ROOT
        d = (self.state_edit.text().strip() or os.path.join(ROOT, "调整状态"))
        if not os.path.isdir(d):
            if silent:
                self._hint(f"状态目录无效: {d}")
            else:
                QMessageBox.warning(self, "提示", f"状态目录无效:\n{d}")
            return
        jsons = self._find_jsons(d)
        if not jsons:
            # 兜底: 在项目目录下查找历史状态目录(调整状态*)
            for cand in sorted(glob.glob(os.path.join(ROOT, "调整状态*")), reverse=True):
                if os.path.isdir(cand):
                    jsons = self._find_jsons(cand)
                    if jsons:
                        d = cand
                        break
        if not jsons:
            if silent:
                self._hint(f"目录中没有 .json 状态文件: {d}")
            else:
                QMessageBox.warning(self, "提示",
                                    f"目录中没有 .json 状态文件:\n{d}\n\n"
                                    "请先点击「翻译并导出调整状态」，或浏览选择正确的状态目录。")
            return
        self.state_dir = d
        self.state_edit.setText(d)
        self.jsons = jsons
        self.idx = 0
        self._load_page(silent=silent)
        self._hint(f"已载入 {len(self.jsons)} 页状态（{d}），开始调整文本框吧")

    def _load_page(self, silent=False):
        self.scene.clear()
        self._boxes = []
        self._selected_idx = -1
        self._selected_box = None
        self._panel_dirty = False
        self.trans_edit.blockSignals(True)
        self.trans_edit.clear()
        self.trans_edit.blockSignals(False)
        for w in self._style_widgets:
            w.setEnabled(False)
        j = self.jsons[self.idx]
        base = os.path.splitext(j)[0]
        inpainted = base + "_inpainted.png"
        if not os.path.exists(inpainted):
            if silent:
                self._hint(f"缺少擦字图: {inpainted}")
            else:
                QMessageBox.warning(self, "错误", f"缺少擦字图: {inpainted}")
            return
        pix = QPixmap(inpainted)
        if pix.isNull():
            if silent:
                self._hint(f"图片加载失败: {inpainted}")
            else:
                QMessageBox.warning(self, "错误", "图片加载失败")
            return
        self.scene.addPixmap(pix)
        # 原图参照层（与擦字图同分辨率，1:1 对齐）
        self._orig_overlay = None
        orig_path = base + "_orig.png"
        if os.path.exists(orig_path):
            opix = QPixmap(orig_path)
            if not opix.isNull():
                self._orig_overlay = self.scene.addPixmap(opix)
                self._orig_overlay.setZValue(0.5)
                self._orig_overlay.setOpacity(0.55)
                self._orig_overlay.setVisible(self.orig_check.isChecked())
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, d in enumerate(data.get("regions", [])):
            xs = [p[0] for p in d["lines"]]
            ys = [p[1] for p in d["lines"]]
            rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            box = TextBox(rect, i, d.get("text", ""), d.get("translation", ""))
            box.text_height = d.get("font_size", 22)
            box.angle = d.get("angle", 0)
            box.fg = d.get("fg") or [0, 0, 0]
            box.bg = d.get("bg")
            # PS 式样式字段（旧状态缺省 None）
            for k in ("font", "font_bold", "stroke_w", "line_spacing", "char_spacing",
                      "align_h", "align_v", "force_dir", "fill_bg", "fill_bg_color"):
                if k in d:
                    setattr(box, k, d[k])
            box.on_geometry_changed = self._on_box_geometry_changed
            self.scene.addItem(box)
            box.set_visible_handles(False)
            self._boxes.append(box)
            self._update_preview(box)
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.page_label.setText(f"{os.path.basename(j)}  ({self.idx + 1}/{len(self.jsons)})  文本框: {len(self._boxes)}")
        self.info.setPlainText("")
        self._hint(f"当前页: {os.path.basename(j)}  拖动蓝色框移动，拖角点缩放，点「保存本页」")
        self._fit_view()

    def _fit_view(self):
        """让整页图适应视图大小（滚轮仍可缩放）"""
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _toggle_focus(self):
        """全屏编辑模式: 隐藏输入/输出设置与日志卡片，图片操作区最大化"""
        self._focused = not getattr(self, "_focused", False)
        for w in (self.card_settings, self.card_log):
            w.setVisible(not self._focused)
        self.btn_focus.setText("退出全屏" if self._focused else "全屏编辑")
        self.btn_focus.setObjectName("primary" if self._focused else "")
        self.btn_focus.style().unpolish(self.btn_focus)
        self.btn_focus.style().polish(self.btn_focus)
        self._fit_view()

    # ---------------- 调整交互 ----------------

    def _goto(self, i):
        if 0 <= i < len(self.jsons):
            self.idx = i
            self._load_page()

    def _on_box_selected(self, idx):
        self._selected_idx = idx
        self._panel_dirty = False
        if idx < 0 or idx >= len(self._boxes):
            self._selected_box = None
            for w in self._style_widgets:
                w.setEnabled(False)
            return
        self._selected_box = self._boxes[idx]
        box = self._boxes[idx]
        self.font_spin.blockSignals(True)
        self.font_spin.setValue(box.text_height or 22)
        self.font_spin.blockSignals(False)
        self.trans_edit.blockSignals(True)
        self.trans_edit.setText(box.translation or "")
        self.trans_edit.blockSignals(False)
        # 字符面板同步（blockSignals 防递归）
        self.font_combo.blockSignals(True)
        idxf = self.font_combo.findData(getattr(box, "font", None))
        if idxf < 0:
            idxf = self.font_combo.findData("msyh.ttc")
        self.font_combo.setCurrentIndex(max(0, idxf))
        self.font_combo.blockSignals(False)
        self.bold_check.blockSignals(True)
        self.bold_check.setChecked(bool(getattr(box, "font_bold", False)))
        self.bold_check.blockSignals(False)
        self.stroke_spin.blockSignals(True)
        self.stroke_spin.setValue(int(getattr(box, "stroke_w", 0) or 0))
        self.stroke_spin.blockSignals(False)
        self.fill_check.blockSignals(True)
        self.fill_check.setChecked(bool(getattr(box, "fill_bg", False)))
        self.fill_check.blockSignals(False)
        self.line_spin.blockSignals(True)
        self.line_spin.setValue(float(getattr(box, "line_spacing", 0) or 1.2))
        self.line_spin.blockSignals(False)
        self.char_spin.blockSignals(True)
        self.char_spin.setValue(int(getattr(box, "char_spacing", 0) or 0))
        self.char_spin.blockSignals(False)
        self.align_h_combo.blockSignals(True)
        self.align_h_combo.setCurrentIndex(max(0, self.align_h_combo.findData(
            getattr(box, "align_h", None) or "center")))
        self.align_h_combo.blockSignals(False)
        self.align_v_combo.blockSignals(True)
        self.align_v_combo.setCurrentIndex(max(0, self.align_v_combo.findData(
            getattr(box, "align_v", None) or "center")))
        self.align_v_combo.blockSignals(False)
        self.force_combo.blockSignals(True)
        self.force_combo.setCurrentIndex(max(0, self.force_combo.findData(
            getattr(box, "force_dir", None) or "auto")))
        self.force_combo.blockSignals(False)
        self.angle_spin.blockSignals(True)
        self.angle_spin.setValue(int(box.angle or 0))
        self.angle_spin.blockSignals(False)
        self._sync_panel_colors(box)
        for w in self._style_widgets:
            w.setEnabled(True)
        self.info.setPlainText(f"原文: {box.text}\n译文: {box.translation}")
        self._update_preview(box)

    def _box_vertical(self, box) -> bool:
        force = getattr(box, "force_dir", None) or "auto"
        if force == "h":
            return False
        if force == "v":
            return True
        r = box.rect()
        return r.height() >= r.width()

    def _update_preview(self, box):
        """在框内实时预览（PIL 渲染，画布=框尺寸，与重嵌一致）"""
        if box.preview_item is None:
            box.preview_item = QGraphicsPixmapItem(box)
            box.preview_item.setZValue(1)
        font_size = box.text_height or 22
        r = box.rect()
        bw, bh = int(r.width()), int(r.height())
        img = render_text_preview(box.translation or box.text, font_size,
                                  self._box_vertical(box), bw, bh, box.fg, box.bg,
                                  rtl=self._current_rtl(),
                                  font_name=getattr(box, "font", None),
                                  bold=bool(getattr(box, "font_bold", False)),
                                  stroke_width=getattr(box, "stroke_w", None),
                                  line_spacing=getattr(box, "line_spacing", None) or 1.2,
                                  char_spacing=int(getattr(box, "char_spacing", 0) or 0),
                                  align_h=getattr(box, "align_h", None) or "center",
                                  align_v=getattr(box, "align_v", None) or "center",
                                  fill_bg=bool(getattr(box, "fill_bg", False)),
                                  fill_bg_color=getattr(box, "fill_bg_color", None) or (255, 255, 255))
        if img is None:
            box.preview_item.hide()
            return
        arr = img.convert("RGBA").tobytes("raw", "RGBA")
        qimg = QImage(arr, img.width, img.height, img.width * 4, QImage.Format_RGBA8888).copy()
        box.preview_item.setPixmap(QPixmap.fromImage(qimg))
        box.preview_item.setPos(r.left(), r.top())
        box.preview_item.show()

    def _current_rtl(self) -> bool:
        """当前阅读方向是否右→左（auto 默认右→左）"""
        return self.dir_combo.currentData() != "ltr"

    def _on_dir_change(self, _):
        for box in self._boxes:
            self._update_preview(box)

    def _reindex_boxes(self):
        for i, b in enumerate(self._boxes):
            b.index = i

    def _add_box(self):
        if not self.jsons:
            QMessageBox.warning(self, "提示", "请先载入状态")
            return
        self._snapshot()
        # 默认白底黑字 + 实心底（白底抹去原文，黑字显示译文）
        view_rect = self.view.viewport().rect()
        size = 140, 60
        rect = QRectF(self.view.mapToScene(view_rect.center()).x() - size[0] / 2,
                      self.view.mapToScene(view_rect.center()).y() - size[1] / 2,
                      size[0], size[1])
        box = TextBox(rect, len(self._boxes), "", "")
        box.text_height = 22
        box.angle = 0
        box.fg = [0, 0, 0]
        box.bg = [255, 255, 255]
        box.fill_bg = True
        box.fill_bg_color = [255, 255, 255]
        box.on_geometry_changed = self._on_box_geometry_changed
        self.scene.addItem(box)
        self._boxes.append(box)
        self._reindex_boxes()
        self._on_box_selected(box.index)
        self._hint("已添加文本框: 白底黑字盖住原文，拖到目标位置、拖角点缩放，在「译文」框填写内容")

    def _pick_box_fg(self):
        """修改选中文本框的文字颜色（默认黑字），预览与重嵌同步"""
        idx = self._selected_idx
        if not (0 <= idx < len(self._boxes)):
            self._hint("请先选中一个文本框")
            return
        box = self._boxes[idx]
        cur = box.fg or [0, 0, 0]
        color = QColorDialog.getColor(
            QColor(*cur[:3]), self, "选择文字颜色")
        if not color.isValid():
            return
        if not self._panel_dirty:
            self._snapshot()
            self._panel_dirty = True
        box.fg = [color.red(), color.green(), color.blue()]
        self._sync_panel_colors(box)
        self._update_preview(box)
        self._hint(f"文字颜色已改为 RGB{tuple(box.fg)}（保存后生效）")

    def _delete_box(self, idx=None):
        # 优先用对象引用（正在编辑的框），索引作兜底，彻底避免删错框
        box = getattr(self, "_selected_box", None)
        if box is None or box not in self._boxes:
            if idx is None or idx < 0:
                idx = self._selected_idx
            if not (0 <= idx < len(self._boxes)):
                self._hint("请先选中一个文本框再删除")
                return
            box = self._boxes[idx]
        self._snapshot()
        self.scene.removeItem(box)
        self._boxes.remove(box)
        self._reindex_boxes()
        self._selected_idx = -1
        self._selected_box = None
        self._on_box_selected(-1)
        self._hint(f"已删除文本框（剩余 {len(self._boxes)} 个，保存后生效）")

    def _remember_translation(self):
        idx = self._selected_idx
        if not (0 <= idx < len(self._boxes)):
            self._hint("请先选中一个文本框")
            return
        box = self._boxes[idx]
        src = (box.text or "").strip()
        dst = (box.translation or "").strip()
        if not src or not dst:
            self._hint("原文或译文为空，无法记住")
            return
        Glossary().add(src, dst, "global")
        self._hint(f"已加入全局术语表: {src} => {dst}")

    def _on_font_change(self, v):
        if 0 <= self._selected_idx < len(self._boxes):
            box = self._boxes[self._selected_idx]
            box.text_height = v
            self._update_preview(box)

    def _on_orig_toggle(self, v):
        if self._orig_overlay is not None:
            self._orig_overlay.setVisible(v)

    def _on_trans_edit(self, text):
        if 0 <= self._selected_idx < len(self._boxes):
            box = self._boxes[self._selected_idx]
            box.translation = text
            self._update_preview(box)

    def _on_angle_change(self, v):
        if 0 <= self._selected_idx < len(self._boxes):
            box = self._boxes[self._selected_idx]
            if not self._panel_dirty:
                self._snapshot()
                self._panel_dirty = True
            box.angle = v
            self._update_preview(box)

    def _on_box_geometry_changed(self, box):
        """拖动/缩放文本框后：按新框尺寸自动适配字号（可开关），预览与输出保持一致"""
        import time as _t
        now = _t.monotonic()
        if now - self._last_geo_snap > 1.5:
            self._snapshot()
            self._last_geo_snap = now
        if not self.fit_check.isChecked():
            return
        r = box.rect()
        w, h = r.width(), r.height()
        if w < 12 or h < 12:
            return
        new_font = int((w * 0.9) if h >= w else (h * 0.75))
        new_font = max(6, min(new_font, 120))
        if new_font == box.text_height:
            return
        box.text_height = new_font
        if self._selected_idx == box.index:
            self.font_spin.blockSignals(True)
            self.font_spin.setValue(new_font)
            self.font_spin.blockSignals(False)
            self._update_preview(box)

    # ---------------- 快照 / 撤销（页面级） ----------------

    def _regions_data(self) -> dict:
        """从当前 _boxes 全量序列化页面数据（含 PS 样式字段）"""
        with open(self.jsons[self.idx], "r", encoding="utf-8") as f:
            data = json.load(f)
        old = data.get("regions", [])
        regions_new = []
        for i, box in enumerate(self._boxes):
            r = box.rect()
            sx, sy = box.pos().x(), box.pos().y()  # 拖动改的是 pos，需叠加
            d = dict(old[i]) if i < len(old) else {
                "text": "", "translation": "", "angle": 0.0, "font_size": 22,
                "fg": [0, 0, 0], "bg": [255, 255, 255],
                "direction": "auto", "alignment": "auto",
            }
            new_bounds = (sx + r.left(), sy + r.top(), sx + r.right(), sy + r.bottom())
            old_lines = d.get("lines") or []
            old_bounds = _bounds_for_lines(old_lines)
            if old_bounds:
                d["lines"] = _scale_points(old_lines, old_bounds, new_bounds)
            else:
                x1, y1, x2, y2 = new_bounds
                d["lines"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            source_segments = d.get("source_segments") or []
            if source_segments and old_bounds:
                for segment in source_segments:
                    segment["lines"] = _scale_points(segment.get("lines") or [], old_bounds, new_bounds)
            if box.text_height:
                d["font_size"] = box.text_height
            d["translation"] = box.translation or ""
            source_texts = [segment.get("text", "") for segment in source_segments]
            aligned = align_translation_segments(source_texts, d["translation"])
            if aligned:
                d["translation_segments"] = aligned
            else:
                d.pop("translation_segments", None)
            if box.angle is not None:
                d["angle"] = float(box.angle)
            if box.fg is not None:
                d["fg"] = list(box.fg)
            if box.bg is not None:
                d["bg"] = list(box.bg)
            # PS 样式字段（仅写入非默认值，保持旧状态兼容）
            for attr, key in (("font", "font"), ("font_bold", "font_bold"),
                              ("stroke_w", "stroke_w"), ("line_spacing", "line_spacing"),
                              ("char_spacing", "char_spacing"), ("align_h", "align_h"),
                              ("align_v", "align_v"), ("force_dir", "force_dir"),
                              ("fill_bg", "fill_bg")):
                v = getattr(box, attr, None)
                if v is not None:
                    d[key] = v
            if getattr(box, "fill_bg_color", None) is not None and box.fill_bg:
                d["fill_bg_color"] = list(box.fill_bg_color)
            regions_new.append(d)
        data["regions"] = regions_new
        return data

    def _snapshot(self):
        if not self.jsons:
            return
        j = self.jsons[self.idx]
        stack = self._undo_stack.setdefault(j, [])
        stack.append(self._regions_data())
        if len(stack) > 30:
            stack.pop(0)

    def _undo(self):
        if not self.jsons:
            return
        j = self.jsons[self.idx]
        stack = self._undo_stack.get(j, [])
        if not stack:
            self._hint("没有可撤销的操作")
            return
        data = stack.pop()
        with open(j, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self._load_page()
        self._hint("已撤销（Ctrl+Z）")

    def _save_page(self):
        if not self.jsons:
            return
        j = self.jsons[self.idx]
        data = self._regions_data()
        with open(j, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self._hint(f"已保存: {os.path.basename(j)}（{len(data['regions'])} 个文本框）")

    # ---------------- 重嵌字 / PDF ----------------

    def _rerender_all(self):
        if not self.jsons:
            QMessageBox.warning(self, "提示", "请先载入状态")
            return
        self._save_page()  # 先保存当前页的调整
        # 范围: 全部页面 或 仅当前页（部分嵌字）
        if self.rerange_combo.currentData() == "page":
            jsons = [self.jsons[self.idx]]
        else:
            jsons = list(self.jsons)
        from paths import ROOT
        out_dir = self.out_edit.text().strip() or os.path.join(ROOT, "调整状态", "输出")
        self.out_dir = out_dir
        self._set_busy(True)
        self.progress_bar.setRange(0, len(jsons))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("嵌字中 %v/%m")
        self.worker2 = RerenderWorker(jsons, out_dir, self.fmt_combo.currentData(),
                                      self.dir_combo.currentData())
        self.worker2.progress.connect(self._on_rerender_progress)
        self.worker2.log.connect(self._hint)
        self.worker2.done.connect(self._on_rerender_done)
        self.worker2.start()

    def _on_rerender_progress(self, d, t, n):
        self.progress_label.setText(f"嵌字中 {d}/{t} {n}")
        self.progress_bar.setRange(0, t)
        self.progress_bar.setValue(d)

    def _on_rerender_done(self, out_dir, err):
        self._set_busy(False)
        self.progress_label.setText("")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("完成")
        self.out_dir = out_dir
        self.out_edit.setText(out_dir)
        self._hint(f"重新嵌字完成: {out_dir}" + (f"\n失败页:\n{err}" if err else ""))
        try:
            os.startfile(out_dir)
        except Exception:
            pass

    def _merge_pdf(self):
        if not self.out_dir or not os.path.isdir(self.out_dir):
            QMessageBox.warning(self, "提示", "请先执行「重新嵌字全部」")
            return
        from paths import ROOT
        pdf = os.path.join(os.path.dirname(self.out_dir),
                           os.path.basename(self.out_dir) + ".pdf")
        self._hint("正在合并 PDF…")
        r = subprocess.run([PYTHON, os.path.join(ROOT, "merge_pdf.py"),
                            self.out_dir, pdf],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=CREATE_NO_WINDOW)
        if r.returncode == 0:
            self._hint(f"PDF 已生成: {pdf}")
        else:
            self._hint(f"PDF 合并失败: {r.stdout[-300:] if r.stdout else r.stderr[-300:]}")


class StandaloneWindow(QMainWindow):
    """独立运行模式（start_workbench.bat 备用入口）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("翻译调整工作台")
        self.resize(1200, 860)
        self.setCentralWidget(WorkbenchPage(self))


def main():
    from theme import apply_theme, is_dark_mode
    app = QApplication(sys.argv)
    apply_theme(app, is_dark_mode())
    win = StandaloneWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
