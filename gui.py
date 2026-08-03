"""快速翻译页 - 可嵌入统一界面 / 独立运行双模式

功能: 选择漫画文件/文件夹/zip → 一键翻译嵌字 → 输出到桌面\\翻译
作为页面嵌入 app.py 时传入共享的 SakuraManager；独立运行时自建。
"""
import os
import subprocess
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, QPlainTextEdit,
    QFrame, QFileDialog, QComboBox,
)

from paths import PYTHON, TRANSLATE
from service_manager import SakuraManager

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def make_card(title: str = None) -> tuple:
    """创建卡片容器, 返回 (frame, vbox)"""
    card = QFrame()
    card.setObjectName("card")
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 14, 16, 14)
    v.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("h2")
        v.addWidget(t)
    return card, v


class TranslateWorker(QThread):
    """后台线程: 调用 translate.py 子进程并转发输出"""
    log = Signal(str)
    done = Signal(int, str)  # returncode, 输出路径

    def __init__(self, args: list, out_dir: str):
        super().__init__()
        self.args = args
        self.out_dir = out_dir
        self._proc = None

    def run(self):
        cmd = [PYTHON, TRANSLATE, *self.args]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW)
        except Exception as e:
            self.done.emit(-1, f"启动失败: {e}")
            return
        for line in self._proc.stdout:
            self.log.emit(line.rstrip())
        code = self._proc.wait()
        self.done.emit(code, self.out_dir)

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class TranslatePage(QWidget):
    """快速翻译页面（卡片式布局）"""

    def __init__(self, service: SakuraManager = None, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.worker = None
        self.service = service if service is not None else SakuraManager(self)
        self.service.status_changed.connect(self._on_service_status)
        self._init_ui()
        self._on_service_status(self.service.is_running() and "running" or "stopped")

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        # ---- 标题行 ----
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("快速翻译")
        title.setObjectName("title")
        sub = QLabel("日文漫画 → 中文嵌字，一键完成")
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

        # ---- 输入卡片 ----
        in_card, in_v = make_card("输入")
        in_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("选择漫画图片 / 文件夹 / zip / cbz")
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._browse)
        in_row.addWidget(self.input_edit, 1)
        in_row.addWidget(btn_browse)
        in_v.addLayout(in_row)
        root.addWidget(in_card)

        # ---- 选项卡片 ----
        opt_card, opt_v = make_card("选项")
        form = QFormLayout()
        form.setSpacing(10)

        self.font_min = QSpinBox()
        self.font_min.setRange(8, 60)
        self.font_min.setValue(22)
        self.font_min.setSuffix(" px")
        form.addRow("最小字号:", self.font_min)

        self.format_combo = QComboBox()
        self.format_combo.addItem("与原图相同", None)
        self.format_combo.addItem("png", "png")
        self.format_combo.addItem("jpg", "jpg")
        self.format_combo.addItem("webp", "webp")
        form.addRow("输出格式:", self.format_combo)

        self.dir_combo = QComboBox()
        self.dir_combo.addItem("跟随原文(竖排右→左)", "auto")
        self.dir_combo.addItem("从左到右", "ltr")
        self.dir_combo.addItem("从右到左", "rtl")
        form.addRow("阅读方向:", self.dir_combo)

        self.sfx_check = QCheckBox("气泡内拟声词翻译，画面上的保留原文")
        self.sfx_check.setChecked(True)
        form.addRow("拟声词:", self.sfx_check)

        self.overwrite_check = QCheckBox("覆盖已存在的输出文件")
        form.addRow("输出:", self.overwrite_check)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("留空 = 桌面\\翻译\\ 按序号命名")
        btn_out = QPushButton("浏览…")
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(btn_out)
        form.addRow("输出目录:", out_row)

        self.state_check = QCheckBox("同时导出调整状态（供工作台调整）")
        self.state_check.toggled.connect(self._on_state_toggle)
        form.addRow("调整状态:", self.state_check)

        state_row = QHBoxLayout()
        self.state_edit = QLineEdit()
        self.state_edit.setPlaceholderText("留空 = 桌面\\翻译\\调整状态\\<输入名>\\")
        btn_state = QPushButton("浏览…")
        btn_state.clicked.connect(self._browse_state)
        state_row.addWidget(self.state_edit, 1)
        state_row.addWidget(btn_state)
        form.addRow("状态目录:", state_row)
        self._on_state_toggle(False)

        opt_v.addLayout(form)
        root.addWidget(opt_card)

        # ---- 操作行 ----
        act_card, act_v = make_card()
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始翻译")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start)
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        self.btn_sakura = QPushButton("启动翻译服务")
        self.btn_sakura.clicked.connect(self.service.start)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_sakura)
        act_v.addLayout(btn_row)
        root.addWidget(act_card)

        # ---- 日志卡片 ----
        log_card, log_v = make_card("运行日志")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        log_v.addWidget(self.log_view)
        root.addWidget(log_card, 1)

        self.input_edit.textChanged.connect(
            lambda t: self.btn_start.setEnabled(bool(t.strip())))

    # ---------- 服务状态 ----------

    def _on_service_status(self, state: str):
        if state == "running":
            self.svc_badge.setObjectName("badgeOk")
            self.svc_badge.setText("服务运行中")
            self.svc_label.setText("sakura 已就绪，可直接翻译")
            self.btn_sakura.setText("翻译服务已启动")
            self.btn_sakura.setEnabled(False)
        elif state == "loading":
            self.svc_badge.setObjectName("badgeWarn")
            self.svc_badge.setText("加载中")
            self.svc_label.setText("模型加载约 20 秒，请稍候…")
            self.btn_sakura.setText("加载中…")
            self.btn_sakura.setEnabled(False)
        else:
            self.svc_badge.setObjectName("badgeErr")
            self.svc_badge.setText("服务未运行")
            self.svc_label.setText("翻译前需先启动翻译服务")
            self.btn_sakura.setText("启动翻译服务")
            self.btn_sakura.setEnabled(True)
        self.svc_badge.style().unpolish(self.svc_badge)
        self.svc_badge.style().polish(self.svc_badge)

    # ---------- 行为 ----------

    def _browse(self):
        """浏览按钮：弹出菜单选择 文件/压缩包 或 文件夹（取消不会连锁弹出第二个对话框）"""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        a_files = menu.addAction("选择图片 / 压缩包")
        a_dir = menu.addAction("选择文件夹")
        a = menu.exec(self.input_edit.mapToGlobal(
            self.input_edit.rect().bottomLeft()))
        if a is a_files:
            path, _ = QFileDialog.getOpenFileName(self, "选择漫画", "",
                                                  "图片 (*.jpg *.jpeg *.png *.webp *.bmp);;压缩包 (*.zip *.cbz)")
            if path:
                self.input_edit.setText(path)
        elif a is a_dir:
            folder = QFileDialog.getExistingDirectory(self, "选择漫画文件夹")
            if folder:
                self.input_edit.setText(folder)

    def _browse_out(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.out_edit.setText(folder)

    def _browse_state(self):
        folder = QFileDialog.getExistingDirectory(self, "选择调整状态目录")
        if folder:
            self.state_edit.setText(folder)

    def _on_state_toggle(self, checked):
        self.state_edit.setEnabled(checked)

    def _default_state_dir(self, src: str) -> str:
        base = os.path.splitext(os.path.basename(src.rstrip("\\/")))[0] or "输入"
        d = os.path.join(os.path.expanduser("~"), "Desktop", "翻译", "调整状态", base)
        # 防冲突: 状态目录不得与输入路径相同或嵌套（避免状态写进输入目录）
        src_abs = os.path.abspath(src.rstrip("\\/"))
        if os.path.abspath(d).startswith(src_abs + os.sep) or os.path.abspath(d) == src_abs:
            d += "_状态"
        return d

    def _start(self):
        src = self.input_edit.text().strip()
        if not src:
            return
        # 位置参数必须紧邻放在前面（argparse 坑: 选项夹在位置参数中间时
        # nargs="?" 的 output 不会被识别，会报 unrecognized arguments）
        args = [src]
        out = self.out_edit.text().strip()
        if out:
            args.append(out)
        args.append("--no-popup")
        args.append(f"--font-min={self.font_min.value()}")
        fmt = self.format_combo.currentData()
        if fmt:
            args.append(f"--format={fmt}")
        args.append(f"--direction={self.dir_combo.currentData()}")
        if not self.sfx_check.isChecked():
            args.append("--no-sfx")
        if self.overwrite_check.isChecked():
            args.append("--overwrite")
        if self.state_check.isChecked():
            state_dir = self.state_edit.text().strip() or self._default_state_dir(src)
            args.extend(["--manual-dir", state_dir])

        self.log_view.clear()
        self._log(f"开始翻译: {src}")
        self._log(f"命令: {PYTHON} {TRANSLATE} {' '.join(args)}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = TranslateWorker(args, out)
        self.worker.log.connect(self._log)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self._log("[停止] 正在终止…")

    def _on_done(self, code, out):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self._log(f"\n✅ 翻译完成，输出: {out}")
            self.svc_label.setText(f"翻译完成: {out}")
        else:
            self._log("\n❌ 翻译失败（见上方日志）")

    def _log(self, text):
        self.log_view.appendPlainText(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


class StandaloneWindow(QMainWindow):
    """独立运行模式（start_gui.bat 备用入口）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("漫画翻译助手")
        self.resize(760, 640)
        self.setCentralWidget(TranslatePage(self))


def main():
    from theme import apply_theme, is_dark_mode
    app = QApplication(sys.argv)
    apply_theme(app, is_dark_mode())
    win = StandaloneWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
