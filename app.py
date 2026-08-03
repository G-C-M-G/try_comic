"""漫画翻译助手 - 统一入口（侧边栏导航 + 三页面）

整合原三个入口: 快速翻译(gui.py) / 调整工作台(workbench.py) / 翻译服务(sakura)。
启动无控制台（用 pythonw / PyInstaller --noconsole），sakura 服务由本界面统一管理。
"""
import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QMessageBox,
    QPlainTextEdit, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QLineEdit, QAbstractItemView,
)

from paths import ROOT, ICON_PATH, SAKURA_LOG
from theme import apply_theme, is_dark_mode, load_ui_config, save_ui_config, fit_buttons
from service_manager import SakuraManager
from settings import load_settings, save_settings
from gui import TranslatePage
from workbench import WorkbenchPage
from glossary import Glossary
from memory import TranslationMemory
from cache import PageCache

PAGE_NAMES = ["快速翻译", "调整工作台", "词典与缓存", "翻译服务"]


def _card(title: str = None) -> tuple:
    c = QFrame()
    c.setObjectName("card")
    v = QVBoxLayout(c)
    v.setContentsMargins(16, 14, 16, 14)
    v.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("h2")
        v.addWidget(t)
    return c, v


def _reload_table(table: QTableWidget, rows: list):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(str(val)))


class DictionaryPage(QWidget):
    """词典与缓存页: 术语表 / 翻译记忆 / 译文缓存"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self._init_ui()
        self._refresh_all()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        head = QVBoxLayout()
        t = QLabel("词典与缓存")
        t.setObjectName("title")
        s = QLabel("术语表（译名一致）· 翻译记忆（越用越稳定）· 译文缓存（跳过已译页）")
        s.setObjectName("subtitle")
        head.addWidget(t)
        head.addWidget(s)
        root.addLayout(head)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_glossary_tab(), "术语表")
        self.tabs.addTab(self._build_memory_tab(), "翻译记忆")
        self.tabs.addTab(self._build_cache_tab(), "译文缓存")
        root.addWidget(self.tabs, 1)

    # ---------- 术语表 ----------

    def _build_glossary_tab(self):
        card, v = _card("术语表：翻译前把原文替换为指定译文，保证译名一致（如 アリス => 阿莉苏）")
        self.gloss_enable = QCheckBox("启用术语表")
        self.gloss_enable.toggled.connect(
            lambda chk: save_settings({"enable_glossary": chk}))
        v.addWidget(self.gloss_enable)

        self.gloss_table = QTableWidget()
        self.gloss_table.setColumnCount(4)
        self.gloss_table.setHorizontalHeaderLabels(["原文", "译文", "范围", "目录"])
        self.gloss_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.gloss_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.gloss_table.setColumnWidth(2, 80)
        self.gloss_table.setColumnWidth(3, 120)
        self.gloss_table.setEditTriggers(QAbstractItemView.DoubleClicked |
                                         QAbstractItemView.EditKeyPressed)
        v.addWidget(self.gloss_table, 1)

        row = QHBoxLayout()
        btn_add = QPushButton("＋ 添加一行")
        btn_add.clicked.connect(lambda: self.gloss_table.insertRow(self.gloss_table.rowCount()))
        btn_del = QPushButton("－ 删除选中行")
        btn_del.clicked.connect(self._glossary_delete)
        btn_save = QPushButton("保存术语表")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._glossary_save)
        row.addWidget(btn_add)
        row.addWidget(btn_del)
        row.addStretch(1)
        row.addWidget(btn_save)
        v.addLayout(row)
        return card

    def _glossary_delete(self):
        rows = sorted({i.row() for i in self.gloss_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.gloss_table.removeRow(r)

    def _glossary_save(self):
        g = Glossary()
        g.entries = []
        for r in range(self.gloss_table.rowCount()):
            src = (self.gloss_table.item(r, 0).text() if self.gloss_table.item(r, 0) else "").strip()
            dst = (self.gloss_table.item(r, 1).text() if self.gloss_table.item(r, 1) else "").strip()
            if not src or not dst:
                continue
            scope = (self.gloss_table.item(r, 2).text() if self.gloss_table.item(r, 2) else "").strip()
            dname = (self.gloss_table.item(r, 3).text() if self.gloss_table.item(r, 3) else "").strip()
            if scope != "全局" and dname:
                g.entries.append({"src": src, "dst": dst, "scope": "dir", "dir": dname})
            else:
                g.entries.append({"src": src, "dst": dst, "scope": "global", "dir": ""})
        g.save()
        self._refresh_glossary()
        self._info("术语表已保存")

    # ---------- 翻译记忆 ----------

    def _build_memory_tab(self):
        card, v = _card()
        top = QHBoxLayout()
        self.mem_stats = QLabel("")
        self.mem_stats.setObjectName("muted")
        top.addWidget(self.mem_stats)
        top.addStretch(1)
        self.mem_enable = QCheckBox("启用翻译记忆")
        self.mem_enable.toggled.connect(
            lambda chk: save_settings({"enable_memory": chk}))
        top.addWidget(self.mem_enable)
        v.addLayout(top)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("搜索:"))
        self.mem_search = QLineEdit()
        self.mem_search.setPlaceholderText("按原文/译文关键字过滤")
        self.mem_search.textChanged.connect(self._refresh_memory)
        search_row.addWidget(self.mem_search, 1)
        v.addLayout(search_row)

        self.mem_table = QTableWidget()
        self.mem_table.setColumnCount(2)
        self.mem_table.setHorizontalHeaderLabels(["原文", "译文"])
        self.mem_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.mem_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        v.addWidget(self.mem_table, 1)

        row = QHBoxLayout()
        btn_del = QPushButton("－ 删除选中")
        btn_del.clicked.connect(self._memory_delete)
        btn_clear = QPushButton("清空全部")
        btn_clear.clicked.connect(self._memory_clear)
        row.addWidget(btn_del)
        row.addStretch(1)
        row.addWidget(btn_clear)
        v.addLayout(row)
        return card

    def _memory_delete(self):
        m = TranslationMemory()
        rows = sorted({i.row() for i in self.mem_table.selectedIndexes()}, reverse=True)
        items = self._memory_filtered()
        for r in rows:
            if r < len(items):
                m.data.pop(items[r][0], None)
        m.save()
        self._refresh_memory()

    def _memory_clear(self):
        if QMessageBox.question(self, "确认", "确定清空全部翻译记忆？") == QMessageBox.Yes:
            TranslationMemory().clear()
            self._refresh_memory()

    def _memory_filtered(self):
        kw = self.mem_search.text().strip().lower()
        items = []
        for k, v in TranslationMemory().data.items():
            if kw and kw not in k.lower() and kw not in v.lower():
                continue
            items.append((k, v))
            if len(items) >= 500:
                break
        return items

    # ---------- 译文缓存 ----------

    def _build_cache_tab(self):
        card, v = _card("缓存已翻译页的成品图+调整状态；整卷重翻时命中直接跳过，不重复翻译")
        top = QHBoxLayout()
        self.cache_stats = QLabel("")
        self.cache_stats.setObjectName("muted")
        top.addWidget(self.cache_stats)
        top.addStretch(1)
        self.cache_enable = QCheckBox("启用译文缓存")
        self.cache_enable.toggled.connect(
            lambda chk: save_settings({"enable_cache": chk}))
        top.addWidget(self.cache_enable)
        v.addLayout(top)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("缓存上限 (MB):"))
        self.cache_limit = QSpinBox()
        self.cache_limit.setRange(100, 20000)
        self.cache_limit.setSuffix(" MB")
        btn_limit = QPushButton("应用")
        btn_limit.setObjectName("primary")
        btn_limit.clicked.connect(self._cache_apply_limit)
        limit_row.addWidget(self.cache_limit)
        limit_row.addWidget(btn_limit)
        limit_row.addStretch(1)
        btn_clear = QPushButton("清空缓存")
        btn_clear.clicked.connect(self._cache_clear)
        limit_row.addWidget(btn_clear)
        v.addLayout(limit_row)

        hint = QLabel("提示：修改术语表/记忆/选项后，旧缓存会自动失效；也可直接清空。")
        hint.setObjectName("muted")
        v.addWidget(hint)
        return card

    def _cache_apply_limit(self):
        save_settings({"cache_limit_mb": self.cache_limit.value()})
        PageCache(limit_mb=self.cache_limit.value()).evict()
        self._refresh_cache()

    def _cache_clear(self):
        if QMessageBox.question(self, "确认", "确定清空全部译文缓存？") == QMessageBox.Yes:
            PageCache().clear()
            self._refresh_cache()

    # ---------- 刷新 ----------

    def _refresh_all(self):
        s = load_settings()
        self.gloss_enable.setChecked(bool(s.get("enable_glossary", True)))
        self.mem_enable.setChecked(bool(s.get("enable_memory", True)))
        self.cache_enable.setChecked(bool(s.get("enable_cache", True)))
        self.cache_limit.setValue(int(s.get("cache_limit_mb", 2048)))
        self._refresh_glossary()
        self._refresh_memory()
        self._refresh_cache()

    def _refresh_glossary(self):
        g = Glossary()
        rows = []
        for e in g.entries:
            scope = "全局" if e["scope"] != "dir" else "目录"
            rows.append((e["src"], e["dst"], scope, e.get("dir", "")))
        _reload_table(self.gloss_table, rows)

    def _refresh_memory(self):
        m = TranslationMemory()
        items = self._memory_filtered()
        self.mem_stats.setText(f"共 {len(m.data)} 条（上限 20000，自动淘汰最旧）")
        _reload_table(self.mem_table, items)

    def _refresh_cache(self):
        pc = PageCache()
        st = pc.stats()
        self.cache_stats.setText(
            f"条目 {st['entries']} · 占用 {st['size_mb']} MB · 累计命中 {st['hits']} 次 · "
            f"上限 {pc.limit_mb} MB")


class ServicePage(QWidget):
    """翻译服务管理页"""

    def __init__(self, service: SakuraManager, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.service = service
        self.service.status_changed.connect(self._on_status)
        self.service.log.connect(self._append_log)
        self._init_ui()
        self._on_status("stopped")

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        head = QVBoxLayout()
        title = QLabel("翻译服务")
        title.setObjectName("title")
        sub = QLabel("sakura 本地翻译引擎（127.0.0.1:8080，占用约 6GB 显存）")
        sub.setObjectName("subtitle")
        head.addWidget(title)
        head.addWidget(sub)
        root.addLayout(head)

        # ---- 状态卡片 ----
        card = QFrame()
        card.setObjectName("card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(18, 16, 18, 16)
        cv.setSpacing(10)
        row = QHBoxLayout()
        row.addWidget(QLabel("当前状态:"))
        self.badge = QLabel()
        self.badge.setObjectName("badge")
        row.addWidget(self.badge)
        self.detail = QLabel("")
        row.addWidget(self.detail)
        row.addStretch(1)
        cv.addLayout(row)

        info = QLabel()
        info.setObjectName("muted")
        info.setText(
            "服务地址: http://127.0.0.1:8080\n"
            "模型: sakura-7B (本地, 离线可用)\n"
            "说明: 翻译前需保持服务运行；关闭本界面时可按设置自动停止。")
        cv.addWidget(info)

        mrow = QHBoxLayout()
        self.hf_offline = QCheckBox("模型离线模式（OCR 等用本地缓存，不联网）")
        self.hf_offline.setChecked(bool(load_settings().get("hf_offline", True)))
        self.hf_offline.toggled.connect(
            lambda v: save_settings({"hf_offline": v}))
        mrow.addWidget(self.hf_offline)
        mrow.addStretch(1)
        cv.addLayout(mrow)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("启动翻译服务")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.service.start)
        self.btn_stop = QPushButton("停止服务")
        self.btn_stop.clicked.connect(self.service.stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        self.stop_on_exit = QCheckBox("退出本界面时自动停止服务")
        self.stop_on_exit.setChecked(bool(load_ui_config().get("stop_on_exit", True)))
        self.stop_on_exit.toggled.connect(
            lambda v: save_ui_config({**load_ui_config(), "stop_on_exit": v}))
        btn_row.addWidget(self.stop_on_exit)
        cv.addLayout(btn_row)
        root.addWidget(card)

        # ---- 日志卡片 ----
        log_card = QFrame()
        log_card.setObjectName("card")
        lv = QVBoxLayout(log_card)
        lv.setContentsMargins(16, 14, 16, 14)
        h = QLabel("服务日志")
        h.setObjectName("h2")
        lv.addWidget(h)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        lv.addWidget(self.log_view)
        root.addWidget(log_card, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._tail_log)
        self._timer.start()
        self._tail_log()

    def _on_status(self, state: str):
        owned = self.service.is_owned()
        if state == "running":
            self.badge.setObjectName("badgeOk")
            self.badge.setText("运行中")
            self.detail.setText("sakura 已就绪" + ("（本界面启动）" if owned else "（外部启动，不可由本界面停止）"))
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(owned)
        elif state == "loading":
            self.badge.setObjectName("badgeWarn")
            self.badge.setText("加载中")
            self.detail.setText("模型加载约 20 秒…")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.badge.setObjectName("badgeErr")
            self.badge.setText("未运行")
            self.detail.setText("")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)

    def _append_log(self, text):
        self.log_view.appendPlainText(text)
        self._scroll_bottom()

    def _tail_log(self):
        try:
            if not os.path.exists(SAKURA_LOG):
                return
            size = os.path.getsize(SAKURA_LOG)
            if size == 0:
                return
            with open(SAKURA_LOG, "r", encoding="utf-8", errors="replace") as f:
                f.seek(max(0, size - 16384))
                text = f.read().strip()
            if text and text != self._last_tail:
                self._last_tail = text
                self.log_view.setPlainText(text)
                self._scroll_bottom()
        except Exception:
            pass

    def _scroll_bottom(self):
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("漫画翻译助手")
        self.resize(1280, 900)
        self.setMinimumSize(1020, 700)
        icon = ICON_PATH if os.path.exists(ICON_PATH) else None
        if icon:
            self.setWindowIcon(QIcon(icon))

        self.service = SakuraManager(self)
        self._dark = is_dark_mode()

        self._init_ui()
        self.service.status_changed.connect(self._on_service_status)
        self._on_service_status(self.service.is_running() and "running" or "stopped")
        # 样式生效后适配按钮宽度（QSS 应用后 sizeHint 才准确）
        QTimer.singleShot(0, self._fit_all_buttons)
        QTimer.singleShot(400, self._ask_start_service)

    def _fit_all_buttons(self):
        for p in self.pages:
            fit_buttons(p)
        fit_buttons(self)

    # ---------------- UI ----------------

    def _init_ui(self):
        central = QWidget()
        central.setObjectName("root")
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.pages = [
            TranslatePage(self.service),
            WorkbenchPage(self.service),
            DictionaryPage(),
            ServicePage(self.service),
        ]
        for p in self.pages:
            self.stack.addWidget(p)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(220)
        v = QVBoxLayout(side)
        v.setContentsMargins(14, 18, 14, 14)
        v.setSpacing(6)

        # 标题区
        title_row = QHBoxLayout()
        logo = QLabel("漫")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(38, 38)
        logo.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #14b8a6, stop:1 #0d9488);"
            "color: white; font-size: 18px; font-weight: 700; border-radius: 12px;")
        title_box = QVBoxLayout()
        t1 = QLabel("漫画翻译助手")
        t1.setStyleSheet("font-size: 15px; font-weight: 700;")
        t2 = QLabel("v2.1 统一界面")
        t2.setObjectName("muted")
        title_box.addWidget(t1)
        title_box.addWidget(t2)
        title_row.addWidget(logo)
        title_row.addLayout(title_box)
        title_row.addStretch(1)
        v.addLayout(title_row)
        v.addSpacing(10)

        # 导航
        self.nav_btns = []
        for i, name in enumerate(PAGE_NAMES):
            b = QPushButton(name)
            b.setObjectName("navItem")
            b.setProperty("active", "true" if i == 0 else "false")
            b.clicked.connect(lambda _=False, idx=i: self._switch_page(idx))
            self.nav_btns.append(b)
            v.addWidget(b)

        v.addStretch(1)

        # 服务状态（侧边栏底部）
        self.svc_badge = QLabel()
        self.svc_badge.setObjectName("badge")
        svc_row = QHBoxLayout()
        svc_row.addWidget(self.svc_badge)
        svc_row.addStretch(1)
        v.addLayout(svc_row)

        # 主题切换
        self.btn_theme = QPushButton("暗色模式" if not self._dark else "亮色模式")
        self.btn_theme.setObjectName("ghost")
        self.btn_theme.clicked.connect(self._toggle_theme)
        v.addWidget(self.btn_theme)
        return side

    # ---------------- 行为 ----------------

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, b in enumerate(self.nav_btns):
            b.setProperty("active", "true" if i == idx else "false")
            b.style().unpolish(b)
            b.style().polish(b)

    def _toggle_theme(self):
        self._dark = not self._dark
        apply_theme(QApplication.instance(), self._dark)
        self.btn_theme.setText("暗色模式" if not self._dark else "亮色模式")

    def _on_service_status(self, state: str):
        if state == "running":
            self.svc_badge.setObjectName("badgeOk")
            self.svc_badge.setText("服务运行中")
        elif state == "loading":
            self.svc_badge.setObjectName("badgeWarn")
            self.svc_badge.setText("加载中…")
        else:
            self.svc_badge.setObjectName("badgeErr")
            self.svc_badge.setText("服务未运行")
        self.svc_badge.style().unpolish(self.svc_badge)
        self.svc_badge.style().polish(self.svc_badge)

    def _ask_start_service(self):
        """服务未运行时静默自动启动（不弹询问框，避免打扰）"""
        if not self.service.is_running():
            self.service.start()

    def closeEvent(self, e):
        if self.service.is_owned():
            if load_ui_config().get("stop_on_exit", True):
                self.service.stop()
        # 外部启动的服务保持运行，不再弹提示
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    apply_theme(app, is_dark_mode())
    win = MainApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
