"""主题设计系统: 亮/暗双主题 QSS（参照 ssstoken 仪表盘风格）

- 主色 teal #14b8a6（暗色提亮 #2dd4bf）
- 圆角: 按钮/输入框 12px, 卡片 16px, 胶囊全圆
- 阴影用 QGraphicsDropShadowEffect 在代码中附加（QSS 不支持 box-shadow）
"""
import json
import os

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

from paths import UI_CONFIG

TOKENS = {
    "light": {
        "bg": "#f9fafb",
        "card": "#ffffff",
        "card_hover": "#f9fafb",
        "border": "#e5e7eb",
        "border_soft": "#f3f4f6",
        "text": "#111827",
        "text_sec": "#4b5563",
        "text_mut": "#9ca3af",
        "primary": "#14b8a6",
        "primary_hover": "#0d9488",
        "primary_active": "#0f766e",
        "primary_soft": "#f0fdfa",
        "primary_soft_hover": "#ccfbf1",
        "primary_text": "#ffffff",
        "success": "#10b981",
        "success_bg": "#d1fae5",
        "success_text": "#047857",
        "warning": "#f59e0b",
        "warning_bg": "#fef3c7",
        "warning_text": "#b45309",
        "danger": "#ef4444",
        "danger_bg": "#fee2e2",
        "danger_text": "#b91c1c",
        "hover": "#f3f4f6",
        "input_bg": "#ffffff",
        "log_bg": "#f9fafb",
        "sidebar": "#ffffff",
        "shadow": QColor(0, 0, 0, 18),
    },
    "dark": {
        "bg": "#0f172a",
        "card": "#1e293b",
        "card_hover": "#243244",
        "border": "#334155",
        "border_soft": "#2b3949",
        "text": "#e2e8f0",
        "text_sec": "#cbd5e1",
        "text_mut": "#94a3b8",
        "primary": "#2dd4bf",
        "primary_hover": "#14b8a6",
        "primary_active": "#0d9488",
        "primary_soft": "rgba(19,78,74,0.45)",
        "primary_soft_hover": "rgba(19,78,74,0.7)",
        "primary_text": "#042f2e",
        "success": "#34d399",
        "success_bg": "rgba(16,185,129,0.18)",
        "success_text": "#34d399",
        "warning": "#fbbf24",
        "warning_bg": "rgba(245,158,11,0.18)",
        "warning_text": "#fbbf24",
        "danger": "#f87171",
        "danger_bg": "rgba(239,68,68,0.18)",
        "danger_text": "#f87171",
        "hover": "#243244",
        "input_bg": "#16213a",
        "log_bg": "#0b1220",
        "sidebar": "#0d1526",
        "shadow": QColor(0, 0, 0, 90),
    },
}


def build_qss(dark: bool = False) -> str:
    t = TOKENS["dark" if dark else "light"]
    return f"""
* {{
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 13px;
    color: {t["text"]};
}}
QMainWindow, QDialog, QWidget#root {{
    background: {t["bg"]};
}}
QLabel {{ background: transparent; }}
QLabel#title {{
    font-size: 20px; font-weight: 600;
}}
QLabel#subtitle {{
    color: {t["text_mut"]};
}}
QLabel#h1 {{ font-size: 16px; font-weight: 600; }}
QLabel#h2 {{ font-size: 14px; font-weight: 600; }}
QLabel#muted {{ color: {t["text_mut"]}; }}
QLabel#mono {{ font-family: Consolas, "Courier New", monospace; }}

/* ---------- 卡片 ---------- */
QFrame#card {{
    background: {t["card"]};
    border: 1px solid {t["border_soft"]};
    border-radius: 16px;
}}

/* ---------- 胶囊 badge ---------- */
QLabel#badge {{
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
}}
QLabel#badgeOk   {{ background: {t["success_bg"]}; color: {t["success_text"]}; }}
QLabel#badgeWarn {{ background: {t["warning_bg"]}; color: {t["warning_text"]}; }}
QLabel#badgeErr  {{ background: {t["danger_bg"]};  color: {t["danger_text"]}; }}
QLabel#badgeIdle {{ background: {t["hover"]}; color: {t["text_sec"]}; }}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {t["card"]};
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 8px 16px;
    color: {t["text"]};
}}
QPushButton:hover {{ background: {t["hover"]}; border-color: {t["text_mut"]}; }}
QPushButton:pressed {{ background: {t["hover"]}; }}
QPushButton:disabled {{ color: {t["text_mut"]}; background: {t["card"]}; border-color: {t["border_soft"]}; }}

QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t["primary"]}, stop:1 {t["primary_hover"]});
    border: none; color: {t["primary_text"]};
    font-weight: 600;
}}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t["primary_hover"]}, stop:1 {t["primary_active"]}); }}
QPushButton#primary:disabled {{ background: {t["border"]}; color: {t["text_mut"]}; }}
QPushButton#primary:focus {{ border: 2px solid {t["primary"]}; }}

QPushButton#ghost {{ background: transparent; border: none; color: {t["text_sec"]}; }}
QPushButton#ghost:hover {{ background: {t["hover"]}; color: {t["text"]}; }}

QPushButton#danger {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t["danger"]}, stop:1 {t["danger"]});
    border: none; color: #ffffff; font-weight: 600;
}}
QPushButton#danger:hover {{ background: {t["danger"]}; }}

/* ---------- 侧边栏导航 ---------- */
QWidget#sidebar {{
    background: {t["sidebar"]};
    border-right: 1px solid {t["border_soft"]};
}}
QPushButton#navItem {{
    background: transparent; border: none;
    border-radius: 12px;
    padding: 10px 14px;
    text-align: left;
    color: {t["text_sec"]};
    font-size: 14px;
}}
QPushButton#navItem:hover {{ background: {t["hover"]}; color: {t["text"]}; }}
QPushButton#navItem[active="true"] {{
    background: {t["primary_soft"]};
    color: {t["primary_hover"]};
    font-weight: 600;
}}
QPushButton#navItem[active="true"]:hover {{ background: {t["primary_soft_hover"]}; }}

/* ---------- 输入控件 ---------- */
QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox {{
    background: {t["input_bg"]};
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 7px 12px;
    selection-background-color: {t["primary"]};
    selection-color: {t["primary_text"]};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {t["primary"]};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {t["card"]};
    border: 1px solid {t["border"]};
    border-radius: 10px;
    padding: 4px;
    selection-background-color: {t["primary_soft"]};
    selection-color: {t["primary_hover"]};
    outline: none;
}}

/* ---------- 多行文本/日志 ---------- */
QPlainTextEdit, QTextEdit {{
    background: {t["log_bg"]};
    border: 1px solid {t["border_soft"]};
    border-radius: 12px;
    padding: 8px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}}
QPlainTextEdit:focus {{ border: 1px solid {t["primary"]}; }}

/* ---------- 复选框 ---------- */
QCheckBox {{
    spacing: 8px;
    color: {t["text"]};
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t["border"]};
    border-radius: 5px;
    background: {t["input_bg"]};
}}
QCheckBox::indicator:hover {{ border-color: {t["primary"]}; }}
QCheckBox::indicator:checked {{
    background: {t["primary"]};
    border-color: {t["primary"]};
}}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {t["hover"]};
    border: none;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: {t["text_sec"]};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t["primary"]}, stop:1 {t["primary_hover"]});
    border-radius: 6px;
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {t["text_mut"]}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {t["text_sec"]}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t["text_mut"]}; border-radius: 5px; min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 7px 16px;
    color: {t["text_sec"]};
    font-size: 13px;
}}
QTabBar::tab:hover {{ background: {t["hover"]}; }}
QTabBar::tab:selected {{
    background: {t["primary_soft"]};
    color: {t["primary_hover"]};
    font-weight: 600;
}}

/* ---------- 下拉菜单 ---------- */
QMenu {{
    background: {t["card"]};
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 6px;
    color: {t["text"]};
}}
QMenu::item {{
    background: transparent;
    border-radius: 8px;
    padding: 7px 14px;
    color: {t["text"]};
}}
QMenu::item:selected {{
    background: {t["primary_soft"]};
    color: {t["primary_hover"]};
}}

/* ---------- 表格 ---------- */
QTableWidget, QTableView {{
    background: {t["card"]};
    border: 1px solid {t["border_soft"]};
    border-radius: 12px;
    color: {t["text"]};
    gridline-color: {t["border_soft"]};
    selection-background-color: {t["primary_soft"]};
    selection-color: {t["primary_hover"]};
}}
QHeaderView::section {{
    background: {t["card"]};
    color: {t["text_sec"]};
    border: none;
    border-bottom: 1px solid {t["border_soft"]};
    padding: 8px 12px;
    font-weight: 600;
}}
QTableWidget::item {{
    padding: 4px 8px;
    color: {t["text"]};
}}
QTableWidget::item:selected {{
    background: {t["primary_soft"]};
    color: {t["primary_hover"]};
}}

/* ---------- 图形视图（暗色下场景背景跟随主题） ---------- */
QGraphicsView {{
    background: {t["bg"]};
    border: 1px solid {t["border_soft"]};
    border-radius: 12px;
}}

/* ---------- 消息框 ---------- */
QMessageBox {{ background: {t["card"]}; }}
QMessageBox QLabel {{ color: {t["text"]}; }}
"""


def load_ui_config() -> dict:
    try:
        with open(UI_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ui_config(cfg: dict):
    try:
        os.makedirs(os.path.dirname(UI_CONFIG), exist_ok=True)
        with open(UI_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def is_dark_mode() -> bool:
    return bool(load_ui_config().get("dark", False))


def apply_theme(app, dark: bool):
    app.setStyleSheet(build_qss(dark))
    cfg = load_ui_config()
    cfg["dark"] = dark
    save_ui_config(cfg)


def add_shadow(widget: QWidget, dark: bool = False, blur: int = 14):
    """给卡片附加轻阴影（QSS 不支持 box-shadow，用效果代替）"""
    e = QGraphicsDropShadowEffect(widget)
    e.setBlurRadius(blur)
    e.setOffset(0, 2)
    e.setColor(TOKENS["dark" if dark else "light"]["shadow"])
    widget.setGraphicsEffect(e)


def fit_buttons(widget):
    """让按钮按文字宽度自适应，避免布局压缩导致文字显示不全"""
    from PySide6.QtWidgets import QPushButton
    for b in widget.findChildren(QPushButton):
        try:
            w = b.sizeHint().width() + 8
            if b.minimumWidth() < w:
                b.setMinimumWidth(w)
        except Exception:
            pass
