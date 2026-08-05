"""PySide6 theme derived from D:\\tools\\gui-standard\\theme_pyside6.py."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


TOKENS = {
    "light": {
        "bg": "#f9fafb",
        "card": "#ffffff",
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
        "success_bg": "#d1fae5",
        "success_text": "#047857",
        "warning_bg": "#fef3c7",
        "warning_text": "#b45309",
        "danger": "#ef4444",
        "danger_bg": "#fee2e2",
        "danger_text": "#b91c1c",
        "hover": "#f3f4f6",
        "input_bg": "#ffffff",
        "log_bg": "#f9fafb",
        "sidebar": "#ffffff",
        "canvas": "#20252b",
        "shadow": QColor(0, 0, 0, 18),
    },
    "dark": {
        "bg": "#0f172a",
        "card": "#1e293b",
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
        "success_bg": "rgba(16,185,129,0.18)",
        "success_text": "#34d399",
        "warning_bg": "rgba(245,158,11,0.18)",
        "warning_text": "#fbbf24",
        "danger": "#f87171",
        "danger_bg": "rgba(239,68,68,0.18)",
        "danger_text": "#f87171",
        "hover": "#243244",
        "input_bg": "#16213a",
        "log_bg": "#0b1220",
        "sidebar": "#0d1526",
        "canvas": "#080d16",
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
    letter-spacing: 0px;
}}
QMainWindow, QDialog, QWidget#root {{ background: {t["bg"]}; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 20px; font-weight: 700; }}
QLabel#subtitle, QLabel#muted {{ color: {t["text_mut"]}; }}
QLabel#h2 {{ font-size: 14px; font-weight: 600; }}
QLabel#mono {{ font-family: Consolas, "Courier New", monospace; }}
QFrame#card {{
    background: {t["card"]};
    border: 1px solid {t["border_soft"]};
    border-radius: 16px;
}}
QLabel#badge, QLabel#badgeOk, QLabel#badgeWarn, QLabel#badgeErr, QLabel#badgeIdle {{
    border-radius: 11px; padding: 3px 12px; font-size: 12px; font-weight: 600;
}}
QLabel#badgeOk {{ background: {t["success_bg"]}; color: {t["success_text"]}; }}
QLabel#badgeWarn {{ background: {t["warning_bg"]}; color: {t["warning_text"]}; }}
QLabel#badgeErr {{ background: {t["danger_bg"]}; color: {t["danger_text"]}; }}
QLabel#badgeIdle {{ background: {t["hover"]}; color: {t["text_sec"]}; }}
QPushButton {{
    background: {t["card"]}; border: 1px solid {t["border"]}; border-radius: 12px;
    padding: 8px 16px; color: {t["text"]};
}}
QPushButton:hover {{ background: {t["hover"]}; border-color: {t["text_mut"]}; }}
QPushButton:pressed {{ background: {t["hover"]}; }}
QPushButton:disabled {{ color: {t["text_mut"]}; background: {t["card"]}; border-color: {t["border_soft"]}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t["primary"]},stop:1 {t["primary_hover"]});
    border: none; color: {t["primary_text"]}; font-weight: 600;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t["primary_hover"]},stop:1 {t["primary_active"]});
}}
QPushButton#primary:disabled {{ background: {t["border"]}; color: {t["text_mut"]}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {t["text_sec"]}; }}
QPushButton#ghost:hover {{ background: {t["hover"]}; color: {t["text"]}; }}
QPushButton#danger {{ background: {t["danger"]}; border: none; color: #ffffff; font-weight: 600; }}
QWidget#sidebar {{ background: {t["sidebar"]}; border-right: 1px solid {t["border_soft"]}; }}
QPushButton#navItem {{
    background: transparent; border: none; border-radius: 12px; padding: 10px 14px;
    text-align: left; color: {t["text_sec"]}; font-size: 14px;
}}
QPushButton#navItem:hover {{ background: {t["hover"]}; color: {t["text"]}; }}
QPushButton#navItem[active="true"] {{
    background: {t["primary_soft"]}; color: {t["primary_hover"]}; font-weight: 600;
}}
QPushButton#navItem[active="true"]:hover {{ background: {t["primary_soft_hover"]}; }}
QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox {{
    background: {t["input_bg"]}; border: 1px solid {t["border"]}; border-radius: 12px;
    padding: 7px 12px; selection-background-color: {t["primary"]};
    selection-color: {t["primary_text"]};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {t["primary"]}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {t["card"]}; border: 1px solid {t["border"]}; border-radius: 10px;
    padding: 4px; selection-background-color: {t["primary_soft"]};
    selection-color: {t["primary_hover"]}; outline: none;
}}
QPlainTextEdit, QTextEdit {{
    background: {t["log_bg"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px;
    padding: 8px; font-family: Consolas, "Courier New", monospace; font-size: 12px;
}}
QCheckBox, QRadioButton {{ spacing: 8px; color: {t["text"]}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {t["border"]}; border-radius: 5px;
    background: {t["input_bg"]};
}}
QCheckBox::indicator:hover {{ border-color: {t["primary"]}; }}
QCheckBox::indicator:checked {{ background: {t["primary"]}; border-color: {t["primary"]}; }}
QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {t["border"]}; border-radius: 9px;
    background: {t["input_bg"]};
}}
QRadioButton::indicator:checked {{ border: 5px solid {t["primary"]}; }}
QTableWidget {{
    background: {t["card"]}; border: 1px solid {t["border_soft"]}; border-radius: 12px;
    gridline-color: {t["border_soft"]}; selection-background-color: {t["primary_soft"]};
    selection-color: {t["text"]}; outline: none;
}}
QHeaderView::section {{
    background: {t["bg"]}; color: {t["text_sec"]}; border: none;
    border-bottom: 1px solid {t["border_soft"]}; padding: 9px 12px; font-weight: 500;
}}
QTableWidget::item {{ padding: 8px 12px; border-bottom: 1px solid {t["border_soft"]}; }}
QProgressBar {{
    background: {t["hover"]}; border: none; border-radius: 6px; height: 10px;
    text-align: center; color: {t["text_sec"]}; font-size: 11px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t["primary"]},stop:1 {t["primary_hover"]});
    border-radius: 6px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t["text_mut"]}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t["text_mut"]}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QGraphicsView#reviewCanvas {{ background: {t["canvas"]}; border: none; border-radius: 12px; }}
QSplitter::handle {{ background: {t["border_soft"]}; width: 2px; height: 2px; }}
QToolTip {{ background: {t["card"]}; color: {t["text"]}; border: 1px solid {t["border"]}; padding: 5px; }}
QMessageBox {{ background: {t["card"]}; }}
"""


def load_ui_config(config_path: Path) -> dict:
    try:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def save_ui_config(config: dict, config_path: Path) -> None:
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def apply_theme(app, dark: bool, config_path: Path) -> None:
    app.setStyleSheet(build_qss(dark))
    config = load_ui_config(config_path)
    config["dark"] = bool(dark)
    save_ui_config(config, config_path)


def is_dark_mode(config_path: Path) -> bool:
    return bool(load_ui_config(config_path).get("dark", False))


def add_shadow(widget: QWidget, dark: bool = False, blur: int = 14) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, 2)
    effect.setColor(TOKENS["dark" if dark else "light"]["shadow"])
    widget.setGraphicsEffect(effect)
