"""文本框手动调整器（PPT 式: 拖拽移动/角点缩放）

用法:
    python adjust_gui.py [状态目录或json]
"""
import glob
import json
import os
import sys

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QGraphicsItem, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
    QSpinBox, QLineEdit, QPlainTextEdit, QFileDialog, QMessageBox,
)

HANDLE_SIZE = 10


class Handle(QGraphicsRectItem):
    """角点缩放手柄（文本框的子项，跟随移动；手动计算缩放，不用 ItemIsMovable）"""
    def __init__(self, corner, parent_rect):
        super().__init__(0, 0, HANDLE_SIZE, HANDLE_SIZE)
        self.corner = corner  # 'tl','tr','bl','br','l','r','t','b'
        self.parent_rect = parent_rect
        self.setBrush(QBrush(QColor(0, 120, 215)))
        self.setPen(QPen(QColor(255, 255, 255)))
        self.setZValue(20)
        self.setCursor(Qt.SizeAllCursor)
        self._pressed = False
        self._start_scene = None
        self._start_rect = None

    def mousePressEvent(self, e):
        self._pressed = True
        self._start_scene = e.scenePos()
        self._start_rect = QRectF(self.parent_rect.rect())
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._pressed or self._start_rect is None:
            return
        delta = e.scenePos() - self._start_scene
        rect = QRectF(self._start_rect)
        c = self.corner
        if c == 'tl':
            rect.setLeft(self._start_rect.left() + delta.x())
            rect.setTop(self._start_rect.top() + delta.y())
        elif c == 'tr':
            rect.setRight(self._start_rect.right() + delta.x())
            rect.setTop(self._start_rect.top() + delta.y())
        elif c == 'bl':
            rect.setLeft(self._start_rect.left() + delta.x())
            rect.setBottom(self._start_rect.bottom() + delta.y())
        elif c == 'br':
            rect.setRight(self._start_rect.right() + delta.x())
            rect.setBottom(self._start_rect.bottom() + delta.y())
        elif c == 'l':
            rect.setLeft(self._start_rect.left() + delta.x())
        elif c == 'r':
            rect.setRight(self._start_rect.right() + delta.x())
        elif c == 't':
            rect.setTop(self._start_rect.top() + delta.y())
        elif c == 'b':
            rect.setBottom(self._start_rect.bottom() + delta.y())
        if rect.width() > 8 and rect.height() > 8:
            self.parent_rect.setRect(rect)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._pressed = False
        if self.parent_rect is not None:
            self.parent_rect._after_manual_change()
        e.accept()


class TextBox(QGraphicsRectItem):
    """可移动文本框（带角点手柄）"""
    def __init__(self, rect: QRectF, index: int, text: str, translation: str):
        super().__init__(rect)
        self.index = index
        self.text = text
        self.translation = translation
        self.text_height = None
        self.angle = None
        self.fg = None
        self.bg = None
        self.on_geometry_changed = None  # 拖动/缩放后的回调（workbench 用于字号自动适配）
        self.setPen(QPen(QColor(0, 120, 215), 2))
        self.setBrush(QBrush(QColor(0, 120, 215, 40)))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(10)
        self.handles = []
        self.preview_item = None
        self._make_handles()

    def _after_manual_change(self):
        if self.on_geometry_changed is not None:
            try:
                self.on_geometry_changed(self)
            except Exception:
                pass

    def itemChange(self, change, value):
        res = super().itemChange(change, value)
        if change == QGraphicsItem.ItemPositionHasChanged and self.on_geometry_changed is not None:
            self._after_manual_change()
        return res

    def _make_handles(self):
        for h in self.handles:
            if h.scene():
                h.scene().removeItem(h)
        self.handles = []
        r = self.rect()
        corners = {
            'tl': QPointF(r.left(), r.top()), 'tr': QPointF(r.right(), r.top()),
            'bl': QPointF(r.left(), r.bottom()), 'br': QPointF(r.right(), r.bottom()),
            'l': QPointF(r.left(), r.center().y()), 'r': QPointF(r.right(), r.center().y()),
            't': QPointF(r.center().x(), r.top()), 'b': QPointF(r.center().x(), r.bottom()),
        }
        for name, pos in corners.items():
            h = Handle(name, self)
            h.setParentItem(self)
            h.setPos(pos.x() - HANDLE_SIZE / 2, pos.y() - HANDLE_SIZE / 2)
            self.handles.append(h)

    def setRect(self, rect):
        super().setRect(rect)
        self._sync_handles()
        if self.preview_item is not None and self.preview_item.scene() is not None:
            pm = self.preview_item.pixmap()
            if not pm.isNull():
                self.preview_item.setPos(
                    self.rect().center().x() - pm.width() / 2,
                    self.rect().center().y() - pm.height() / 2)

    def _sync_handles(self):
        r = self.rect()
        positions = {
            'tl': QPointF(r.left(), r.top()), 'tr': QPointF(r.right(), r.top()),
            'bl': QPointF(r.left(), r.bottom()), 'br': QPointF(r.right(), r.bottom()),
            'l': QPointF(r.left(), r.center().y()), 'r': QPointF(r.right(), r.center().y()),
            't': QPointF(r.center().x(), r.top()), 'b': QPointF(r.center().x(), r.bottom()),
        }
        for h in self.handles:
            p = positions[h.corner]
            h.setPos(p.x() - HANDLE_SIZE / 2, p.y() - HANDLE_SIZE / 2)

    def set_visible_handles(self, visible: bool):
        for h in self.handles:
            h.setVisible(visible)
            h.setFlag(QGraphicsItem.ItemIsMovable, visible)


class View(QGraphicsView):
    box_selected = Signal(int)
    box_moved = Signal(int)
    box_delete_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self._zoom *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        item = self.itemAt(e.position().toPoint())
        while item is not None and not isinstance(item, TextBox):
            item = item.parentItem()
        if isinstance(item, TextBox):
            self.box_selected.emit(item.index)
            item.set_visible_handles(True)
            item.setZValue(30)
        else:
            for box in self._all_boxes():
                box.set_visible_handles(False)
                box.setZValue(10)
            self.box_selected.emit(-1)  # 点击空白处清除当前选中

    def _all_boxes(self):
        boxes = []
        for item in self.scene().items():
            if isinstance(item, TextBox):
                boxes.append(item)
        return boxes

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            # 只发"请求删除"信号（不带索引），由宿主用其维护的选中索引删除，
            # 避免 isSelected() 在点击预览/手柄时不可靠导致误删
            if self._all_boxes():
                self.box_delete_requested.emit(-1)
            return
        super().keyPressEvent(e)


class AdjustWindow(QMainWindow):
    def __init__(self, start_path=None):
        super().__init__()
        self.setWindowTitle("文本框手动调整器")
        self.resize(1100, 800)
        self.jsons = []
        self.idx = 0
        self.regions = []  # TextBox items
        self._selected_idx = -1
        self.current_json = None
        self._init_ui()
        self._load_dir(start_path)

    def _init_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.page_label = QLabel("未打开状态")
        btn_open = QPushButton("打开状态目录…")
        btn_open.clicked.connect(self._pick_dir)
        btn_prev = QPushButton("◀ 上一页")
        btn_prev.clicked.connect(lambda: self._goto(self.idx - 1))
        btn_next = QPushButton("下一页 ▶")
        btn_next.clicked.connect(lambda: self._goto(self.idx + 1))
        btn_save = QPushButton("保存调整")
        btn_save.clicked.connect(self._save)
        btn_rerender = QPushButton("重新嵌字")
        btn_rerender.clicked.connect(self._rerender)
        top.addWidget(btn_open)
        top.addWidget(btn_prev)
        top.addWidget(btn_next)
        top.addWidget(self.page_label, 1)
        top.addWidget(btn_save)
        top.addWidget(btn_rerender)
        root.addLayout(top)

        # 视图
        self.scene = QGraphicsScene()
        self.view = View()
        self.view.setScene(self.scene)
        self.view.box_selected.connect(self._on_box_selected)
        root.addWidget(self.view, 1)

        # 底部信息
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("字号:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 120)
        self.font_spin.valueChanged.connect(self._on_font_change)
        bottom.addWidget(self.font_spin)
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(80)
        bottom.addWidget(self.info, 1)
        hint = QLabel("操作: 拖动框=移动  拖动蓝色角点=缩放  滚轮=缩放视图\n调整后点「保存调整」→「重新嵌字」")
        hint.setStyleSheet("color:#888;")
        bottom.addWidget(hint)
        root.addLayout(bottom)

        self.setCentralWidget(central)
        self._boxes = []

    # ---------- 载入 ----------

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择调整状态目录")
        if d:
            self._load_dir(d)

    def _load_dir(self, path):
        if not path:
            return
        if os.path.isdir(path):
            self.jsons = sorted(glob.glob(os.path.join(path, "*.json")))
            if not self.jsons:
                QMessageBox.warning(self, "提示", "目录中没有 .json 状态文件")
                return
            self.idx = 0
        elif os.path.isfile(path) and path.endswith(".json"):
            self.jsons = [path]
            self.idx = 0
        else:
            return
        self._load_page()

    def _load_page(self):
        self.scene.clear()
        self._boxes = []
        self._selected_idx = -1
        self.current_json = self.jsons[self.idx]
        base = os.path.splitext(self.current_json)[0]

        inpainted = base + "_inpainted.png"
        if not os.path.exists(inpainted):
            QMessageBox.warning(self, "错误", f"缺少擦字图: {inpainted}")
            return
        self.pixmap = QPixmap(inpainted)
        if self.pixmap.isNull():
            QMessageBox.warning(self, "错误", "图片加载失败")
            return
        self.scene.addPixmap(self.pixmap)

        with open(self.current_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, d in enumerate(data.get("regions", [])):
            xs = [p[0] for p in d["lines"]]
            ys = [p[1] for p in d["lines"]]
            rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            box = TextBox(rect, i, d.get("text", ""), d.get("translation", ""))
            self.scene.addItem(box)
            box.set_visible_handles(False)
            self._boxes.append(box)
        self.scene.setSceneRect(0, 0, self.pixmap.width(), self.pixmap.height())
        self.page_label.setText(f"{os.path.basename(self.current_json)}  ({self.idx + 1}/{len(self.jsons)})  文本框: {len(self._boxes)}")

    # ---------- 交互 ----------

    def _goto(self, i):
        if 0 <= i < len(self.jsons):
            self.idx = i
            self._load_page()

    def _on_box_selected(self, idx):
        self._selected_idx = idx
        if idx < 0 or idx >= len(self._boxes):
            return
        box = self._boxes[idx]
        self.font_spin.blockSignals(True)
        self.font_spin.setValue(box.text_height or 22)
        self.font_spin.blockSignals(False)
        self.info.setPlainText(f"原文: {box.text}\n译文: {box.translation}")

    def _on_font_change(self, v):
        # 记录字号到当前选中框（保存时写入 JSON）
        if 0 <= self._selected_idx < len(self._boxes):
            self._boxes[self._selected_idx].text_height = v

    # ---------- 保存 ----------

    def _save(self):
        with open(self.current_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, box in enumerate(self._boxes):
            r = box.rect()
            d = data["regions"][i]
            sx, sy = box.pos().x(), box.pos().y()  # 拖动改的是 pos，需叠加
            pts = [[sx + r.left(), sy + r.top()], [sx + r.right(), sy + r.top()],
                   [sx + r.right(), sy + r.bottom()], [sx + r.left(), sy + r.bottom()]]
            d["lines"] = pts
            if box.text_height:
                d["font_size"] = box.text_height
        with open(self.current_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        QMessageBox.information(self, "完成", f"已保存: {os.path.basename(self.current_json)}")

    def _rerender(self):
        self._save()
        import subprocess
        out = os.path.splitext(self.current_json)[0] + "_rerender.png"
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "re_render.py"),
                        self.current_json, out])
        QMessageBox.information(self, "完成", f"已重新嵌字:\n{out}")


def main():
    app = QApplication(sys.argv)
    start = sys.argv[1] if len(sys.argv) > 1 else None
    win = AdjustWindow(start)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
