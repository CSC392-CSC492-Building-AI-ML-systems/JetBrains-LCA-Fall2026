```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import GraphItem, GraphicsLayoutWidget

class CustomGraphItem(GraphItem):
    def __init__(self):
        super().__init__()
        self.dragging = False
        self.dragged_node = None
        self.offset = None
        self.node_positions = None
        self.symbols = None
        self.labels = []
        self.text_items = []
        self.setAcceptHoverEvents(True)

    def setData(self, pos, adj, pen=None, symbolBrush=None, symbolPen=None, symbols=None, texts=None):
        self.node_positions = pos
        self.adj = adj
        self.symbols = symbols
        self.texts = texts
        self.pen = pen if pen is not None else pg.mkPen('w', width=2)
        self.symbolBrush = symbolBrush if symbolBrush is not None else [pg.mkBrush('b')] * len(pos)
        self.symbolPen = symbolPen if symbolPen is not None else [pg.mkPen('w')] * len(pos)
        self.updateGraph()

    def updateGraph(self):
        self.setData_(
            pos=self.node_positions,
            adj=self.adj,
            pen=self.pen,
            symbolBrush=self.symbolBrush,
            symbolPen=self.symbolPen,
            symbols=self.symbols
        )
        # Remove old text items
        for item in self.text_items:
            self.scene().removeItem(item)
        self.text_items = []
        # Add new text items
        if self.texts is not None:
            for i, txt in enumerate(self.texts):
                text_item = pg.TextItem(txt, anchor=(0.5, -1.0), color='w')
                text_item.setParentItem(self)
                text_item.setPos(*self.node_positions[i])
                self.text_items.append(text_item)

    def mousePressEvent(self, event):
        pos = event.pos()
        pts = self.scatter.pointsAt(event.pos())
        if len(pts) > 0:
            idx = pts[0].data()
            print(f"Node {idx} clicked")
            self.dragging = True
            self.dragged_node = idx
            node_pos = self.node_positions[idx]
            self.offset = node_pos - np.array([pos.x(), pos.y()])
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging and self.dragged_node is not None:
            pos = event.pos()
            new_pos = np.array([pos.x(), pos.y()]) + self.offset
            self.node_positions[self.dragged_node] = new_pos
            self.updateGraph()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.dragged_node = None
            self.offset = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = GraphicsLayoutWidget(show=True, title='pyqtgraph example: CustomGraphItem')
    win.resize(800, 600)
    view = win.addViewBox()
    view.setAspectLocked()
    graph = CustomGraphItem()

    # Example graph data
    pos = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1],
        [1.5, 1],
        [2, 0]
    ])
    adj = np.array([
        [0, 1],
        [1, 2],
        [1, 3],
        [3, 4]
    ])
    symbols = ['o', 't', 's', 'd', '+']
    texts = ['A', 'B', 'C', 'D', 'E']
    pen = [pg.mkPen(color=(255,0,0), width=2), pg.mkPen(color=(0,255,0), width=2),
           pg.mkPen(color=(0,0,255), width=2), pg.mkPen(color=(255,255,0), width=2)]
    symbolBrush = [pg.mkBrush('r'), pg.mkBrush('g'), pg.mkBrush('b'), pg.mkBrush('y'), pg.mkBrush('m')]
    symbolPen = [pg.mkPen('w')] * 5

    graph.setData(pos=pos, adj=adj, pen=pen, symbolBrush=symbolBrush, symbolPen=symbolPen, symbols=symbols, texts=texts)
    view.addItem(graph)
    win.show()
    sys.exit(app.exec_())
```