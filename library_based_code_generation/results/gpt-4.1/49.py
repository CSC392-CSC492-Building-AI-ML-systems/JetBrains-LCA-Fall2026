```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import GraphItem, GraphicsLayoutWidget, TextItem

class CustomGraphItem(GraphItem):
    def __init__(self):
        super().__init__()
        self.textItems = []
        self.nodeClicked = None
        self.dragging = False
        self.dragIndex = None
        self.offset = None
        self.setAcceptHoverEvents(True)
        self.setFlag(self.ItemIsFocusable)
        self.setFlag(self.ItemIsSelectable)
        self.setFlag(self.ItemSendsScenePositionChanges)
        self.setZValue(1)
        self.symbolBrush = None
        self.symbolPen = None

    def setData(self, pos, adj, labels, pen=None, symbolBrush=None, symbolPen=None):
        self.pos = np.array(pos)
        self.adj = np.array(adj)
        self.labels = labels
        self.symbolBrush = symbolBrush
        self.symbolPen = symbolPen
        self.updateGraph()

    def updateGraph(self):
        self.setData_(pos=self.pos, adj=self.adj, pen=self.symbolPen, symbolBrush=self.symbolBrush, symbolPen=self.symbolPen, size=20, pxMode=True)
        self.updateLabels()

    def updateLabels(self):
        # Remove old labels
        for item in self.textItems:
            self.scene().removeItem(item)
        self.textItems = []
        for i, (x, y) in enumerate(self.pos):
            text = TextItem(self.labels[i], anchor=(0.5, -1.0), color=(0,0,0))
            text.setPos(x, y)
            self.textItems.append(text)
            if self.scene() is not None:
                self.scene().addItem(text)

    def mousePressEvent(self, event):
        pos = event.pos()
        pts = self.pos
        distances = np.linalg.norm(pts - np.array([pos.x(), pos.y()]), axis=1)
        idx = np.argmin(distances)
        if distances[idx] < 20:  # threshold for clicking a node
            self.nodeClicked = idx
            self.dragging = True
            self.dragIndex = idx
            self.offset = self.pos[idx] - np.array([pos.x(), pos.y()])
            print(f"Node {self.labels[idx]} clicked at {self.pos[idx]}")
            event.accept()
        else:
            self.nodeClicked = None
            self.dragging = False
            self.dragIndex = None
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging and self.dragIndex is not None:
            pos = event.pos()
            new_pos = np.array([pos.x(), pos.y()]) + self.offset
            self.pos[self.dragIndex] = new_pos
            self.updateGraph()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.dragIndex = None
        self.offset = None
        super().mouseReleaseEvent(event)

    def setData_(self, **kwds):
        # Call the original GraphItem.setData
        super().setData(**kwds)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = GraphicsLayoutWidget(show=True, title="pyqtgraph example: CustomGraphItem")
    win.resize(800,600)
    view = win.addViewBox()
    view.setAspectLocked()

    # Example data
    pos = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.5, 1.5]
    ]) * 200

    adj = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [2, 4],
        [3, 4]
    ])

    labels = ['A', 'B', 'C', 'D', 'E']

    pen = pg.mkPen(color=(0, 0, 255), width=2, style=QtCore.Qt.DashLine)
    symbolBrush = [pg.mkBrush(255, 0, 0, 150) for _ in range(len(pos))]
    symbolPen = [pg.mkPen(0, 0, 0, 200) for _ in range(len(pos))]

    graph = CustomGraphItem()
    graph.setData(pos=pos, adj=adj, labels=labels, pen=pen, symbolBrush=symbolBrush, symbolPen=symbolPen)
    view.addItem(graph)
    view.autoRange()

    sys.exit(app.exec_())
```