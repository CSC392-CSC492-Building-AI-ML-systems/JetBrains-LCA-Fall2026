```python
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphItem, GraphicsLayoutWidget

class CustomGraphItem(GraphItem):
    def __init__(self):
        super().__init__()
        self.dragged_node = None
        self.offset = None
        self.symbolBrush = []
        self.symbolPen = []
        self.textItems = []
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    def setData(self, **kwds):
        self.pos = kwds.get('pos', None)
        self.adj = kwds.get('adj', None)
        self.symbols = kwds.get('symbols', None)
        self.symbolBrush = kwds.get('symbolBrush', [pg.mkBrush('w')]*len(self.pos))
        self.symbolPen = kwds.get('symbolPen', [pg.mkPen('k')]*len(self.pos))
        self.labels = kwds.get('labels', ['']*len(self.pos))
        self.linePen = kwds.get('linePen', pg.mkPen('b', width=2))
        self.updateGraph()

    def updateGraph(self):
        super().setData(
            pos=self.pos,
            adj=self.adj,
            symbol=self.symbols,
            symbolBrush=self.symbolBrush,
            symbolPen=self.symbolPen,
            pxMode=True,
            pen=self.linePen,
            size=30
        )
        self.updateLabels()

    def updateLabels(self):
        # Remove old labels
        for item in self.textItems:
            self.scene().removeItem(item)
        self.textItems = []
        for i, (x, y) in enumerate(self.pos):
            text = pg.TextItem(self.labels[i], anchor=(0.5, -1.0), color='k')
            text.setParentItem(self)
            text.setPos(x, y)
            self.textItems.append(text)

    def mousePressEvent(self, event):
        pos = event.pos()
        pts = self.scatter.pointsAt(event.pos())
        if pts:
            idx = pts[0].data()
            self.dragged_node = idx
            node_pos = self.pos[idx]
            self.offset = np.array([pos.x(), pos.y()]) - node_pos
            print(f"Node {self.labels[idx]} clicked at {node_pos}")
            event.accept()
        else:
            self.dragged_node = None
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragged_node is not None:
            pos = event.pos()
            new_pos = np.array([pos.x(), pos.y()]) - self.offset
            self.pos[self.dragged_node] = new_pos
            self.updateGraph()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragged_node = None
        self.offset = None
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
        [2, 3],
        [3, 4],
        [1, 3]
    ])
    labels = ['A', 'B', 'C', 'D', 'E']
    symbols = ['o', 't', 's', 'd', 'h']
    symbolBrush = [pg.mkBrush(c) for c in ['r', 'g', 'b', 'y', 'm']]
    symbolPen = [pg.mkPen('k', width=2) for _ in range(len(pos))]
    linePen = pg.mkPen((100, 100, 255), width=3, style=QtCore.Qt.DashLine)

    graph.setData(
        pos=pos,
        adj=adj,
        symbols=symbols,
        symbolBrush=symbolBrush,
        symbolPen=symbolPen,
        labels=labels,
        linePen=linePen
    )

    view.addItem(graph)
    view.autoRange()
    win.show()
    sys.exit(app.exec_())
```