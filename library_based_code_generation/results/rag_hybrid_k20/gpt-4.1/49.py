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
        self.node_radius = 15
        self.text_items = []
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges)
        self.setZValue(1)
        self.data = None

    def setData(self, **kwds):
        self.data = kwds
        self.updateGraph()

    def updateGraph(self):
        pos = self.data['pos']
        adj = self.data['adj']
        pen = self.data.get('pen', pg.mkPen('w', width=2))
        symbolBrush = self.data.get('symbolBrush', [pg.mkBrush('b')] * len(pos))
        symbolPen = self.data.get('symbolPen', [pg.mkPen('w')] * len(pos))
        size = self.data.get('size', self.node_radius*2)
        labels = self.data.get('labels', [str(i) for i in range(len(pos))])

        self.setData_(pos=pos, adj=adj, pen=pen, symbolBrush=symbolBrush, symbolPen=symbolPen, size=size)
        self.clearLabels()
        for i, p in enumerate(pos):
            text = pg.TextItem(labels[i], anchor=(0.5, -1.0), color='w')
            text.setParentItem(self)
            text.setPos(p[0], p[1])
            self.text_items.append(text)

    def clearLabels(self):
        for t in self.text_items:
            t.setParentItem(None)
            t.scene() and t.scene().removeItem(t)
        self.text_items = []

    def mousePressEvent(self, event):
        pos = event.pos()
        mousePoint = QtCore.QPointF(pos.x(), pos.y())
        node_idx = self.getNodeAt(mousePoint)
        if node_idx is not None:
            self.dragged_node = node_idx
            node_pos = self.data['pos'][node_idx]
            self.offset = mousePoint - QtCore.QPointF(*node_pos)
            print(f"Node {node_idx} clicked")
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragged_node is not None:
            pos = event.pos()
            new_pos = np.array([pos.x(), pos.y()]) - np.array([self.offset.x(), self.offset.y()])
            self.data['pos'][self.dragged_node] = new_pos
            self.updateGraph()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragged_node = None
        self.offset = None
        super().mouseReleaseEvent(event)

    def getNodeAt(self, point):
        pos = self.data['pos']
        for i, p in enumerate(pos):
            dist = np.linalg.norm(np.array([point.x(), point.y()]) - np.array(p))
            if dist <= self.node_radius:
                return i
        return None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = GraphicsLayoutWidget(show=True, title='pyqtgraph example: CustomGraphItem')
    win.setWindowTitle('pyqtgraph example: CustomGraphItem')
    view = win.addViewBox()
    view.setAspectLocked()
    graph = CustomGraphItem()
    view.addItem(graph)

    # Example graph data
    positions = np.array([
        [0, 0],
        [100, 0],
        [50, 86],
        [150, 50],
        [200, 0]
    ], dtype=float)
    adj = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [1, 3],
        [3, 4]
    ])
    labels = ['A', 'B', 'C', 'D', 'E']
    pen = [pg.mkPen(color=(255,0,0), width=2), pg.mkPen(color=(0,255,0), width=2),
           pg.mkPen(color=(0,0,255), width=2), pg.mkPen(color=(255,255,0), width=2),
           pg.mkPen(color=(255,0,255), width=2)]
    symbolBrush = [pg.mkBrush('r'), pg.mkBrush('g'), pg.mkBrush('b'), pg.mkBrush('y'), pg.mkBrush('m')]
    symbolPen = [pg.mkPen('w', width=2)]*5

    graph.setData(
        pos=positions,
        adj=adj,
        pen=pen,
        symbolBrush=symbolBrush,
        symbolPen=symbolPen,
        size=30,
        labels=labels
    )

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
```