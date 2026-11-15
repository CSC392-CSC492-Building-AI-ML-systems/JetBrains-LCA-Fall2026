```python
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
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
        self.edge_styles = None

    def setData(self, pos, adj, size=20, symbol='o', pxMode=True, pen=None, brush=None, edge_pen=None, labels=None):
        self.node_positions = np.array(pos)
        self.adj = np.array(adj)
        self.size = size
        self.symbols = symbol if isinstance(symbol, list) else [symbol]*len(pos)
        self.pxMode = pxMode
        self.pen = pen if pen is not None else pg.mkPen('w', width=2)
        self.brush = brush if brush is not None else pg.mkBrush('b')
        self.edge_pen = edge_pen if edge_pen is not None else pg.mkPen('g', width=1)
        self.labels = labels if labels is not None else [str(i) for i in range(len(pos))]
        self.updateGraph()

    def updateGraph(self):
        self.setData_(pos=self.node_positions, adj=self.adj, size=self.size, symbol=self.symbols,
                     pxMode=self.pxMode, pen=self.pen, brush=self.brush, edgePen=self.edge_pen)
        # Remove old text items
        for item in self.text_items:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.text_items = []
        # Add new text items
        for i, (x, y) in enumerate(self.node_positions):
            text = pg.TextItem(self.labels[i], anchor=(0.5, -1.0), color='w')
            text.setPos(x, y)
            self.text_items.append(text)
            if self.scene() is not None:
                self.scene().addItem(text)

    def mousePressEvent(self, event):
        pos = event.pos()
        pts = self.node_positions
        mouse_point = self.mapToView(pos)
        distances = np.linalg.norm(pts - np.array([mouse_point.x(), mouse_point.y()]), axis=1)
        idx = np.argmin(distances)
        if distances[idx] < self.size * 0.05:
            self.dragging = True
            self.dragged_node = idx
            self.offset = self.node_positions[idx] - np.array([mouse_point.x(), mouse_point.y()])
            print(f"Node {self.labels[idx]} clicked.")
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging and self.dragged_node is not None:
            mouse_point = self.mapToView(event.pos())
            new_pos = np.array([mouse_point.x(), mouse_point.y()]) + self.offset
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

    def setData_(self, **kwds):
        # Call the original GraphItem.setData
        super().setData(**kwds)

    def setViewBox(self, vb):
        super().setViewBox(vb)
        # Add text items to the scene if not already present
        for item in self.text_items:
            if item.scene() is None and self.scene() is not None:
                self.scene().addItem(item)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = GraphicsLayoutWidget(show=True, title="pyqtgraph example: CustomGraphItem")
    win.resize(800, 600)
    view = win.addViewBox()
    view.setAspectLocked()

    # Example graph data
    positions = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1],
        [1.5, 1],
        [2, 0],
    ])
    adj = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 3],
    ])
    labels = ['A', 'B', 'C', 'D', 'E']

    graph = CustomGraphItem()
    graph.setData(
        pos=positions,
        adj=adj,
        size=30,
        symbol=['o', 't', 's', 'd', 'h'],
        pen=pg.mkPen('w', width=2),
        brush=[pg.mkBrush('r'), pg.mkBrush('g'), pg.mkBrush('b'), pg.mkBrush('y'), pg.mkBrush('m')],
        edge_pen=pg.mkPen('c', width=2, style=QtCore.Qt.DashLine),
        labels=labels
    )
    view.addItem(graph)
    win.show()
    sys.exit(app.exec_())
```