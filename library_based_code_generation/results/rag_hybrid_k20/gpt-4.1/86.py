```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg

class ScatterDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Scatter Plot Features Demo")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout)

        # Store references to scatter plots and highlighted spots
        self.scatter_items = []
        self.highlighted_spots = []

        # 1. All spots identical and transform-invariant
        p1 = self.graphics_layout.addPlot(title="Identical, Transform-Invariant Spots")
        x1 = np.random.normal(size=100)
        y1 = np.random.normal(size=100)
        spots1 = [{'pos': (x, y)} for x, y in zip(x1, y1)]
        scatter1 = pg.ScatterPlotItem(
            spots=spots1,
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            pxMode=True  # transform-invariant
        )
        scatter1.sigClicked.connect(lambda plot, pts: self.onClicked(plot, pts, 0))
        p1.addItem(scatter1)
        self.scatter_items.append(scatter1)
        self.highlighted_spots.append([])

        # 2. Spots transform-invariant, but not identical
        self.graphics_layout.nextRow()
        p2 = self.graphics_layout.addPlot(title="Non-Identical, Transform-Invariant Spots")
        x2 = np.random.normal(size=100)
        y2 = np.random.normal(size=100)
        colors = [pg.intColor(i, hues=100, alpha=200) for i in range(100)]
        sizes = np.random.uniform(5, 20, size=100)
        spots2 = [{'pos': (x, y), 'brush': c, 'size': s} for x, y, c, s in zip(x2, y2, colors, sizes)]
        scatter2 = pg.ScatterPlotItem(
            spots=spots2,
            pxMode=True  # transform-invariant
        )
        scatter2.sigClicked.connect(lambda plot, pts: self.onClicked(plot, pts, 1))
        p2.addItem(scatter2)
        self.scatter_items.append(scatter2)
        self.highlighted_spots.append([])

        # 3. Spots not transform-invariant and not identical
        p3 = self.graphics_layout.addPlot(title="Non-Identical, Not Transform-Invariant Spots")
        x3 = np.random.normal(size=100)
        y3 = np.random.normal(size=100)
        colors3 = [pg.intColor(i, hues=100, alpha=200) for i in range(100)]
        sizes3 = np.random.uniform(0.05, 0.2, size=100)
        spots3 = [{'pos': (x, y), 'brush': c, 'size': s} for x, y, c, s in zip(x3, y3, colors3, sizes3)]
        scatter3 = pg.ScatterPlotItem(
            spots=spots3,
            pxMode=False  # not transform-invariant
        )
        scatter3.sigClicked.connect(lambda plot, pts: self.onClicked(plot, pts, 2))
        p3.addItem(scatter3)
        self.scatter_items.append(scatter3)
        self.highlighted_spots.append([])

        # 4. Performance test: large scatter plot
        self.graphics_layout.nextRow()
        p4 = self.graphics_layout.addPlot(title="Performance: Large Scatter Plot (10,000 pts)")
        x4 = np.random.normal(size=10000)
        y4 = np.random.normal(size=10000)
        scatter4 = pg.ScatterPlotItem(
            x=x4, y=y4,
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 80),
            pxMode=True
        )
        scatter4.sigClicked.connect(lambda plot, pts: self.onClicked(plot, pts, 3))
        p4.addItem(scatter4)
        self.scatter_items.append(scatter4)
        self.highlighted_spots.append([])

    def onClicked(self, plot, points, idx):
        # Remove previous highlights
        for spot in self.highlighted_spots[idx]:
            plot.removeItem(spot)
        self.highlighted_spots[idx] = []

        # Highlight clicked points
        for pt in points:
            pos = pt.pos()
            highlight = pg.ScatterPlotItem(
                [pos.x()], [pos.y()],
                size=pt.size() * 1.5 if hasattr(pt, 'size') else 15,
                brush=pg.mkBrush(255, 255, 0, 200),
                pen=pg.mkPen('k', width=2),
                pxMode=pt.pxMode()
            )
            plot.addItem(highlight)
            self.highlighted_spots[idx].append(highlight)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterDemo()
    win.show()
    sys.exit(app.exec_())
```