```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg

class ScatterDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Scatter Plot Demo")
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
        scatter1.sigClicked.connect(self.on_scatter_clicked)
        p1.addItem(scatter1)
        self.scatter_items.append(scatter1)

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
        scatter2.sigClicked.connect(self.on_scatter_clicked)
        p2.addItem(scatter2)
        self.scatter_items.append(scatter2)

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
        scatter3.sigClicked.connect(self.on_scatter_clicked)
        p3.addItem(scatter3)
        self.scatter_items.append(scatter3)

        # 4. Performance test: large scatter plot
        self.graphics_layout.nextRow()
        p4 = self.graphics_layout.addPlot(title="Large Scatter Plot (Performance Test)")
        n_points = 10000
        x4 = np.random.normal(size=n_points)
        y4 = np.random.normal(size=n_points)
        spots4 = [{'pos': (x, y)} for x, y in zip(x4, y4)]
        scatter4 = pg.ScatterPlotItem(
            spots=spots4,
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 80),
            pxMode=True
        )
        scatter4.sigClicked.connect(self.on_scatter_clicked)
        p4.addItem(scatter4)
        self.scatter_items.append(scatter4)

    def on_scatter_clicked(self, scatter, points):
        # Remove previous highlights
        for item in self.highlighted_spots:
            scatter = item['scatter']
            point = item['point']
            # Restore original brush
            orig_brush = item['orig_brush']
            point.setBrush(orig_brush)
        self.highlighted_spots.clear()

        # Highlight clicked points
        for point in points:
            orig_brush = point.brush()
            point.setBrush(pg.mkBrush('y'))
            self.highlighted_spots.append({'scatter': scatter, 'point': point, 'orig_brush': orig_brush})

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterDemo()
    win.show()
    sys.exit(app.exec_())
```