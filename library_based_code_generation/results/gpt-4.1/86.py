```python
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

class ScatterPlotDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Scatter Plot Demo")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout)

        self.scatter_plots = []
        self.selected_points = []

        self.init_plots()

    def init_plots(self):
        # Plot 1: All spots identical and transform-invariant
        p1 = self.graphics_layout.addPlot(title="Identical, Transform-Invariant Spots")
        x1 = np.random.normal(size=100)
        y1 = np.random.normal(size=100)
        spots1 = [{'pos': (x, y)} for x, y in zip(x1, y1)]
        sp1 = pg.ScatterPlotItem(
            spots=spots1,
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            pxMode=True
        )
        sp1.sigClicked.connect(lambda plot, pts, idx=0: self.on_point_clicked(idx, pts))
        p1.addItem(sp1)
        self.scatter_plots.append(sp1)
        self.selected_points.append([])

        # Plot 2: Transform-invariant, but not identical (vary color/size)
        self.graphics_layout.nextRow()
        p2 = self.graphics_layout.addPlot(title="Varied, Transform-Invariant Spots")
        x2 = np.random.normal(size=100)
        y2 = np.random.normal(size=100)
        colors = [pg.intColor(i, hues=100, alpha=150) for i in range(100)]
        sizes = np.random.uniform(8, 18, size=100)
        spots2 = [{'pos': (x, y), 'brush': c, 'size': s} for x, y, c, s in zip(x2, y2, colors, sizes)]
        sp2 = pg.ScatterPlotItem(
            spots=spots2,
            pen=pg.mkPen(None),
            pxMode=True
        )
        sp2.sigClicked.connect(lambda plot, pts, idx=1: self.on_point_clicked(idx, pts))
        p2.addItem(sp2)
        self.scatter_plots.append(sp2)
        self.selected_points.append([])

        # Plot 3: Not transform-invariant and not identical (use pxMode=False, shapes)
        p3 = self.graphics_layout.addPlot(title="Varied, Not Transform-Invariant Spots")
        x3 = np.random.normal(size=100)
        y3 = np.random.normal(size=100)
        symbols = ['o', 't', 's', 'd', '+', 'x']
        spots3 = [{
            'pos': (x, y),
            'brush': pg.intColor(i, hues=100, alpha=200),
            'size': np.random.uniform(10, 20),
            'symbol': symbols[i % len(symbols)]
        } for i, (x, y) in enumerate(zip(x3, y3))]
        sp3 = pg.ScatterPlotItem(
            spots=spots3,
            pen=pg.mkPen('k', width=1),
            pxMode=False
        )
        sp3.sigClicked.connect(lambda plot, pts, idx=2: self.on_point_clicked(idx, pts))
        p3.addItem(sp3)
        self.scatter_plots.append(sp3)
        self.selected_points.append([])

        # Plot 4: Performance test with large scatterplot
        self.graphics_layout.nextRow()
        p4 = self.graphics_layout.addPlot(title="Large Scatter Plot (Performance Test)")
        n = 10000
        x4 = np.random.normal(size=n)
        y4 = np.random.normal(size=n)
        sp4 = pg.ScatterPlotItem(
            x=x4,
            y=y4,
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 80),
            pxMode=True
        )
        sp4.sigClicked.connect(lambda plot, pts, idx=3: self.on_point_clicked(idx, pts))
        p4.addItem(sp4)
        self.scatter_plots.append(sp4)
        self.selected_points.append([])

    def on_point_clicked(self, plot_idx, points):
        sp = self.scatter_plots[plot_idx]
        # Unhighlight previously selected points
        for pt in self.selected_points[plot_idx]:
            pt.resetPen()
        # Highlight newly selected points
        for pt in points:
            pt.setPen('y', width=2)
        self.selected_points[plot_idx] = points

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterPlotDemo()
    win.show()
    sys.exit(app.exec_())
```