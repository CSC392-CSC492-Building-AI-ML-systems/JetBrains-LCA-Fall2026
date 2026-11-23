```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg

class ScatterDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Scatter Plot Features Demo")
        self.resize(1200, 900)
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
        s1 = pg.ScatterPlotItem(
            spots=spots1,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=15,
            symbol='o'
        )
        s1.sigClicked.connect(self.on_spot_clicked)
        p1.addItem(s1)
        self.scatter_items.append(s1)

        # 2. Transform-invariant, but not identical (vary color/size/symbol)
        p2 = self.graphics_layout.addPlot(title="Varied, Transform-Invariant Spots")
        self.graphics_layout.nextRow()
        x2 = np.random.normal(size=100)
        y2 = np.random.normal(size=100)
        colors = [pg.intColor(i, hues=10, values=5, maxValue=255, minValue=100) for i in range(100)]
        sizes = np.random.uniform(10, 20, size=100)
        symbols = np.random.choice(['o', 't', 's', 'd', '+', 'x'], size=100)
        spots2 = [{'pos': (x, y), 'brush': c, 'size': sz, 'symbol': sym}
                  for x, y, c, sz, sym in zip(x2, y2, colors, sizes, symbols)]
        s2 = pg.ScatterPlotItem(spots=spots2, pen=pg.mkPen(None))
        s2.sigClicked.connect(self.on_spot_clicked)
        p2.addItem(s2)
        self.scatter_items.append(s2)

        # 3. Not transform-invariant and not identical (custom shapes, rotated, etc.)
        p3 = self.graphics_layout.addPlot(title="Non-Transform-Invariant, Non-Identical Spots")
        x3 = np.random.normal(size=50)
        y3 = np.random.normal(size=50)
        spots3 = []
        for i in range(50):
            angle = np.random.uniform(0, 360)
            size = np.random.uniform(15, 30)
            color = pg.intColor(i, hues=50, maxValue=255)
            # Use custom symbol: triangle rotated by angle
            spots3.append({
                'pos': (x3[i], y3[i]),
                'brush': color,
                'size': size,
                'symbol': 't',
                'angle': angle
            })
        s3 = pg.ScatterPlotItem(spots=spots3, pen=pg.mkPen('k', width=1))
        s3.sigClicked.connect(self.on_spot_clicked)
        p3.addItem(s3)
        self.scatter_items.append(s3)

        # 4. Performance test: large scatter plot
        p4 = self.graphics_layout.addPlot(title="Large Scatter Plot (Performance Test)")
        x4 = np.random.normal(size=10000)
        y4 = np.random.normal(size=10000)
        s4 = pg.ScatterPlotItem(
            x=x4, y=y4,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 200, 0, 80),
            size=5,
            symbol='o'
        )
        s4.sigClicked.connect(self.on_spot_clicked)
        p4.addItem(s4)
        self.scatter_items.append(s4)

    def on_spot_clicked(self, plot, points):
        # Remove previous highlights
        for item in self.highlighted_spots:
            plot.removeItem(item)
        self.highlighted_spots.clear()
        # Highlight clicked points
        for pt in points:
            pos = pt.pos()
            highlight = pg.ScatterPlotItem(
                x=[pos.x()], y=[pos.y()],
                pen=pg.mkPen('r', width=2),
                brush=pg.mkBrush(255, 0, 0, 120),
                size=pt.size() * 1.5 if hasattr(pt, 'size') else 20,
                symbol=pt.symbol() if hasattr(pt, 'symbol') else 'o'
            )
            plot.addItem(highlight)
            self.highlighted_spots.append(highlight)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterDemo()
    win.show()
    sys.exit(app.exec_())
```