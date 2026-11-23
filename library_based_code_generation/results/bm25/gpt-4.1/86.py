```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class ScatterDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Scatter Plot Demo")
        self.resize(1200, 800)
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.layout = QtWidgets.QVBoxLayout(self.central)
        self.graphicsLayout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphicsLayout)

        # Store references to scatter plots and highlighted spots
        self.scatter_items = []
        self.highlighted_spots = []

        # 1. All spots identical and transform-invariant
        p1 = self.graphicsLayout.addPlot(title="Identical, Transform-Invariant Spots")
        x1 = np.random.normal(size=100)
        y1 = np.random.normal(size=100)
        spots1 = [{'pos': (x, y)} for x, y in zip(x1, y1)]
        s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(100, 100, 255, 120), pxMode=True)
        s1.addPoints(spots1)
        s1.sigClicked.connect(lambda plot, pts: self.onClicked(s1, pts))
        p1.addItem(s1)
        self.scatter_items.append(s1)

        # 2. Transform-invariant, but not identical
        self.graphicsLayout.nextRow()
        p2 = self.graphicsLayout.addPlot(title="Non-Identical, Transform-Invariant Spots")
        x2 = np.random.normal(size=100)
        y2 = np.random.normal(size=100)
        brushes = [pg.mkBrush(int(r), int(g), int(b), 150) for r, g, b in np.random.randint(0, 255, (100, 3))]
        sizes = np.random.uniform(5, 20, 100)
        spots2 = [{'pos': (x, y), 'brush': b, 'size': sz} for x, y, b, sz in zip(x2, y2, brushes, sizes)]
        s2 = pg.ScatterPlotItem(pxMode=True)
        s2.addPoints(spots2)
        s2.sigClicked.connect(lambda plot, pts: self.onClicked(s2, pts))
        p2.addItem(s2)
        self.scatter_items.append(s2)

        # 3. Not transform-invariant and not identical
        p3 = self.graphicsLayout.addPlot(title="Non-Identical, Not Transform-Invariant Spots")
        x3 = np.random.normal(size=100)
        y3 = np.random.normal(size=100)
        brushes3 = [pg.mkBrush(int(r), int(g), int(b), 200) for r, g, b in np.random.randint(0, 255, (100, 3))]
        sizes3 = np.random.uniform(10, 30, 100)
        symbols = np.random.choice(['o', 't', 's', 'd', '+', 'x'], 100)
        spots3 = [{'pos': (x, y), 'brush': b, 'size': sz, 'symbol': sym} for x, y, b, sz, sym in zip(x3, y3, brushes3, sizes3, symbols)]
        s3 = pg.ScatterPlotItem(pxMode=False)
        s3.addPoints(spots3)
        s3.sigClicked.connect(lambda plot, pts: self.onClicked(s3, pts))
        p3.addItem(s3)
        self.scatter_items.append(s3)

        # 4. Performance test with large scatterplot
        self.graphicsLayout.nextRow()
        p4 = self.graphicsLayout.addPlot(title="Large Scatter Plot (Performance Test)")
        n = 10000
        x4 = np.random.normal(size=n)
        y4 = np.random.normal(size=n)
        s4 = pg.ScatterPlotItem(x=x4, y=y4, pen=None, brush=pg.mkBrush(0, 255, 0, 80), size=5, pxMode=True)
        s4.sigClicked.connect(lambda plot, pts: self.onClicked(s4, pts))
        p4.addItem(s4)
        self.scatter_items.append(s4)

    def onClicked(self, scatter, points):
        # Remove previous highlights
        for item in self.highlighted_spots:
            scatter.removeItem(item)
        self.highlighted_spots.clear()
        # Highlight clicked points
        for pt in points:
            pos = pt.pos()
            highlight = pg.ScatterPlotItem([pos.x()], [pos.y()],
                                           size=pt.size() * 1.5 if hasattr(pt, 'size') else 15,
                                           brush=pg.mkBrush(255, 0, 0, 200),
                                           pen=pg.mkPen('w', width=2),
                                           pxMode=pt.pxMode())
            scatter.addItem(highlight)
            self.highlighted_spots.append(highlight)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterDemo()
    win.show()
    sys.exit(app.exec_())
```