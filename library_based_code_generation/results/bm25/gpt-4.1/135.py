```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class PlotDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic plotting examples")
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QGridLayout()
        cw.setLayout(layout)

        # 1. Basic array plotting
        self.plot1 = pg.PlotWidget(title="Basic Array Plot")
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.plot1.plot(x, y, pen='b')
        layout.addWidget(self.plot1, 0, 0)

        # 2. Multiple curves
        self.plot2 = pg.PlotWidget(title="Multiple Curves")
        y2 = np.cos(x)
        self.plot2.plot(x, y, pen='r', name='sin(x)')
        self.plot2.plot(x, y2, pen='g', name='cos(x)')
        layout.addWidget(self.plot2, 0, 1)

        # 3. Drawing with points
        self.plot3 = pg.PlotWidget(title="Drawing with Points")
        self.plot3.plot(x, y, pen=None, symbol='o', symbolBrush='m')
        layout.addWidget(self.plot3, 0, 2)

        # 4. Parametric plot with grid enabled
        self.plot4 = pg.PlotWidget(title="Parametric Plot with Grid")
        t = np.linspace(0, 2*np.pi, 100)
        self.plot4.plot(np.cos(t), np.sin(t), pen='y')
        self.plot4.showGrid(x=True, y=True)
        layout.addWidget(self.plot4, 1, 0)

        # 5. Scatter plot with axis labels and log scale
        self.plot5 = pg.PlotWidget(title="Scatter Plot with Log Scale")
        x5 = np.logspace(0, 2, 100)
        y5 = np.random.normal(size=100) * 10 + 50
        scatter = pg.ScatterPlotItem(x=x5, y=y5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 120))
        self.plot5.addItem(scatter)
        self.plot5.setLabel('left', 'Random Value')
        self.plot5.setLabel('bottom', 'Log X')
        self.plot5.setLogMode(x=True, y=False)
        layout.addWidget(self.plot5, 1, 1)

        # 6. Updating plot
        self.plot6 = pg.PlotWidget(title="Updating Plot")
        self.update_x = np.linspace(0, 2*np.pi, 100)
        self.update_y = np.sin(self.update_x)
        self.update_curve = self.plot6.plot(self.update_x, self.update_y, pen='c')
        layout.addWidget(self.plot6, 1, 2)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.phase = 0
        self.timer.start(50)

        # 7. Filled plot with axis disabled
        self.plot7 = pg.PlotWidget(title="Filled Plot (No Axis)")
        x7 = np.linspace(0, 10, 100)
        y7 = np.abs(np.sin(x7)) * 10
        fill = pg.PlotCurveItem(x7, y7, pen='w')
        fill.setFillLevel(0)
        fill.setBrush((50, 200, 50, 100))
        self.plot7.addItem(fill)
        self.plot7.hideAxis('bottom')
        self.plot7.hideAxis('left')
        layout.addWidget(self.plot7, 2, 0)

        # 8. Region selection
        self.plot8 = pg.PlotWidget(title="Region Selection")
        self.region_x = np.linspace(0, 10, 1000)
        self.region_y = np.sin(self.region_x) + np.random.normal(size=1000)*0.1
        self.region_curve = self.plot8.plot(self.region_x, self.region_y, pen='m')
        self.region = pg.LinearRegionItem([2, 4])
        self.plot8.addItem(self.region)
        layout.addWidget(self.plot8, 2, 1)

        # 9. Zoom on selected region
        self.plot9 = pg.PlotWidget(title="Zoom on Selected Region")
        self.zoom_curve = self.plot9.plot([], [], pen='m')
        layout.addWidget(self.plot9, 2, 2)
        self.region.sigRegionChanged.connect(self.update_zoom_plot)
        self.update_zoom_plot()

        # Enable interactivity (panning, scaling)
        for pw in [self.plot1, self.plot2, self.plot3, self.plot4, self.plot5, self.plot6, self.plot7, self.plot8, self.plot9]:
            pw.setInteractive(True)
            pw.setMouseEnabled(x=True, y=True)

    def update_plot(self):
        self.phase += 0.1
        self.update_y = np.sin(self.update_x + self.phase)
        self.update_curve.setData(self.update_x, self.update_y)

    def update_zoom_plot(self):
        minX, maxX = self.region.getRegion()
        mask = (self.region_x >= minX) & (self.region_x <= maxX)
        self.zoom_curve.setData(self.region_x[mask], self.region_y[mask])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PlotDemo()
    win.show()
    sys.exit(app.exec_())
```