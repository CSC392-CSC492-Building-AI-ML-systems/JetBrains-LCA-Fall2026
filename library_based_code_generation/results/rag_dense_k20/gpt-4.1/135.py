```python
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class PlottingExamples(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic plotting examples")
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QtWidgets.QGridLayout(self.main_widget)

        # 1. Basic array plotting
        self.plot1 = pg.PlotWidget(title="Basic Array Plot")
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.plot1.plot(x, y, pen='b')
        self.layout.addWidget(self.plot1, 0, 0)

        # 2. Multiple curves
        self.plot2 = pg.PlotWidget(title="Multiple Curves")
        y2 = np.cos(x)
        self.plot2.plot(x, y, pen='r', name='sin(x)')
        self.plot2.plot(x, y2, pen='g', name='cos(x)')
        self.layout.addWidget(self.plot2, 0, 1)

        # 3. Drawing with points
        self.plot3 = pg.PlotWidget(title="Drawing with Points")
        self.plot3.plot(x, y, pen=None, symbol='o', symbolBrush='m')
        self.layout.addWidget(self.plot3, 0, 2)

        # 4. Parametric plot with grid enabled
        self.plot4 = pg.PlotWidget(title="Parametric Plot with Grid")
        t = np.linspace(0, 2 * np.pi, 100)
        self.plot4.plot(np.cos(t), np.sin(t), pen='y')
        self.plot4.showGrid(x=True, y=True)
        self.layout.addWidget(self.plot4, 1, 0)

        # 5. Scatter plot with axis labels and log scale
        self.plot5 = pg.PlotWidget(title="Scatter Plot with Log Scale")
        x5 = np.logspace(0, 2, 100)
        y5 = np.random.normal(size=100) * 10 + 50
        scatter = pg.ScatterPlotItem(x=x5, y=y5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 120))
        self.plot5.addItem(scatter)
        self.plot5.setLabel('left', 'Random Value')
        self.plot5.setLabel('bottom', 'Log X')
        self.plot5.setLogMode(x=True, y=False)
        self.layout.addWidget(self.plot5, 1, 1)

        # 6. Updating plot
        self.plot6 = pg.PlotWidget(title="Updating Plot")
        self.curve6 = self.plot6.plot(pen='c')
        self.data6 = np.random.normal(size=100)
        self.ptr6 = 0
        self.layout.addWidget(self.plot6, 1, 2)

        # 7. Filled plot with axis disabled
        self.plot7 = pg.PlotWidget(title="Filled Plot (No Axis)")
        x7 = np.linspace(0, 10, 100)
        y7 = np.abs(np.sin(x7)) * 10
        fill = pg.FillBetweenItem(
            pg.PlotCurveItem(x7, y7, pen='w'),
            pg.PlotCurveItem(x7, np.zeros_like(x7), pen='w'),
            brush=pg.mkBrush(100, 200, 100, 150)
        )
        self.plot7.addItem(fill)
        self.plot7.hideAxis('bottom')
        self.plot7.hideAxis('left')
        self.layout.addWidget(self.plot7, 2, 0)

        # 8. Region selection
        self.plot8 = pg.PlotWidget(title="Region Selection")
        self.region8 = pg.LinearRegionItem([2, 4])
        self.plot8.addItem(self.region8)
        self.plot8.plot(x, y, pen='m')
        self.layout.addWidget(self.plot8, 2, 1)

        # 9. Zoom on selected region
        self.plot9 = pg.PlotWidget(title="Zoom on Selected Region")
        self.curve9 = self.plot9.plot(x, y, pen='b')
        self.layout.addWidget(self.plot9, 2, 2)

        # Connect region selection to zoom plot
        self.region8.sigRegionChanged.connect(self.update_zoom_plot)

        # Timer for updating plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot6)
        self.timer.start(50)

    def update_plot6(self):
        self.data6[self.ptr6 % 100] = np.random.normal()
        self.curve6.setData(self.data6)
        self.ptr6 += 1

    def update_zoom_plot(self):
        minX, maxX = self.region8.getRegion()
        mask = (np.linspace(0, 10, 100) >= minX) & (np.linspace(0, 10, 100) <= maxX)
        x = np.linspace(0, 10, 100)[mask]
        y = np.sin(np.linspace(0, 10, 100))[mask]
        self.curve9.setData(x, y)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PlottingExamples()
    win.show()
    sys.exit(app.exec_())
```