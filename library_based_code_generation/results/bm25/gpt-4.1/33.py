```python
import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, ImageItem, GraphicsLayoutWidget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Multi-Plot Example")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.graphics_layout = GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout)

        # First plot: non-interactive image with vertical color bar
        p1 = self.graphics_layout.addPlot(row=0, col=0)
        p1.setMenuEnabled(False)
        p1.setMouseEnabled(x=False, y=False)
        img1_data = np.linspace(0, 255, 256*256, dtype=np.uint8).reshape(256, 256)
        img1 = ImageItem(img1_data)
        p1.addItem(img1)
        color_bar1 = ColorBarItem(interactive=False, values=(img1_data.min(), img1_data.max()), width=15)
        color_bar1.setImageItem(img1)
        color_bar1.setOrientation('vertical')
        color_bar1.setParentItem(p1.getViewBox())

        # Second plot: interactive image with horizontal color bar
        p2 = self.graphics_layout.addPlot(row=0, col=1)
        img2_data = np.random.normal(loc=128, scale=40, size=(256, 256)).clip(0, 255)
        img2 = ImageItem(img2_data)
        p2.addItem(img2)
        color_bar2 = ColorBarItem(interactive=True, values=(img2_data.min(), img2_data.max()), width=15)
        color_bar2.setImageItem(img2)
        color_bar2.setOrientation('horizontal')
        color_bar2.setParentItem(p2.getViewBox())

        # Third and fourth plots: noisy images sharing a color bar
        p3 = self.graphics_layout.addPlot(row=1, col=0)
        img3_data = np.random.normal(loc=100, scale=30, size=(256, 256)).clip(0, 255)
        img3 = ImageItem(img3_data)
        p3.addItem(img3)

        p4 = self.graphics_layout.addPlot(row=1, col=1)
        img4_data = np.random.normal(loc=180, scale=50, size=(256, 256)).clip(0, 255)
        img4 = ImageItem(img4_data)
        p4.addItem(img4)

        # Shared color bar for p3 and p4
        shared_color_bar = ColorBarItem(interactive=True, values=(0, 255), width=15)
        shared_color_bar.setImageItem(img3)
        shared_color_bar.setImageItem(img4)
        shared_color_bar.setOrientation('vertical')
        # Place the color bar to the right of p4
        shared_color_bar.setParentItem(p4.getViewBox())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtWidgets, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
```