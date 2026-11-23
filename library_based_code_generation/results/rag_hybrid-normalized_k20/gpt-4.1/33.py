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
        self.plot1 = self.graphics_layout.addPlot(row=0, col=0)
        self.plot1.setMenuEnabled(False)
        self.plot1.setMouseEnabled(x=False, y=False)
        img_data1 = np.linspace(0, 255, 256*256, dtype=np.uint8).reshape(256, 256)
        image_item1 = ImageItem(img_data1)
        self.plot1.addItem(image_item1)
        color_bar1 = ColorBarItem(values=(img_data1.min(), img_data1.max()), cmap='viridis', orientation='vertical')
        color_bar1.setImageItem(image_item1)
        self.graphics_layout.addItem(color_bar1, row=0, col=1)

        # Second plot: interactive noisy image with horizontal color bar
        self.plot2 = self.graphics_layout.addPlot(row=1, col=0)
        img_data2 = np.random.normal(loc=128, scale=40, size=(256, 256)).clip(0, 255).astype(np.uint8)
        image_item2 = ImageItem(img_data2)
        self.plot2.addItem(image_item2)
        color_bar2 = ColorBarItem(values=(img_data2.min(), img_data2.max()), cmap='plasma', orientation='horizontal')
        color_bar2.setImageItem(image_item2)
        self.graphics_layout.addItem(color_bar2, row=2, col=0)

        # Third and fourth plots: noisy images sharing a color bar
        img_data3 = np.random.normal(loc=100, scale=30, size=(256, 256)).clip(0, 255).astype(np.uint8)
        img_data4 = np.random.normal(loc=180, scale=50, size=(256, 256)).clip(0, 255).astype(np.uint8)
        self.plot3 = self.graphics_layout.addPlot(row=0, col=2)
        self.plot4 = self.graphics_layout.addPlot(row=1, col=2)
        image_item3 = ImageItem(img_data3)
        image_item4 = ImageItem(img_data4)
        self.plot3.addItem(image_item3)
        self.plot4.addItem(image_item4)
        min_val = min(img_data3.min(), img_data4.min())
        max_val = max(img_data3.max(), img_data4.max())
        color_bar_shared = ColorBarItem(values=(min_val, max_val), cmap='inferno', orientation='vertical')
        color_bar_shared.setImageItem([image_item3, image_item4])
        self.graphics_layout.addItem(color_bar_shared, row=0, col=3, rowspan=2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtWidgets, 'PYQT_VERSION'):
        pg.exec()
```