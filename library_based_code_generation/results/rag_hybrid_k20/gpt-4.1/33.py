```python
import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, ImageItem, GraphicsLayoutWidget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Multi-Plot with ColorBars")
        self.resize(1200, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.graphics_layout = GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout)

        # First plot: non-interactive image with vertical color bar
        self.plot1 = self.graphics_layout.addPlot(row=0, col=0)
        self.plot1.setMenuEnabled(False)
        self.plot1.setMouseEnabled(x=False, y=False)
        self.plot1.hideButtons()
        img_data1 = np.linspace(0, 255, 256*256, dtype=np.uint8).reshape(256, 256)
        self.img_item1 = ImageItem(img_data1)
        self.plot1.addItem(self.img_item1)
        self.plot1.setAspectLocked(True)
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('left')
        self.cbar1 = ColorBarItem(
            values=(img_data1.min(), img_data1.max()),
            cmap='viridis',
            interactive=False,
            width=15,
            orientation='vertical'
        )
        self.cbar1.setImageItem(self.img_item1)
        self.graphics_layout.addItem(self.cbar1, row=0, col=1)

        # Second plot: interactive image with horizontal color bar
        self.plot2 = self.graphics_layout.addPlot(row=1, col=0)
        img_data2 = np.random.normal(loc=128, scale=50, size=(256, 256)).clip(0, 255).astype(np.uint8)
        self.img_item2 = ImageItem(img_data2)
        self.plot2.addItem(self.img_item2)
        self.plot2.setAspectLocked(True)
        self.plot2.hideAxis('bottom')
        self.plot2.hideAxis('left')
        self.cbar2 = ColorBarItem(
            values=(img_data2.min(), img_data2.max()),
            cmap='plasma',
            interactive=True,
            width=15,
            orientation='horizontal'
        )
        self.cbar2.setImageItem(self.img_item2)
        self.graphics_layout.addItem(self.cbar2, row=2, col=0)

        # Third and fourth plots: noisy images sharing a color bar
        self.plot3 = self.graphics_layout.addPlot(row=0, col=2)
        img_data3 = np.random.normal(loc=100, scale=30, size=(256, 256)).clip(0, 255).astype(np.uint8)
        self.img_item3 = ImageItem(img_data3)
        self.plot3.addItem(self.img_item3)
        self.plot3.setAspectLocked(True)
        self.plot3.hideAxis('bottom')
        self.plot3.hideAxis('left')

        self.plot4 = self.graphics_layout.addPlot(row=1, col=2)
        img_data4 = np.random.normal(loc=180, scale=40, size=(256, 256)).clip(0, 255).astype(np.uint8)
        self.img_item4 = ImageItem(img_data4)
        self.plot4.addItem(self.img_item4)
        self.plot4.setAspectLocked(True)
        self.plot4.hideAxis('bottom')
        self.plot4.hideAxis('left')

        # Shared color bar for plot3 and plot4
        min_val = min(img_data3.min(), img_data4.min())
        max_val = max(img_data3.max(), img_data4.max())
        self.cbar_shared = ColorBarItem(
            values=(min_val, max_val),
            cmap='inferno',
            interactive=True,
            width=15,
            orientation='vertical'
        )
        self.cbar_shared.setImageItem([self.img_item3, self.img_item4])
        self.graphics_layout.addItem(self.cbar_shared, row=0, col=3, rowspan=2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    pg.exec()
```