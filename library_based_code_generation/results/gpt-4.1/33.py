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
        self.initUI()

    def initUI(self):
        glw = GraphicsLayoutWidget()
        self.setCentralWidget(glw)

        # First plot: non-interactive image with vertical color bar
        p1 = glw.addPlot(row=0, col=0)
        p1.setMenuEnabled(False)
        p1.setMouseEnabled(x=False, y=False)
        p1.hideButtons()
        img1_data = np.linspace(0, 255, 256*256, dtype=np.uint8).reshape(256, 256)
        img1 = ImageItem(img1_data)
        p1.addItem(img1)
        cbar1 = ColorBarItem(values=(img1_data.min(), img1_data.max()), cmap='viridis', interactive=False, width=15)
        cbar1.setImageItem(img1)
        glw.addItem(cbar1, row=0, col=1)

        # Second plot: interactive image with horizontal color bar
        p2 = glw.addPlot(row=1, col=0)
        img2_data = np.random.normal(128, 50, (256, 256)).clip(0, 255)
        img2 = ImageItem(img2_data)
        p2.addItem(img2)
        cbar2 = ColorBarItem(values=(img2_data.min(), img2_data.max()), cmap='plasma', interactive=True, width=15, orientation='horizontal')
        cbar2.setImageItem(img2)
        glw.addItem(cbar2, row=2, col=0)

        # Third and fourth plots: noisy images sharing a color bar
        p3 = glw.addPlot(row=0, col=2)
        img3_data = np.random.normal(100, 30, (256, 256)).clip(0, 255)
        img3 = ImageItem(img3_data)
        p3.addItem(img3)

        p4 = glw.addPlot(row=1, col=2)
        img4_data = np.random.normal(180, 40, (256, 256)).clip(0, 255)
        img4 = ImageItem(img4_data)
        p4.addItem(img4)

        # Shared color bar for p3 and p4
        cbar_shared = ColorBarItem(values=(min(img3_data.min(), img4_data.min()), max(img3_data.max(), img4_data.max())),
                                   cmap='inferno', interactive=True, width=15)
        cbar_shared.setImageItem([img3, img4])
        glw.addItem(cbar_shared, row=0, col=3, rowspan=2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
```