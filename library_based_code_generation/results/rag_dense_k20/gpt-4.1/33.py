```python
import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, ImageItem, GraphicsLayoutWidget

def create_image(shape=(200, 200), noise=False):
    img = np.linspace(0, 255, shape[0]*shape[1], dtype=np.uint8).reshape(shape)
    if noise:
        img = img + np.random.normal(0, 30, shape)
        img = np.clip(img, 0, 255)
    return img

app = QtWidgets.QApplication(sys.argv)
win = GraphicsLayoutWidget(show=True, title="PyQtGraph Multi-Plot Example")
win.resize(1200, 800)

# First plot: non-interactive image with vertical color bar
p1 = win.addPlot(row=0, col=0)
p1.setMenuEnabled(False)
p1.setMouseEnabled(x=False, y=False)
img1 = ImageItem(create_image())
p1.addItem(img1)
p1.hideAxis('bottom')
p1.hideAxis('left')
cbar1 = ColorBarItem(interactive=False, values=(0, 255), width=20, colorMap=pg.colormap.get('viridis'))
cbar1.setImageItem(img1)
win.addItem(cbar1, row=0, col=1)

# Second plot: interactive image with horizontal color bar
p2 = win.addPlot(row=1, col=0)
img2 = ImageItem(create_image(noise=True))
p2.addItem(img2)
p2.hideAxis('bottom')
p2.hideAxis('left')
cbar2 = ColorBarItem(interactive=True, values=(0, 255), width=20, orientation='horizontal', colorMap=pg.colormap.get('plasma'))
cbar2.setImageItem(img2)
win.addItem(cbar2, row=2, col=0)

# Third and fourth plots: noisy images sharing a color bar
p3 = win.addPlot(row=0, col=2)
img3 = ImageItem(create_image(noise=True))
p3.addItem(img3)
p3.hideAxis('bottom')
p3.hideAxis('left')

p4 = win.addPlot(row=1, col=2)
img4 = ImageItem(create_image(noise=True))
p4.addItem(img4)
p4.hideAxis('bottom')
p4.hideAxis('left')

cbar_shared = ColorBarItem(interactive=True, values=(0, 255), width=20, colorMap=pg.colormap.get('cividis'))
cbar_shared.setImageItem([img3, img4])
win.addItem(cbar_shared, row=0, col=3, rowspan=2)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtWidgets, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
```