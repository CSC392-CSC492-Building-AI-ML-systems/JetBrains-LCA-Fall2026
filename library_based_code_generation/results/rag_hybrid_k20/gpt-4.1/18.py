```python
import sys
from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.RangeSlider import RangeSlider
from silx.gui.widgets.LegendIconWidget import LegendIconWidget
from silx.gui.widgets.ElidedLabel import ElidedLabel
from silx.gui import exception

class SilxDemoApp(qt.QMainWindow):
    def __init__(self, parent=None):
        super(SilxDemoApp, self).__init__(parent)
        self.setWindowTitle("Silx Widgets Demo")

        central_widget = qt.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = qt.QVBoxLayout(central_widget)

        # WaitingPushButton
        waiting_label = qt.QLabel("WaitingPushButton:")
        layout.addWidget(waiting_label)
        self.waiting_btn = WaitingPushButton("Click to Toggle Waiting")
        self.waiting_btn.setToolTip("Click to toggle waiting state")
        self.waiting_btn.clicked.connect(self.toggle_waiting)
        layout.addWidget(self.waiting_btn)

        # ThreadPoolPushButton
        threadpool_label = qt.QLabel("ThreadPoolPushButton (computes 2^16):")
        layout.addWidget(threadpool_label)
        self.threadpool_btn = ThreadPoolPushButton("Compute 2^16")
        self.threadpool_btn.setToolTip("Click to compute 2^16 in a thread")
        self.threadpool_btn.setCallable(pow, 2, 16)
        self.threadpool_btn.sigJobDone.connect(self.threadpool_done)
        layout.addWidget(self.threadpool_btn)

        # RangeSlider
        rangeslider_label = qt.QLabel("RangeSlider:")
        layout.addWidget(rangeslider_label)
        self.rangeslider = RangeSlider(qt.Qt.Horizontal)
        self.rangeslider.setMinimum(0)
        self.rangeslider.setMaximum(100)
        self.rangeslider.setValues((20, 80))
        self.rangeslider.sigValueChanged.connect(self.rangeslider_value_changed)
        self.rangeslider.sigPositionChanged.connect(self.rangeslider_position_changed)
        layout.addWidget(self.rangeslider)

        # LegendIconWidget
        legend_label = qt.QLabel("LegendIconWidget (Line, Symbol, Colormap):")
        layout.addWidget(legend_label)
        legend_layout = qt.QHBoxLayout()
        # Line style
        line_icon = LegendIconWidget()
        line_icon.setLineStyle('-')
        line_icon.setLineColor('blue')
        line_icon.setLineWidth(2)
        legend_layout.addWidget(qt.QLabel("Line:"))
        legend_layout.addWidget(line_icon)
        # Symbol style
        symbol_icon = LegendIconWidget()
        symbol_icon.setSymbol('o')
        symbol_icon.setSymbolColor('red')
        symbol_icon.setSymbolSize(10)
        legend_layout.addWidget(qt.QLabel("Symbol:"))
        legend_layout.addWidget(symbol_icon)
        # Colormap style
        cmap_icon = LegendIconWidget()
        cmap_icon.setColormap('viridis')
        legend_layout.addWidget(qt.QLabel("Colormap:"))
        legend_layout.addWidget(cmap_icon)
        layout.addLayout(legend_layout)

        # ElidedLabel
        elided_label = qt.QLabel("ElidedLabel (Different Elide Modes):")
        layout.addWidget(elided_label)
        elide_layout = qt.QHBoxLayout()
        long_text = "This is a very long text that should be elided if it does not fit in the label."
        elided_left = ElidedLabel(long_text)
        elided_left.setElideMode(qt.Qt.ElideLeft)
        elided_left.setFixedWidth(150)
        elide_layout.addWidget(qt.QLabel("Left:"))
        elide_layout.addWidget(elided_left)
        elided_middle = ElidedLabel(long_text)
        elided_middle.setElideMode(qt.Qt.ElideMiddle)
        elided_middle.setFixedWidth(150)
        elide_layout.addWidget(qt.QLabel("Middle:"))
        elide_layout.addWidget(elided_middle)
        elided_right = ElidedLabel(long_text)
        elided_right.setElideMode(qt.Qt.ElideRight)
        elided_right.setFixedWidth(150)
        elide_layout.addWidget(qt.QLabel("Right:"))
        elide_layout.addWidget(elided_right)
        layout.addLayout(elide_layout)

    def toggle_waiting(self):
        self.waiting_btn.setWaiting(not self.waiting_btn.isWaiting())

    def threadpool_done(self, result):
        qt.QMessageBox.information(self, "ThreadPool Result", f"2^16 = {result}")

    def rangeslider_value_changed(self, values):
        print(f"RangeSlider value changed: {values}")

    def rangeslider_position_changed(self, positions):
        print(f"RangeSlider position changed: {positions}")

def main():
    exception.handle_all_exceptions()
    app = qt.QApplication(sys.argv)
    win = SilxDemoApp()
    win.show()
    try:
        sys.exit(app.exec_())
    finally:
        exception.restore_standard_excepthook()

if __name__ == "__main__":
    main()
```