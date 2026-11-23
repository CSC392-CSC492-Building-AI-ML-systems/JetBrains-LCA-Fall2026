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
        self.setMinimumSize(700, 500)
        exception.handle(self)

        central = qt.QWidget(self)
        self.setCentralWidget(central)
        layout = qt.QVBoxLayout(central)

        # WaitingPushButton
        wait_label = qt.QLabel("WaitingPushButton:")
        self.wait_btn = WaitingPushButton("Click to Toggle Waiting", self)
        self.wait_btn.setToolTip("Click to toggle waiting state")
        self.wait_btn.clicked.connect(self.toggle_waiting)
        layout.addWidget(wait_label)
        layout.addWidget(self.wait_btn)

        # ThreadPoolPushButton
        thread_label = qt.QLabel("ThreadPoolPushButton (computes 2^16):")
        self.thread_btn = ThreadPoolPushButton("Compute 2^16", self)
        self.thread_btn.setToolTip("Click to compute 2^16 in a thread")
        self.thread_btn.setCallable(pow, 2, 16)
        self.thread_btn.sigJobDone.connect(self.thread_job_done)
        layout.addWidget(thread_label)
        layout.addWidget(self.thread_btn)

        # RangeSlider
        range_label = qt.QLabel("RangeSlider (0-100):")
        self.range_slider = RangeSlider(self)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(100)
        self.range_slider.setValues((20, 80))
        self.range_slider.sigValueChanged.connect(self.range_value_changed)
        self.range_slider.sigPositionChanged.connect(self.range_position_changed)
        layout.addWidget(range_label)
        layout.addWidget(self.range_slider)

        # LegendIconWidget
        legend_label = qt.QLabel("LegendIconWidget (Line, Symbol, Colormap):")
        legend_layout = qt.QHBoxLayout()
        # Line style
        line_icon = LegendIconWidget(self)
        line_icon.setLineStyle('-')
        line_icon.setLineColor('blue')
        line_icon.setLineWidth(2)
        line_icon.setToolTip("Line style: solid blue")
        legend_layout.addWidget(line_icon)
        # Symbol style
        symbol_icon = LegendIconWidget(self)
        symbol_icon.setSymbol('o')
        symbol_icon.setSymbolColor('red')
        symbol_icon.setSymbolSize(10)
        symbol_icon.setToolTip("Symbol: red circle")
        legend_layout.addWidget(symbol_icon)
        # Colormap
        cmap_icon = LegendIconWidget(self)
        cmap_icon.setColormap('viridis')
        cmap_icon.setToolTip("Colormap: viridis")
        legend_layout.addWidget(cmap_icon)
        legend_widget = qt.QWidget(self)
        legend_widget.setLayout(legend_layout)
        layout.addWidget(legend_label)
        layout.addWidget(legend_widget)

        # ElidedLabel
        elided_label = qt.QLabel("ElidedLabel (Different Elide Modes):")
        elided_layout = qt.QHBoxLayout()
        long_text = "This is a very long text that will be elided depending on the mode and available space."
        elide_modes = [
            (qt.Qt.ElideLeft, "Left"),
            (qt.Qt.ElideMiddle, "Middle"),
            (qt.Qt.ElideRight, "Right"),
            (qt.Qt.ElideNone, "None"),
        ]
        for mode, name in elide_modes:
            elided = ElidedLabel(long_text, self)
            elided.setElideMode(mode)
            elided.setFixedWidth(120)
            elided.setToolTip(f"Elide mode: {name}")
            elided_layout.addWidget(elided)
        elided_widget = qt.QWidget(self)
        elided_widget.setLayout(elided_layout)
        layout.addWidget(elided_label)
        layout.addWidget(elided_widget)

        layout.addStretch(1)

    def toggle_waiting(self):
        self.wait_btn.setWaiting(not self.wait_btn.isWaiting())

    def thread_job_done(self, result):
        qt.QMessageBox.information(self, "ThreadPool Result", f"2^16 = {result}")

    def range_value_changed(self, values):
        print(f"RangeSlider value changed: {values}")

    def range_position_changed(self, positions):
        print(f"RangeSlider position changed: {positions}")

def main():
    exception.setHandleMode(exception.HANDLE_DIALOG)
    app = qt.QApplication(sys.argv)
    win = SilxDemoApp()
    win.show()
    try:
        sys.exit(app.exec_())
    except Exception:
        exception.handle(sys.exc_info())

if __name__ == "__main__":
    main()
```