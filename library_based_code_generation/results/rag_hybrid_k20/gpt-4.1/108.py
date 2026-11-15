```python
import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QSpinBox, QPushButton, QHBoxLayout, QGroupBox
from nplab.experiment import Experiment, ExperimentWithGui
from nplab.instrument.spectrometer import DummySpectrometer
from nplab.instrument.shutter import DummyShutter
from nplab.ui.spectrometer_ui import SpectrometerControlUI, SpectrometerDisplayUI
from nplab.ui.data_browser import DataBrowser

class ShutterSpectrometerExperiment(ExperimentWithGui, Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add dummy spectrometer and shutter
        self.spectrometer = DummySpectrometer()
        self.shutter = DummyShutter()

        self.irradiation_time = 2  # seconds
        self.wait_time = 1         # seconds

        self.data = []

    def do_experiment(self):
        # Open shutter
        self.shutter.open()
        self.sleep(self.irradiation_time)
        # Close shutter
        self.shutter.close()
        self.sleep(self.wait_time)
        # Take spectrum
        spectrum = self.spectrometer.get_spectrum()
        self.data.append(spectrum)
        self.save_data(spectrum)

    def save_data(self, spectrum):
        # Save spectrum to experiment data
        self.datafile.write('spectrum', spectrum)

    def sleep(self, seconds):
        import time
        time.sleep(seconds)

    def gui(self, parent=None):
        return ShutterSpectrometerExperimentUI(self, parent)

class ShutterSpectrometerExperimentUI(QWidget):
    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self.experiment = experiment

        layout = QVBoxLayout(self)

        # Irradiation time control
        irradiation_group = QGroupBox("Irradiation Time (s)")
        irradiation_layout = QHBoxLayout()
        self.irradiation_spin = QSpinBox()
        self.irradiation_spin.setMinimum(1)
        self.irradiation_spin.setMaximum(100)
        self.irradiation_spin.setValue(self.experiment.irradiation_time)
        self.irradiation_spin.valueChanged.connect(self.set_irradiation_time)
        irradiation_layout.addWidget(self.irradiation_spin)
        irradiation_group.setLayout(irradiation_layout)
        layout.addWidget(irradiation_group)

        # Wait time control
        wait_group = QGroupBox("Wait Time (s)")
        wait_layout = QHBoxLayout()
        self.wait_spin = QSpinBox()
        self.wait_spin.setMinimum(0)
        self.wait_spin.setMaximum(100)
        self.wait_spin.setValue(self.experiment.wait_time)
        self.wait_spin.valueChanged.connect(self.set_wait_time)
        wait_layout.addWidget(self.wait_spin)
        wait_group.setLayout(wait_layout)
        layout.addWidget(wait_group)

        # Shutter controls
        shutter_group = QGroupBox("Shutter Controls")
        shutter_layout = QHBoxLayout()
        self.open_shutter_btn = QPushButton("Open Shutter")
        self.close_shutter_btn = QPushButton("Close Shutter")
        self.open_shutter_btn.clicked.connect(self.experiment.shutter.open)
        self.close_shutter_btn.clicked.connect(self.experiment.shutter.close)
        shutter_layout.addWidget(self.open_shutter_btn)
        shutter_layout.addWidget(self.close_shutter_btn)
        shutter_group.setLayout(shutter_layout)
        layout.addWidget(shutter_group)

        # Spectrometer controls
        spec_control_group = QGroupBox("Spectrometer Controls")
        spec_control_layout = QVBoxLayout()
        self.spec_control_ui = SpectrometerControlUI(self.experiment.spectrometer)
        spec_control_layout.addWidget(self.spec_control_ui)
        spec_control_group.setLayout(spec_control_layout)
        layout.addWidget(spec_control_group)

        # Spectrometer display
        spec_display_group = QGroupBox("Spectrometer Display")
        spec_display_layout = QVBoxLayout()
        self.spec_display_ui = SpectrometerDisplayUI(self.experiment.spectrometer)
        spec_display_layout.addWidget(self.spec_display_ui)
        spec_display_group.setLayout(spec_display_layout)
        layout.addWidget(spec_display_group)

        # Data browser
        data_browser_group = QGroupBox("Data Browser")
        data_browser_layout = QVBoxLayout()
        self.data_browser = DataBrowser(self.experiment)
        data_browser_layout.addWidget(self.data_browser)
        data_browser_group.setLayout(data_browser_layout)
        layout.addWidget(data_browser_group)

        # Run experiment button
        self.run_btn = QPushButton("Run Experiment")
        self.run_btn.clicked.connect(self.run_experiment)
        layout.addWidget(self.run_btn)

    def set_irradiation_time(self, value):
        self.experiment.irradiation_time = value

    def set_wait_time(self, value):
        self.experiment.wait_time = value

    def run_experiment(self):
        self.experiment.do_experiment()
        self.data_browser.refresh()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    exp = ShutterSpectrometerExperiment()
    gui = exp.gui()
    gui.show()
    sys.exit(app.exec_())
```