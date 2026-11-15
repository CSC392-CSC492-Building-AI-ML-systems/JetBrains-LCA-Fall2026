```python
import sys
import time
from nplab.experiment import Experiment
from nplab.instrument.shutter import DummyShutter
from nplab.instrument.spectrometer import DummySpectrometer
from nplab.ui.qt import start_qt_app
from nplab.ui.experiment_ui import ExperimentUI
from nplab.ui.spectrometer_ui import SpectrometerUI, SpectrometerDisplayUI, SpectrometerControlUI
from nplab.ui.data_browser import DataBrowser
from PyQt5 import QtWidgets, QtCore

class ShutterSpectrometerExperiment(Experiment):
    def __init__(self, shutter, spectrometer, irradiation_time=1.0, wait_time=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutter = shutter
        self.spectrometer = spectrometer
        self.parameters.declare('irradiation_time', irradiation_time)
        self.parameters.declare('wait_time', wait_time)

    def do(self):
        self.shutter.open()
        self.log_message("Shutter opened.")
        time.sleep(self.parameters['irradiation_time'])
        self.shutter.close()
        self.log_message("Shutter closed.")
        time.sleep(self.parameters['wait_time'])
        spectrum = self.spectrometer.get_spectrum()
        self.log_message("Spectrum acquired.")
        self.data['spectrum'] = spectrum
        self.data['irradiation_time'] = self.parameters['irradiation_time']
        self.data['wait_time'] = self.parameters['wait_time']
        self.save_data()

class ShutterSpectrometerExperimentUI(QtWidgets.QWidget):
    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self.experiment = experiment
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Parameter controls
        param_group = QtWidgets.QGroupBox("Experiment Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        self.irradiation_time_spin = QtWidgets.QDoubleSpinBox()
        self.irradiation_time_spin.setRange(0.1, 100)
        self.irradiation_time_spin.setValue(self.experiment.parameters['irradiation_time'])
        self.irradiation_time_spin.setSuffix(" s")
        self.irradiation_time_spin.valueChanged.connect(
            lambda val: self.experiment.parameters.set('irradiation_time', val)
        )
        param_layout.addRow("Irradiation Time:", self.irradiation_time_spin)

        self.wait_time_spin = QtWidgets.QDoubleSpinBox()
        self.wait_time_spin.setRange(0.1, 100)
        self.wait_time_spin.setValue(self.experiment.parameters['wait_time'])
        self.wait_time_spin.setSuffix(" s")
        self.wait_time_spin.valueChanged.connect(
            lambda val: self.experiment.parameters.set('wait_time', val)
        )
        param_layout.addRow("Wait Time:", self.wait_time_spin)
        layout.addWidget(param_group)

        # Shutter controls
        shutter_group = QtWidgets.QGroupBox("Shutter Controls")
        shutter_layout = QtWidgets.QHBoxLayout(shutter_group)
        self.open_shutter_btn = QtWidgets.QPushButton("Open Shutter")
        self.close_shutter_btn = QtWidgets.QPushButton("Close Shutter")
        self.open_shutter_btn.clicked.connect(self.experiment.shutter.open)
        self.close_shutter_btn.clicked.connect(self.experiment.shutter.close)
        shutter_layout.addWidget(self.open_shutter_btn)
        shutter_layout.addWidget(self.close_shutter_btn)
        layout.addWidget(shutter_group)

        # Spectrometer controls
        spec_group = QtWidgets.QGroupBox("Spectrometer Controls")
        spec_layout = QtWidgets.QVBoxLayout(spec_group)
        self.spectrometer_ui = SpectrometerUI(self.experiment.spectrometer)
        self.spectrometer_display = SpectrometerDisplayUI(self.experiment.spectrometer)
        self.spectrometer_control = SpectrometerControlUI(self.experiment.spectrometer)
        spec_layout.addWidget(self.spectrometer_ui)
        spec_layout.addWidget(self.spectrometer_display)
        spec_layout.addWidget(self.spectrometer_control)
        layout.addWidget(spec_group)

        # Data browser
        self.data_browser = DataBrowser(self.experiment)
        layout.addWidget(self.data_browser)

        # Run experiment button
        self.run_btn = QtWidgets.QPushButton("Run Experiment")
        self.run_btn.clicked.connect(self.run_experiment)
        layout.addWidget(self.run_btn)

        self.setLayout(layout)

    def run_experiment(self):
        self.experiment.do()
        self.data_browser.refresh()

def main():
    app = start_qt_app()
    # Create dummy instruments
    shutter = DummyShutter()
    spectrometer = DummySpectrometer()
    # Create experiment
    exp = ShutterSpectrometerExperiment(shutter, spectrometer)
    # Create UI
    ui = ShutterSpectrometerExperimentUI(exp)
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(ui)
    main_window.setWindowTitle("Shutter & Spectrometer Experiment")
    main_window.resize(800, 600)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```