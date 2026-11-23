```python
import sys
import time
from nplab.experiment import Experiment, ExperimentWithGui, run_experiment
from nplab.instrument.spectrometer import DummySpectrometer, SpectrometerUI
from nplab.instrument.shutter import DummyShutter
from nplab.ui.data_browser import DataBrowser
from nplab.ui.qt import QtWidgets, start_qt_app

class ShutterSpectrometerExperiment(ExperimentWithGui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add dummy spectrometer and shutter
        self.spectrometer = DummySpectrometer()
        self.shutter = DummyShutter()

        # Default parameters
        self.parameters['irradiation_time'] = 2.0  # seconds
        self.parameters['wait_time'] = 1.0  # seconds

        # Add spectrometer to experiment
        self.add_spectrometer(self.spectrometer)

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        # Parameter controls
        param_group = QtWidgets.QGroupBox("Experiment Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        self.irradiation_time_spin = QtWidgets.QDoubleSpinBox()
        self.irradiation_time_spin.setRange(0.1, 100)
        self.irradiation_time_spin.setValue(self.parameters['irradiation_time'])
        self.irradiation_time_spin.setSuffix(" s")
        self.irradiation_time_spin.valueChanged.connect(
            lambda val: self.parameters.__setitem__('irradiation_time', val)
        )
        param_layout.addRow("Irradiation Time:", self.irradiation_time_spin)

        self.wait_time_spin = QtWidgets.QDoubleSpinBox()
        self.wait_time_spin.setRange(0.1, 100)
        self.wait_time_spin.setValue(self.parameters['wait_time'])
        self.wait_time_spin.setSuffix(" s")
        self.wait_time_spin.valueChanged.connect(
            lambda val: self.parameters.__setitem__('wait_time', val)
        )
        param_layout.addRow("Wait Time:", self.wait_time_spin)
        self.layout.addWidget(param_group)

        # Shutter controls
        shutter_group = QtWidgets.QGroupBox("Shutter Controls")
        shutter_layout = QtWidgets.QHBoxLayout(shutter_group)
        self.open_shutter_btn = QtWidgets.QPushButton("Open Shutter")
        self.close_shutter_btn = QtWidgets.QPushButton("Close Shutter")
        self.open_shutter_btn.clicked.connect(self.shutter.open)
        self.close_shutter_btn.clicked.connect(self.shutter.close)
        shutter_layout.addWidget(self.open_shutter_btn)
        shutter_layout.addWidget(self.close_shutter_btn)
        self.layout.addWidget(shutter_group)

        # Spectrometer controls
        self.spectrometer_ui = SpectrometerUI(self.spectrometer)
        self.layout.addWidget(self.spectrometer_ui)

        # Data browser
        self.data_browser = DataBrowser(self.data)
        self.layout.addWidget(self.data_browser)

        # Run experiment button
        self.run_btn = QtWidgets.QPushButton("Run Experiment")
        self.run_btn.clicked.connect(self.run_experiment)
        self.layout.addWidget(self.run_btn)

        self.setCentralWidget(self.main_widget)

    def run_experiment(self):
        irradiation_time = self.parameters['irradiation_time']
        wait_time = self.parameters['wait_time']

        # Open shutter
        self.shutter.open()
        time.sleep(irradiation_time)
        self.shutter.close()
        time.sleep(wait_time)

        # Take spectrum
        spectrum = self.spectrometer.get_spectrum()
        self.data.new_data_point({'spectrum': spectrum})

        # Update data browser
        self.data_browser.update()

if __name__ == "__main__":
    app = start_qt_app()
    exp = ShutterSpectrometerExperiment()
    exp.show()
    sys.exit(app.exec_())
```