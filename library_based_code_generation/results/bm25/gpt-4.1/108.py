```python
import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QSpinBox, QPushButton, QHBoxLayout
from nplab.experiment import ExperimentWithGui
from nplab.instrument.dummy import DummySpectrometer, DummyShutter
from nplab.datafile import DataFile
from nplab.ui.data_browser import DataBrowser
from nplab.ui.spectrometer_controls import SpectrometerControls
from nplab.ui.shutter_controls import ShutterControls

class ShutterSpectrometerExperiment(ExperimentWithGui):
    def __init__(self, spectrometer, shutter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectrometer = spectrometer
        self.shutter = shutter
        self.datafile = DataFile()
        self.irradiation_time = 2  # seconds
        self.wait_time = 1  # seconds

    def do_experiment(self):
        self.shutter.open()
        self.sleep(self.irradiation_time)
        self.shutter.close()
        self.sleep(self.wait_time)
        spectrum = self.spectrometer.get_spectrum()
        self.datafile.create_group('spectrum')
        self.datafile['spectrum/wavelengths'] = spectrum['wavelengths']
        self.datafile['spectrum/intensities'] = spectrum['intensities']
        self.datafile['spectrum/irradiation_time'] = self.irradiation_time
        self.datafile['spectrum/wait_time'] = self.wait_time

    def sleep(self, seconds):
        import time
        time.sleep(seconds)

    def set_irradiation_time(self, value):
        self.irradiation_time = value

    def set_wait_time(self, value):
        self.wait_time = value

    def gui(self, parent=None):
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)

        # Irradiation time control
        irrad_layout = QHBoxLayout()
        irrad_label = QLabel("Irradiation Time (s):")
        irrad_spin = QSpinBox()
        irrad_spin.setMinimum(1)
        irrad_spin.setMaximum(100)
        irrad_spin.setValue(self.irradiation_time)
        irrad_spin.valueChanged.connect(self.set_irradiation_time)
        irrad_layout.addWidget(irrad_label)
        irrad_layout.addWidget(irrad_spin)
        layout.addLayout(irrad_layout)

        # Wait time control
        wait_layout = QHBoxLayout()
        wait_label = QLabel("Wait Time (s):")
        wait_spin = QSpinBox()
        wait_spin.setMinimum(0)
        wait_spin.setMaximum(100)
        wait_spin.setValue(self.wait_time)
        wait_spin.valueChanged.connect(self.set_wait_time)
        wait_layout.addWidget(wait_label)
        wait_layout.addWidget(wait_spin)
        layout.addLayout(wait_layout)

        # Run experiment button
        run_button = QPushButton("Run Experiment")
        run_button.clicked.connect(self.do_experiment)
        layout.addWidget(run_button)

        # Data browser
        data_browser = DataBrowser(self.datafile)
        layout.addWidget(data_browser)

        # Spectrometer controls
        spec_controls = SpectrometerControls(self.spectrometer)
        layout.addWidget(spec_controls)

        # Shutter controls
        shutter_controls = ShutterControls(self.shutter)
        layout.addWidget(shutter_controls)

        return widget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    spectrometer = DummySpectrometer()
    shutter = DummyShutter()
    experiment = ShutterSpectrometerExperiment(spectrometer, shutter)
    gui = experiment.gui()
    main_window = QWidget()
    main_layout = QVBoxLayout(main_window)
    main_layout.addWidget(gui)
    main_window.setWindowTitle("Shutter & Spectrometer Experiment")
    main_window.show()
    sys.exit(app.exec_())
```