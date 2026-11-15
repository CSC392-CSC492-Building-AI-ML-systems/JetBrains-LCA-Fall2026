```python
from nplab.experiment import Experiment
from nplab.instrument.shutter import DummyShutter
from nplab.instrument.spectrometer import DummySpectrometer
from nplab.ui.qt import ExperimentWindow
from nplab.ui.data_browser import DataBrowser
from nplab.ui.spectrometer import SpectrometerWidget
from nplab.ui.shutter import ShutterWidget

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton
)
import sys
import time

class ShutterSpectrometerExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutter = DummyShutter()
        self.spectrometer = DummySpectrometer()
        self.parameters.define('irradiation_time', 1.0, description='Irradiation Time (s)')
        self.parameters.define('wait_time', 1.0, description='Wait Time (s)')

    def run(self):
        self.shutter.open()
        self.notify('Shutter opened')
        time.sleep(self.parameters['irradiation_time'])
        self.shutter.close()
        self.notify('Shutter closed')
        time.sleep(self.parameters['wait_time'])
        spectrum = self.spectrometer.get_spectrum()
        self.data.new_data_point()
        self.data['spectrum'] = spectrum
        self.data['irradiation_time'] = self.parameters['irradiation_time']
        self.data['wait_time'] = self.parameters['wait_time']
        self.data.save_data_point()
        self.notify('Spectrum acquired')

class ExperimentControlWidget(QWidget):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        layout = QVBoxLayout()
        # Irradiation time control
        ir_layout = QHBoxLayout()
        ir_label = QLabel("Irradiation Time (s):")
        self.ir_spin = QDoubleSpinBox()
        self.ir_spin.setRange(0.1, 100)
        self.ir_spin.setValue(self.experiment.parameters['irradiation_time'])
        self.ir_spin.setSingleStep(0.1)
        self.ir_spin.valueChanged.connect(self.set_irradiation_time)
        ir_layout.addWidget(ir_label)
        ir_layout.addWidget(self.ir_spin)
        layout.addLayout(ir_layout)
        # Wait time control
        wt_layout = QHBoxLayout()
        wt_label = QLabel("Wait Time (s):")
        self.wt_spin = QDoubleSpinBox()
        self.wt_spin.setRange(0.1, 100)
        self.wt_spin.setValue(self.experiment.parameters['wait_time'])
        self.wt_spin.setSingleStep(0.1)
        self.wt_spin.valueChanged.connect(self.set_wait_time)
        wt_layout.addWidget(wt_label)
        wt_layout.addWidget(self.wt_spin)
        layout.addLayout(wt_layout)
        # Run button
        self.run_button = QPushButton("Run Experiment")
        self.run_button.clicked.connect(self.run_experiment)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def set_irradiation_time(self, value):
        self.experiment.parameters['irradiation_time'] = value

    def set_wait_time(self, value):
        self.experiment.parameters['wait_time'] = value

    def run_experiment(self):
        self.experiment.run()

def main():
    app = QApplication(sys.argv)
    experiment = ShutterSpectrometerExperiment()
    main_window = ExperimentWindow(experiment)
    # Data browser
    data_browser = DataBrowser(experiment.data)
    main_window.add_dock_widget(data_browser, "Data Browser")
    # Spectrometer controls
    spec_widget = SpectrometerWidget(experiment.spectrometer)
    main_window.add_dock_widget(spec_widget, "Spectrometer Controls")
    # Shutter controls
    shutter_widget = ShutterWidget(experiment.shutter)
    main_window.add_dock_widget(shutter_widget, "Shutter Controls")
    # Experiment controls
    exp_control = ExperimentControlWidget(experiment)
    main_window.add_dock_widget(exp_control, "Experiment Controls")
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```