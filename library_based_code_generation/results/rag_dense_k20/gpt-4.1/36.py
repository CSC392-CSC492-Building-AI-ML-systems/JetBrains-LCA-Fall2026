```python
import os
import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities.MonteCarlo import (
    Controller,
    RetentionPolicy,
    UniformDispersion,
    NormalDispersion,
    LogUniformDispersion,
    SeedDispersion
)
import scenario_AttFeedback


def displayPlots(data, retentionPolicy):
    time = []
    states = []
    for entry in data:
        time.append(entry['time'])
        states.append(entry['state'])
    time = np.array(time)
    states = np.array(states)
    plt.figure()
    plt.plot(time, states)
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.title('Monte Carlo Simulation States vs Time')
    plt.grid(True)
    plt.show()


def run(show_plots: bool):
    mc = Controller()
    mc.setSimulationFunction(scenario_AttFeedback.run)
    mc.setExecutionFunction(scenario_AttFeedback.execute)
    mc.setExecutionCount(10)
    mc.archiveDir = os.path.join(os.getcwd(), "mc_archive")
    mc.addDispersion(SeedDispersion(seed=12345))
    mc.setThreadCount(4)
    mc.setVerbosity(1)
    mc.setVariableCasting(True)
    mc.setDispersionMagnitudeFile("dispersion_magnitude.json")

    dispersions = [
        UniformDispersion("scenario.parameter1", -0.1, 0.1),
        NormalDispersion("scenario.parameter2", 0.0, 1.0),
        LogUniformDispersion("scenario.parameter3", 1e-3, 1e-1)
    ]
    for disp in dispersions:
        mc.addDispersion(disp)

    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLog("scenario_AttFeedback.log")
    retentionPolicy.setDataCallback(displayPlots)
    mc.addRetentionPolicy(retentionPolicy)

    data = mc.executeSimulations()
    if show_plots:
        retentionPolicy.executeCallbacks(data)
    return


if __name__ == "__main__":
    run(True)
```