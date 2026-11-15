```python
import os
import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities.MonteCarlo import (
    Controller,
    RetentionPolicy,
    SeedDispersion,
    UniformDispersion,
    NormalDispersion,
    LogUniformDispersion
)
import scenario_AttFeedback


def displayPlots(data, retentionPolicy):
    time = np.array(data['time'])
    states = np.array(data['states'])
    plt.figure()
    plt.plot(time, states)
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title('Monte Carlo States vs Time')
    plt.grid(True)
    plt.show()


def run(show_plots):
    mc = Controller()
    mc.setSimulationFunction(scenario_AttFeedback.run)
    mc.setExecutionFunction(scenario_AttFeedback.executionFunction)
    mc.setExecutionCount(100)
    mc.setArchiveDir(os.path.join(os.getcwd(), "mc_archive"))
    mc.setSeedDispersion(SeedDispersion(seed=12345))
    mc.setThreadCount(4)
    mc.setVerbosity(1)
    mc.setVariableCasting(True)
    mc.setDispersionMagnitudeFile("dispersion_magnitude.json")

    dispersions = [
        UniformDispersion(
            variableName="scenario_AttFeedback.initialAttitude",
            lowerLimit=-0.1,
            upperLimit=0.1
        ),
        NormalDispersion(
            variableName="scenario_AttFeedback.initialRate",
            mean=0.0,
            sigma=0.01
        ),
        LogUniformDispersion(
            variableName="scenario_AttFeedback.sensorNoise",
            lowerLimit=1e-5,
            upperLimit=1e-3
        )
    ]
    for disp in dispersions:
        mc.addDispersion(disp)

    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLog("scenario_AttFeedback.log")
    retentionPolicy.setDataCallback(displayPlots)
    mc.addRetentionPolicy(retentionPolicy)

    mc.executeSimulations()
    if show_plots:
        mc.executeCallbacks()
    return


if __name__ == "__main__":
    run(True)
```