```python
import os
import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities.MonteCarlo import (
    Controller,
    RetentionPolicy,
    Dispersions,
    SeedDispersion
)
import scenario_AttFeedback


def displayPlots(data, retentionPolicy):
    # Extract time and states from data
    time = []
    states = []
    for entry in data:
        # Assuming 'time' and 'state' keys exist in entry
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
    # Create Monte Carlo controller
    mc = Controller()
    # Set simulation and execution functions
    mc.setSimulationFunction(scenario_AttFeedback.run)
    mc.setExecutionFunction(scenario_AttFeedback.execute)
    # Set execution count
    mc.setExecutionCount(10)
    # Set archive directory
    mc.archiveDirectory = os.path.join(os.getcwd(), "mcArchive")
    # Set seed dispersion
    mc.setShouldDisperseSeeds(True)
    # Set thread count
    mc.setThreadCount(4)
    # Set verbosity
    mc.setVerbosity(1)
    # Set variable casting
    mc.setVariableCasting(True)
    # Set dispersion magnitude file
    mc.setDispersionMagnitudeFile("dispersion_magnitudes.json")

    # Define dispersions
    dispersions = [
        Dispersions("initial_state", "normal", [0.0, 0.0, 0.0], [0.01, 0.01, 0.01]),
        Dispersions("sensor_bias", "uniform", [-0.001], [0.001]),
        SeedDispersion()
    ]
    # Add dispersions to controller
    for disp in dispersions:
        mc.addDispersion(disp)

    # Create retention policy
    rp = RetentionPolicy()
    # Add message logs
    rp.addMessageLog("scStateOutMsg")
    # Set data callback
    rp.setDataCallback(displayPlots)
    # Add retention policy to controller
    mc.addRetentionPolicy(rp)

    # Execute simulations
    data = mc.executeSimulations()
    # Execute callbacks if show_plots is True
    if show_plots:
        rp.executeCallbacks(data)
    return


if __name__ == "__main__":
    run(True)
```