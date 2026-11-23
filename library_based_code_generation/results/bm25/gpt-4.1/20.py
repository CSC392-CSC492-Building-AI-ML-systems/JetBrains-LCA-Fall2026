```python
import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.architecture import bskSim
from Basilisk.architecture import bskLogging
from Basilisk.architecture import bskUtilities
from Basilisk.architecture import bskSpice
from Basilisk.architecture import bskMonteCarlo
from Basilisk.architecture import bskSimController
from Basilisk.architecture import bskSpiceInterface
from Basilisk.architecture import bskSpiceUtilities
from Basilisk.architecture import bskSpiceState
from Basilisk.architecture import bskSpiceKernel
from Basilisk.architecture import bskSpiceLoader
from Basilisk.architecture import bskSpiceStateProvider
from Basilisk.architecture import bskSpiceStateManager
from Basilisk.architecture import bskSpiceKernelManager
from Basilisk.architecture import bskSpiceKernelLoader
from Basilisk.architecture import bskSpiceKernelUtilities
from Basilisk.architecture import bskSpiceKernelState
from Basilisk.architecture import bskSpiceKernelStateProvider
from Basilisk.architecture import bskSpiceKernelStateManager
from Basilisk.architecture import bskSpiceKernelStateLoader
from Basilisk.architecture import bskSpiceKernelStateUtilities
from Basilisk.architecture import bskSpiceKernelStateKernel
from Basilisk.architecture import bskSpiceKernelStateKernelProvider
from Basilisk.architecture import bskSpiceKernelStateKernelManager
from Basilisk.architecture import bskSpiceKernelStateKernelLoader
from Basilisk.architecture import bskSpiceKernelStateKernelUtilities
from Basilisk.architecture import bskSpiceKernelStateKernelState

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import spiceInterface
from Basilisk.utilities import SimulationParameters
from Basilisk.utilities import SimulationExecutor

# Path to SPICE kernels (update as needed)
KERNEL_PATH = '/path/to/spice/kernels'
HUBBLE_ID = -205  # NAIF ID for Hubble Space Telescope

class Controller:
    def __init__(self):
        pass

    # def load_spice_kernels(self):
    #     spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'naif0012.tls'))
    #     spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'de430.bsp'))
    #     spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'hubble.bsp'))

def get_hubble_state(et):
    spiceInterface.unloadSpiceKernels()
    spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'naif0012.tls'))
    spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'de430.bsp'))
    spiceInterface.loadSpiceKernel(os.path.join(KERNEL_PATH, 'hubble.bsp'))
    state, lt = spiceInterface.getBodyState('HST', et, 'J2000', 'NONE', 'EARTH')
    print(f"Hubble State at ET={et}: {state}")
    spiceInterface.unloadSpiceKernels()
    return state

class MySimulation:
    def __init__(self, run_number):
        self.run_number = run_number
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        self.simulationTime = 10.0  # seconds
        self.initial_conditions = {
            'r_BN_N': [7000e3, 0.0, 0.0],
            'v_BN_N': [0.0, 7.5e3, 0.0]
        }
        self.spice_et = 0.0  # Example ET, can be randomized per run

    def setup_simulation(self):
        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scProcess = self.scSim.CreateNewProcess(self.simProcessName)
        self.scTask = self.scProcess.createTask(self.simTaskName, int(1e-2 * 1e9))

        self.spacecraft = spacecraft.Spacecraft()
        self.spacecraft.hub.r_CN_NInit = self.initial_conditions['r_BN_N']
        self.spacecraft.hub.v_CN_NInit = self.initial_conditions['v_BN_N']
        self.scSim.AddModelToTask(self.simTaskName, self.spacecraft)

    def run(self):
        self.setup_simulation()
        get_hubble_state(self.spice_et)
        self.scSim.InitializeSimulation()
        self.scSim.ConfigureStopTime(self.simulationTime * 1e9)
        self.scSim.ExecuteSimulation()

def monte_carlo_simulation(num_runs=12):
    for run in range(num_runs):
        print(f"\n--- Monte Carlo Run {run+1} ---")
        sim = MySimulation(run)
        sim.run()

def main():
    monte_carlo_simulation(12)
    spiceInterface.unloadSpiceKernels()
    print("\nMonte Carlo simulation complete. Data cleaned up.")

if __name__ == "__main__":
    main()
```