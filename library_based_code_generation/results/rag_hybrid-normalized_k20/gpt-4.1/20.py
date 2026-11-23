```python
import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import spiceInterface
from Basilisk.test import test_MonteCarloSimulation
from Basilisk.test import test_unitSpice
from Basilisk.test import unitSpice
from Basilisk.test import unloadSpiceKernels

class Controller:
    def __init__(self):
        pass

    # def load_spice_kernels(self, spiceObject):
    #     spiceObject.loadSpiceKernel("de430.bsp", "path/to/de430.bsp")
    #     spiceObject.loadSpiceKernel("naif0012.tls", "path/to/naif0012.tls")
    #     spiceObject.loadSpiceKernel("hubbl_201028_201104_201029.bsp", "path/to/hubbl_201028_201104_201029.bsp")

def get_hubble_state(spiceObject, timeUTC):
    hubbleState, hubbleStateTime = spiceObject.getBodyState(
        "HST", "Earth", "J2000", "NONE", timeUTC)
    print(f"Hubble State at {timeUTC}: {hubbleState}")

class MySimulation:
    def __init__(self, run_number):
        self.run_number = run_number
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        self.simulation = SimulationBaseClass.SimBaseClass()
        self.spiceObject = spiceInterface.SpiceInterface()
        self.scObject = spacecraft.Spacecraft()
        self.init_conditions = {
            "position": np.array([7000.0e3, 0.0, 0.0]),
            "velocity": np.array([0.0, 7.5e3, 0.0])
        }

    def setup(self):
        self.simulation.CreateNewProcess(self.simProcessName)
        self.simulation.CreateNewTask(self.simTaskName, int(1E9 / 60.0), self.simProcessName)
        self.simulation.AddModelToTask(self.simTaskName, self.spiceObject)
        self.simulation.AddModelToTask(self.simTaskName, self.scObject)

        # Load SPICE kernels
        self.spiceObject.SPICEDataPath = os.path.dirname(os.path.abspath(__file__))
        self.spiceObject.KernelList = [
            "de430.bsp",
            "naif0012.tls",
            "hubbl_201028_201104_201029.bsp"
        ]
        self.spiceObject.zeroBase = 'Earth'
        self.spiceObject.epochInMsgName = ""
        self.spiceObject.ModelTag = "spiceInterface"
        self.spiceObject.outputNavMsgName = "spice_output"
        self.spiceObject.bodyNames = ["HST"]

        # Set initial spacecraft state
        self.scObject.hub.r_CN_NInit = self.init_conditions["position"]
        self.scObject.hub.v_CN_NInit = self.init_conditions["velocity"]

    def run(self):
        self.simulation.InitializeSimulation()
        self.simulation.ExecuteSimulation(0.0)
        # Query Hubble state from SPICE
        get_hubble_state(self.spiceObject, "2020 OCT 29 12:00:00.000 UTC")
        self.simulation.TerminateSimulation()

    def cleanup(self):
        unloadSpiceKernels()

def monte_carlo_simulation():
    num_runs = 12
    for run in range(num_runs):
        print(f"\n--- Monte Carlo Run {run+1} ---")
        sim = MySimulation(run)
        sim.setup()
        sim.run()
        sim.cleanup()

if __name__ == "__main__":
    monte_carlo_simulation()
```