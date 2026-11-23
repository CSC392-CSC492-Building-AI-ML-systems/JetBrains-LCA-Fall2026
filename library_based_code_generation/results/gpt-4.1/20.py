```python
import os
import numpy as np
import basilisk
from basilisk.architecture import bskLogging
from basilisk.architecture import messaging
from basilisk.simulation import spacecraft
from basilisk.utilities import SimulationBaseClass
from basilisk.utilities import macros
from basilisk.utilities import orbitalMotion
from basilisk.utilities import spiceInterface
from basilisk.utilities import monteCarloInterface
from basilisk.utilities import monteCarloController

# Path to SPICE kernels (update as needed)
SPICE_KERNELS_PATH = os.path.join(os.path.dirname(__file__), "kernels")
HUBBLE_ID = -205

def get_hubble_state(spice_instance, et):
    state, lt = spice_instance.getBodyState(
        "HST", "EARTH", "J2000", "NONE", et)
    print(f"Hubble State at ET={et}: {state}")

class Controller(monteCarloController.MonteCarloController):
    def __init__(self):
        super(Controller, self).__init__()

    def run(self, mcData, runIdx):
        # Example: Uncomment to load kernels here if needed
        # spiceInterface.loadSpiceKernel(os.path.join(SPICE_KERNELS_PATH, "naif0012.tls"))
        pass

class MySimulation(object):
    def __init__(self, run_idx):
        self.run_idx = run_idx
        self.spice = None
        self.scObject = None
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        self.theSim = SimulationBaseClass.SimBaseClass()
        self.init_conditions = {
            "rN": [7000.0e3, 0.0, 0.0],
            "vN": [0.0, 7.5e3, 1.0e3]
        }

    def setup_simulation(self):
        # Set up spacecraft
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraftBody"
        self.scObject.hub.mHub = 750.0
        self.scObject.hub.r_BcB_B = [0.0, 0.0, 0.0]
        self.scObject.hub.IHubPntBc_B = [900.0, 800.0, 600.0]
        self.scObject.hub.r_CN_NInit = self.init_conditions["rN"]
        self.scObject.hub.v_CN_NInit = self.init_conditions["vN"]

        # Add spacecraft to simulation
        self.theSim.AddModelToTask(self.simTaskName, self.scObject)

        # Set up SPICE
        self.spice = spiceInterface.SpiceInterface()
        self.spice.ModelTag = "SpiceInterface"
        self.spice.SPICEDataPath = SPICE_KERNELS_PATH
        self.spice.KernelList = [
            "naif0012.tls",
            "de430.bsp",
            "hst_edited.bsp"
        ]
        self.spice.UTCCalInit = "2024 JUN 01 00:00:00.0"
        self.theSim.AddModelToTask(self.simTaskName, self.spice)

        # Create process and task
        self.theSim.CreateNewProcess(self.simProcessName)
        self.theSim.CreateNewTask(self.simTaskName, macros.sec2nano(1.0))

    def run(self):
        self.setup_simulation()
        self.theSim.InitializeSimulation()
        # Get ET for the initial time
        et = self.spice.getInitialEphemerisTime()
        # Load kernels in each thread
        for kernel in self.spice.KernelList:
            spiceInterface.loadSpiceKernel(os.path.join(SPICE_KERNELS_PATH, kernel))
        # Print Hubble state
        get_hubble_state(self.spice, et)
        self.theSim.TerminateSimulation()

def main():
    num_runs = 12
    mc = monteCarloInterface.MonteCarloSimulation()
    mc.setSimulationBaseClass(MySimulation)
    mc.setExecutionFunction(lambda sim, idx: sim.run())
    mc.setController(Controller())
    mc.setNumRuns(num_runs)
    mc.setArchiveDir("mc_archive")
    mc.executeSimulations()
    # Clean up
    if os.path.exists("mc_archive"):
        import shutil
        shutil.rmtree("mc_archive")

if __name__ == "__main__":
    main()
```