```python
import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.architecture import bskSim
from Basilisk.utilities import SimulationBaseClass
from Basilisk.test import test_MonteCarloSimulation
from Basilisk.test import test_unitSpiceSc
from Basilisk.test import test_scenarioFlybySpice
from Basilisk.test import test_scenarioHelioTransSpice
from Basilisk.test import test_unitSpice
from Basilisk.test import test_scenarioBasicOrbit
from Basilisk.test import test_scenarioMsisAtmosphereOrbit
from Basilisk.test import test_unitSimVSCMG
from Basilisk.test import test_unitSimStarTracker
from Basilisk.test import test_MonteCarloSimulationDatashader
from Basilisk.test import spacecraftReconfigTestFunction
from Basilisk.test import SimulationParameters
from Basilisk.simulation import spiceInterface
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import bskLogging
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import spiceInterface as spiceUtil

NUM_MONTE_CARLO_RUNS = 12

class Controller:
    def __init__(self):
        pass

    # def load_spice_kernels(self):
    #     spiceInterface = spiceUtil.SpiceInterface()
    #     spiceInterface.SPICEDataPath = "path_to_spice_kernels"
    #     spiceInterface.SPICEKernels = [
    #         "naif0012.tls",
    #         "de430.bsp",
    #         "hubbl_edited.bsp"
    #     ]
    #     spiceInterface.ModelTag = "spice_interface"
    #     spiceInterface.epochInMsgName = ""
    #     spiceInterface.epochOutMsgName = ""
    #     spiceInterface.epoch = macros.sec2JD(0)
    #     spiceInterface.bodyNames = ["HUBBLE"]
    #     spiceInterface.centralBody = "EARTH"
    #     spiceInterface.outputMsgName = "spice_output"
    #     return spiceInterface

def get_hubble_state_from_spice():
    spice = spiceUtil.SpiceInterface()
    spice.SPICEDataPath = "path_to_spice_kernels"
    spice.SPICEKernels = [
        "naif0012.tls",
        "de430.bsp",
        "hubbl_edited.bsp"
    ]
    spice.ModelTag = "spice_interface"
    spice.epochInMsgName = ""
    spice.epochOutMsgName = ""
    spice.epoch = macros.sec2JD(0)
    spice.bodyNames = ["HUBBLE"]
    spice.centralBody = "EARTH"
    spice.outputMsgName = "spice_output"
    spice.InitializeSimulation()
    state, _ = spice.getBodyState("HUBBLE", "EARTH", macros.sec2JD(0))
    print("Hubble State from Spice:", state)
    spiceUtil.unloadSpiceKernels()
    return state

class MySimulation:
    def __init__(self, run_number):
        self.run_number = run_number
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        self.simulation = SimulationBaseClass.SimBaseClass()
        self.setup_spacecraft()
        self.setup_spice()
    
    def setup_spacecraft(self):
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraft"
        # Set initial conditions (example values)
        self.scObject.hub.r_CN_NInit = [7000.0*1000, 0.0, 0.0]
        self.scObject.hub.v_CN_NInit = [0.0, 7.5*1000, 0.0]
        self.simulation.AddModelToTask(self.simTaskName, self.scObject)
    
    def setup_spice(self):
        self.spiceObject = spiceUtil.SpiceInterface()
        self.spiceObject.SPICEDataPath = "path_to_spice_kernels"
        self.spiceObject.SPICEKernels = [
            "naif0012.tls",
            "de430.bsp",
            "hubbl_edited.bsp"
        ]
        self.spiceObject.ModelTag = "spice_interface"
        self.spiceObject.epochInMsgName = ""
        self.spiceObject.epochOutMsgName = ""
        self.spiceObject.epoch = macros.sec2JD(0)
        self.spiceObject.bodyNames = ["HUBBLE"]
        self.spiceObject.centralBody = "EARTH"
        self.spiceObject.outputMsgName = "spice_output"
        self.simulation.AddModelToTask(self.simTaskName, self.spiceObject)
    
    def run(self):
        self.simulation.InitializeSimulation()
        state, _ = self.spiceObject.getBodyState("HUBBLE", "EARTH", macros.sec2JD(0))
        print(f"Run {self.run_number}: Hubble State from Spice:", state)
        spiceUtil.unloadSpiceKernels()

def monte_carlo_thread(run_number):
    # Each thread loads Spice and prints Hubble state
    get_hubble_state_from_spice()
    sim = MySimulation(run_number)
    sim.run()

def main():
    for i in range(NUM_MONTE_CARLO_RUNS):
        monte_carlo_thread(i+1)
    # Clean up data after test
    spiceUtil.unloadSpiceKernels()

if __name__ == "__main__":
    main()
```