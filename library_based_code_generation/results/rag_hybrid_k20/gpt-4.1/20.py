```python
import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.architecture import bskLogging
from Basilisk.architecture import bskSim
from Basilisk.architecture import bskSimController
from Basilisk.architecture import bskMonteCarlo
from Basilisk.architecture import bskSpice
from Basilisk.architecture import bskSpiceInterface

NUM_MONTE_CARLO_RUNS = 12
SPICE_KERNELS = [
    "naif0012.tls",
    "de430.bsp",
    "hubbl_edited.bsp"
]
SPICE_PATH = "/path/to/spice/kernels"  # Update this path to your local kernels

class Controller:
    def __init__(self):
        pass

    def load_spice_kernels(self):
        # Example commented out code for loading kernels
        # for kernel in SPICE_KERNELS:
        #     bskSpiceInterface.loadSpiceKernel(os.path.join(SPICE_PATH, kernel))
        pass

class MySimulation:
    def __init__(self, run_number):
        self.run_number = run_number
        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        self.simulation = SimulationBaseClass.SimBaseClass()
        self.spiceObject = None
        self.scObject = None

    def setup_spacecraft(self):
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraftBody"
        self.scObject.hub.mHub = 750.0  # kg
        self.scObject.hub.r_BcB_B = [0.0, 0.0, 0.0]
        self.scObject.hub.IHubPntBc_B = [1400.0, 1000.0, 1200.0,
                                         0.0, 0.0, 0.0]
        self.simulation.AddModelToTask(self.simTaskName, self.scObject)

    def setup_spice(self):
        self.spiceObject = bskSpice.SpiceInterface()
        self.spiceObject.ModelTag = "spiceInterface"
        for kernel in SPICE_KERNELS:
            self.spiceObject.SPICEKernels.append(os.path.join(SPICE_PATH, kernel))
        self.spiceObject.SPICELoadTime = 0.0
        self.spiceObject.SPICEInitialEphemerisTime = 0.0
        self.spiceObject.SPICETarget = "HUBBLE SPACE TELESCOPE"
        self.spiceObject.SPICEOrigin = "EARTH"
        self.spiceObject.SPICEREFrame = "J2000"
        self.simulation.AddModelToTask(self.simTaskName, self.spiceObject)

    def initialize_simulation(self):
        self.simulation.CreateNewProcess(self.simProcessName)
        self.simulation.CreateNewTask(self.simTaskName, int(1E9))
        self.setup_spacecraft()
        self.setup_spice()

    def run(self):
        self.initialize_simulation()
        self.simulation.InitializeSimulation()
        self.simulation.ConfigureStopTime(0.0)
        self.simulation.ExecuteSimulation()

    def get_hubble_state(self):
        spiceMsg = self.spiceObject.spiceOutMsg.recorder()
        self.simulation.AddModelToTask(self.simTaskName, spiceMsg)
        self.simulation.InitializeSimulation()
        self.simulation.ExecuteSimulation()
        state = spiceMsg.PositionVector, spiceMsg.VelocityVector
        print(f"Run {self.run_number}: Hubble State (Position, Velocity): {state}")

def access_spice_and_print_state(run_number):
    sim = MySimulation(run_number)
    sim.run()
    sim.get_hubble_state()

def main():
    controller = Controller()
    # controller.load_spice_kernels()  # This is commented out as per instruction

    for run in range(NUM_MONTE_CARLO_RUNS):
        access_spice_and_print_state(run)

    bskSpiceInterface.unloadSpiceKernels()

if __name__ == "__main__":
    main()
```