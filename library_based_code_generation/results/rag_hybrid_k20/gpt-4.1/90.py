```python
import DREAM
from DREAM import DREAMSettings, runSimulation
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.Equations import IonSpecies
from DREAM.Settings.Equations import Temperature, ElectricField
from DREAM.Settings.Equations import HotTailGrid
from DREAM.Settings.Equations import RunawayGrid
from DREAM.Settings.Solver import Solver
from DREAM.Settings.TimeStepper import TimeStepper

import numpy as np

# --- Initial DREAM run ---

ds = DREAMSettings()

# Radial grid
nr = 10
r = np.linspace(0, 0.22, nr)
ds.radialgrid.setB0(2.5)
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(0.23)

# Time stepper
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(10)

# Add ions
ds.eqsys.ni.addIon(name='D', Z=1, n=2e19, iontype=IonSpecies.IONS_DYNAMIC)
ds.eqsys.ni.addIon(name='Ar', Z=18, n=1e17, iontype=IonSpecies.IONS_DYNAMIC)

# Electric field (evolve self-consistently)
ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(2.0)  # V/m

# Cold electron temperature (evolve self-consistently)
ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setInitialProfile(10)  # eV

# Hot-tail grid (required, but no kinetic eqs solved)
ds.hottailgrid.setEnabled(False)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.setLinearSolver(Solver.LINEARSOLVER_MKL)
ds.solver.setTolerance(1e-6)

# Other necessary settings
ds.output.setTiming(True)
ds.output.setFilename('output1.h5')

# Save settings and run
ds.save('settings1.h5')
runSimulation('settings1.h5', quiet=False)

# --- First restart ---

ds2 = DREAMSettings('output1.h5')

# Adjust time stepper
ds2.timestep.setTmax(2e-3)
ds2.timestep.setNt(20)

ds2.output.setFilename('output2.h5')
ds2.save('settings2.h5')
runSimulation('settings2.h5', quiet=False)

# --- Second restart ---

ds3 = DREAMSettings('output2.h5')

# Adjust time stepper
ds3.timestep.setTmax(3e-3)
ds3.timestep.setNt(30)

ds3.output.setFilename('output3.h5')
ds3.save('settings3.h5')
runSimulation('settings3.h5', quiet=False)
```