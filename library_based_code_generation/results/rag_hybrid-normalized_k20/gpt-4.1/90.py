```python
import DREAM
from DREAM import runSimulation
from DREAM.Settings import DREAMSettings
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.Equations import Eqsys
from DREAM.Settings.Solver import Solver
from DREAM.Settings.TimeStepper import TimeStepper
from DREAM.Settings.ION import Ions
from DREAM.Settings.ColdElectronTemperature import ColdElectronTemperature
from DREAM.Settings.ElectricField import ElectricField
from DREAM.Settings.HotTailGrid import HotTailGrid
from DREAM.Settings.RunawayGrid import RunawayGrid

import numpy as np

# Initial simulation parameters
tmax = 1e-3
nt = 100
nr = 10
r0 = 0
r1 = 0.22
nZ = 1
Z = [1]
nions = [1e20]
T0 = 1e3  # eV
E0 = 0.1  # V/m

# Set up DREAM settings
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(2.5)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)

# Time stepper
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)

# Add ions
for i in range(nZ):
    ds.eqsys.n_i.addIon(name=f"H{i+1}", Z=Z[i], n=nions[i], iontype=Ions.IONTYPE_IMMOBILE)

# Set cold electron temperature (evolved)
ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setInitialProfile(T0*np.ones(nr+1))

# Set electric field (evolved)
ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(E0*np.ones(nr+1))

# Hot-tail grid (needed for fluid, but no kinetic eqs solved)
ds.hottailgrid.setEnabled(True)
ds.hottailgrid.setNxi(1)
ds.hottailgrid.setNp(1)
ds.hottailgrid.setPmax(1)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.setLinearSolver(Solver.LINEARSOLVER_MKL)
ds.solver.setTolerance(1e-6)

# Save settings and run
ds.save("settings_run0.h5")
runSimulation("settings_run0.h5", "output_run0.h5")

# ---- Restart 1 ----
ds1 = DREAMSettings()
ds1.fromOutput("output_run0.h5")

# Adjust time stepper for restart
ds1.timestep.setTmax(tmax*2)
ds1.timestep.setNt(nt*2)

ds1.save("settings_run1.h5")
runSimulation("settings_run1.h5", "output_run1.h5")

# ---- Restart 2 ----
ds2 = DREAMSettings()
ds2.fromOutput("output_run1.h5")

# Adjust time stepper for restart
ds2.timestep.setTmax(tmax*3)
ds2.timestep.setNt(nt*3)

ds2.save("settings_run2.h5")
runSimulation("settings_run2.h5", "output_run2.h5")
```