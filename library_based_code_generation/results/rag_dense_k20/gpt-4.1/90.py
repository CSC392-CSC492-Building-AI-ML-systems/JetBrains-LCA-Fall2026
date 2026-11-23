```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equilibrium as Equilibrium
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Equations as Equations
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.ION as ION
import DREAM.Settings.HotTailGrid as HotTailGrid
import DREAM.Settings.RunawayGrid as RunawayGrid
import numpy as np

# Initial simulation parameters
tmax = 1e-3
nt = 10
nr = 5
r0 = 0
r1 = 0.22
B0 = 2.5
Ip0 = 0.0
E0 = 0.1
Te0 = 1e3

# Set up DREAM settings
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(B0)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)

# Equilibrium
ds.eqsys.E_field.setInitialProfile(E0)
ds.eqsys.T_cold.setInitialProfile(Te0)
ds.eqsys.n_cold.setInitialProfile(5e19)
ds.eqsys.n_cold.setType(Equations.Equilibrium.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setType(Equations.Equilibrium.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setType(Equations.Equilibrium.TYPE_SELFCONSISTENT)

# Time stepper
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)

# Ions
ds.eqsys.n_i.addIon(name='D', Z=1, n=5e19, iontype=ION.TYPE_DYNAMIC)

# Hot-tail grid (required for fluid runs, but not solved)
ds.hottailgrid.setEnabled(False)

# Runaway grid (disable)
ds.runawaygrid.setEnabled(False)

# Solver
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.setLinearSolver(Solver.LINEARSOLVER_MKL)
ds.solver.setTolerance(1e-6)

# Save settings and run
ds.save('settings0.h5')
runiface.runSimulation('settings0.h5', 'output0.h5')

# First restart
ds1 = DREAMSettings('output0.h5')
ds1.timestep.setTmax(tmax*2)
ds1.timestep.setNt(nt*2)
ds1.save('settings1.h5')
runiface.runSimulation('settings1.h5', 'output1.h5')

# Second restart
ds2 = DREAMSettings('output1.h5')
ds2.timestep.setTmax(tmax*3)
ds2.timestep.setNt(nt*3)
ds2.save('settings2.h5')
runiface.runSimulation('settings2.h5', 'output2.h5')
```