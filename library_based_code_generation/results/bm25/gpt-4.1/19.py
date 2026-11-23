```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import numpy as np

ds = DREAMSettings()

# Radial grid: single point (0D)
ds.radialgrid.setMinorRadius(0.5)
ds.radialgrid.setNr(1)
ds.radialgrid.setWallRadius(0.6)

# Momentum grid
pMax = 1.0  # in units of m_e*c
np_p = 300
np_xi = 20
ds.eqsys.E_field.setPrescribedData(6.0)  # V/m
ds.eqsys.n_cold.setPrescribedData(5e19)  # m^-3
ds.eqsys.T_cold.setPrescribedData(100)   # eV

ds.hottailgrid.setNxi(np_xi)
ds.hottailgrid.setNp(np_p)
ds.hottailgrid.setPmax(pMax)

ds.runawaygrid.setNxi(np_xi)
ds.runawaygrid.setNp(np_p)
ds.runawaygrid.setPmax(pMax)

# Enable runaway grid
ds.eqsys.f_hot.setInitialProfiles(n0=0)
ds.eqsys.f_hot.setBoundaryCondition(Equations.FHot.BC_F_0)
ds.eqsys.n_re.setInitialProfile(0)
ds.eqsys.n_re.setAvalanche(True)

# Time settings
tMax = 1e-3
nSaves = 20
ds.timestep.setTmax(tMax)
ds.timestep.setNt(nSaves)

# Solver and time stepper
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setType(TimeStepper.CONSTANT)

# Save output
output_filename = 'output.h5'
runiface(ds, output_filename, quiet=False)
```