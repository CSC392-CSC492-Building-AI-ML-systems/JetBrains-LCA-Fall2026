```python
import DREAM
from DREAM import runiface
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import numpy as np

ds = DREAM.Settings.DREAMSettings()

# Physical parameters
E_field = 6         # V/m
n_e = 5e19          # m^-3
T_e = 100           # eV

# Radial grid (single point)
ds.radialgrid.setB0(2.0)  # Tesla, arbitrary
ds.radialgrid.setMinorRadius(0.5)  # m, arbitrary
ds.radialgrid.setNr(1)

# Momentum grid
pMax = 1.0          # in units of m_e*c
nP = 300
nXi = 20
ds.hottailgrid.setNxi(nXi)
ds.hottailgrid.setNp(nP)
ds.hottailgrid.setPmax(pMax)

# Time grid
tMax = 1e-3
nSteps = 20
ds.timestep.setTmax(tMax)
ds.timestep.setNt(nSteps)

# Set constant temperature and density
ds.eqsys.T_cold.setType(Equations.T_cold.TYPE_CONSTANT)
ds.eqsys.T_cold.setInitialProfile(T_e)
ds.eqsys.n_cold.setType(Equations.n_cold.TYPE_CONSTANT)
ds.eqsys.n_cold.setInitialProfile(n_e)

# Set constant electric field
ds.eqsys.E_field.setType(Equations.E_field.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(E_field)
ds.eqsys.E_field.setBoundaryCondition(Equations.E_field.BC_TYPE_CONSTANT)

# Enable runaway electron generation
ds.eqsys.n_re.setType(Equations.n_re.TYPE_FLUID)
ds.eqsys.n_re.setInitialProfile(0)

# Set up runaway electron distribution function
ds.eqsys.f_hot.setInitialProfiles(np.zeros((nP, nXi, 1)))
ds.eqsys.f_hot.setBoundaryCondition(Equations.f_hot.BC_TYPE_ZERO)

# Solver and time stepper
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setType(TimeStepper.TYPE_LINEAR)

# Save output
ds.output.setFilename('output.h5')

# Run simulation
runiface(ds, 'output.h5', quiet=False)
```