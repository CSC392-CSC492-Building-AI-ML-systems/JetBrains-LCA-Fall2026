```python
import DREAM
from DREAM import DREAMSettings, run
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.Equations import IonSpecies, Temperature, ElectricField
from DREAM.Settings.TimeStepper import TimeStepper
from DREAM.Settings.Solver import Solver
import h5py

# Initial simulation parameters
tMax = 1e-3
dt = 1e-5
nr = 10
nZ = 1
nR = 1
B0 = 5.0
a = 0.22
wallRadius = 0.23
Ip0 = 0.5e6
E0 = 0.0
T0 = 10.0

# Set up DREAM settings
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(B0)
ds.radialgrid.setMinorRadius(a)
ds.radialgrid.setWallRadius(wallRadius)
ds.radialgrid.setNr(nr)

# Time stepper
ds.timestep.setTmax(tMax)
ds.timestep.setDt(dt)

# Add ions (single species, e.g., deuterium)
ds.eqsys.n_i.addIon(name='D', Z=1, n=2e19, iontype=IonSpecies.ION_TYPE_FULLY_IONIZED)

# Set cold electron temperature (evolved)
ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setInitialProfile(T0)

# Set electric field (evolved)
ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(E0)

# Hot-tail grid (required, but not solved)
ds.hottailgrid.setEnabled(False)

# Runaway grid (disable kinetic runaway)
ds.runawaygrid.setEnabled(False)

# Solver settings
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.setLinearSolver(Solver.LINEARSOLVER_MKL)
ds.solver.setTolerance(1e-6)

# Save settings to HDF5 and run
settings_filename = 'dream_selfconsistent_run0.h5'
ds.save(settings_filename)
run(settings_filename)

# First restart
with h5py.File('output.h5', 'r') as f:
    t = f['grid/t'][:]
    tMax_new = t[-1] + 1e-3

ds = DREAMSettings(settings_filename)
ds.timestep.setTmax(tMax_new)
ds.timestep.setDt(dt)
settings_filename1 = 'dream_selfconsistent_run1.h5'
ds.save(settings_filename1)
run(settings_filename1)

# Second restart
with h5py.File('output.h5', 'r') as f:
    t = f['grid/t'][:]
    tMax_new = t[-1] + 1e-3

ds = DREAMSettings(settings_filename1)
ds.timestep.setTmax(tMax_new)
ds.timestep.setDt(dt)
settings_filename2 = 'dream_selfconsistent_run2.h5'
ds.save(settings_filename2)
run(settings_filename2)
```