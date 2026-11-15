```python
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Transport as Transport
import DREAM.Settings.TimeStepping as TimeStepping
import DREAM.Settings.Equilibrium as Equilibrium
import DREAM.Settings as ds
import numpy as np

# Create DREAM settings object
ds = ds.DREAMSettings()

# Set up radial grid
nr = 3
r = np.linspace(0, 0.22, nr)
ds.radialgrid.setCustomGridPoints(r)

# Set electric field strength [V/m], electron density [m^-3], and temperature [eV]
E_field = 5e4
n_e = 5e19
T_e = 100

ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_cold.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Set up momentum grid for hot-tail and runaway grids
pMin = 1e-3
pMax = 2.0
Np = 150
xiMin = -1
xiMax = 1
Nxi = 30

# Hot-tail grid
ds.hottailgrid.setNxi(Nxi)
ds.hottailgrid.setNp(Np)
ds.hottailgrid.setPmin(pMin)
ds.hottailgrid.setPmax(pMax)

# Runaway grid
ds.runawaygrid.setEnabled(True)
ds.runawaygrid.setNxi(Nxi)
ds.runawaygrid.setNp(Np)
ds.runawaygrid.setPmin(pMin)
ds.runawaygrid.setPmax(10.0)

# Initial hot electron Maxwellian
ds.eqsys.f_hot.setInitialProfiles(n_e, T_e, r, isNormalized=False)

# Include Dreicer and avalanche generation
ds.eqsys.n_re.setDreicer()
ds.eqsys.n_re.setAvalanche(True)

# Disable the runaway grid (fluid runaway only)
ds.runawaygrid.setEnabled(False)

# Set Svensson transport coefficients
D = 1.0
ds.transport.setSvenssonDiffusion(D)
ds.transport.setSvenssonAdvection(0.0)

# Use nonlinear solver
ds.solver.setType(Solver.NONLINEAR)

# Set time stepper
tMax = 1e-3
dt = 1e-5
ds.timestep.setTmax(tMax)
ds.timestep.setDt(dt)

# Save settings to HDF5 file
ds.save('dream_fluid_kinetic_simulation_settings.h5')
```