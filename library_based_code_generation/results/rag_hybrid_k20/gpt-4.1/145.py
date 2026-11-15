```python
import dream
import dream.settings
import numpy as np

ds = dream.settings.Scenario()

# Set up radial grid
nr = 5
r = np.linspace(0, 0.5, nr)
ds.radialgrid.setB0(2.5)  # Tesla
ds.radialgrid.setMinorRadius(0.5)
ds.radialgrid.setNr(nr)

# Set electric field, density, and temperature
E_field = 5e5  # V/m
n_e = 5e19     # m^-3
T_e = 2e3      # eV

ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_cold.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Momentum grid
pmax = 2.0
Np = 150
Nxi = 40
ds.hottailgrid.setNxi(Nxi)
ds.hottailgrid.setNp(Np)
ds.hottailgrid.setPmax(pmax)

# Initial hot electron Maxwellian
ds.eqsys.f_hot.setInitialProfiles(n0=n_e, T0=T_e, rT0=r, r=n_e*0+r)
ds.eqsys.f_hot.setBoundaryCondition(dream.BoundaryConditionType.REFLECTING)

# Dreicer and avalanche
ds.eqsys.n_re.setDreicer(dream.DREICER_RATE_NEURAL_NETWORK)
ds.eqsys.n_re.setAvalanche(dream.AVALANCHE_MODE_SVENSSON)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Svensson transport coefficients
ds.eqsys.n_re.transport.setSvenssonDiffusion(1.0)
ds.eqsys.n_re.transport.setSvenssonConvection(1.0)

# Nonlinear solver
ds.solver.setType(dream.Solver.NONLINEAR)

# Time stepper
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(10)

# Save settings to HDF5
ds.save('dream_fluid_kinetic_simulation.h5')
```