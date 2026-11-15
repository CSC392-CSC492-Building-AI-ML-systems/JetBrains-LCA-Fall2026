```python
import dream
import dream.tools as dt
import numpy as np

# Create DREAM settings object
ds = dream.Settings()

# Set up radial grid
nr = 5
r = np.linspace(0, 0.22, nr)
ds.radialgrid.setB0(2.5)  # Tesla
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(0.23)

# Set plasma parameters
E_field = 10e3  # V/m
n_e = 5e19      # m^-3
T_e = 2e3       # eV

ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_cold.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Define momentum grid
pmax = 2.0
Np = 150
Nxi = 40
ds.hottailgrid.setNxi(Nxi)
ds.hottailgrid.setNp(Np)
ds.hottailgrid.setPmax(pmax)

# Set up initial hot electron Maxwellian
ds.eqsys.f_hot.setInitialProfiles(n0=n_e, T0=T_e, rT0=r, rN0=r)
ds.eqsys.f_hot.setBoundaryCondition(dream.BoundaryCondition.REFLECTING)

# Enable Dreicer and avalanche generation
ds.eqsys.n_re.setDreicer(dream.DREICER_RATE_NEURAL_NETWORK)
ds.eqsys.n_re.setAvalanche(dream.AVALANCHE_MODE_KNOCKON)

# Set up runaway electron grid (disabled)
ds.runawaygrid.setEnabled(False)

# Set Svensson transport coefficients
D = 1.0  # m^2/s
ds.eqsys.f_hot.setTransport(dream.TRANSPORT_MODE_SVENSSON, D=D)

# Use nonlinear solver
ds.solver.setType(dream.Solver.NONLINEAR)

# Set time stepper
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(10)

# Save settings to HDF5 file
ds.save('dream_fluid_kinetic_simulation.h5')
```