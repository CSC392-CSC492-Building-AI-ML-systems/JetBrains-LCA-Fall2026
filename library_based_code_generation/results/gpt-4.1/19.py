```python
import DREAM.Settings.Equations as Equations
from DREAM import DREAMSettings, runiface
import numpy as np

ds = DREAMSettings()

# Physical parameters
E_field = 6         # V/m
n_e = 5e19          # m^-3
T_e = 100           # eV

# Radial grid (single zone)
ds.radialgrid.setB0(5)  # Tesla, arbitrary
ds.radialgrid.setMinorRadius(0.5)  # m, arbitrary
ds.radialgrid.setNr(1)

# Momentum grid
pMax = 1            # in units of m_e*c
n_p = 300
n_xi = 20
ds.hottailgrid.setNxi(n_xi)
ds.hottailgrid.setNp(n_p)
ds.hottailgrid.setPmax(pMax)

# Time grid
tMax = 1e-3
nTime = 20
ds.timestep.setTmax(tMax)
ds.timestep.setNt(nTime)

# Set constant temperature, density, and electric field
ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_i.setPrescribedData(n_e)
ds.eqsys.n_e.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Ion species (single hydrogenic)
ds.eqsys.n_i.addIon(name='D', Z=1, n=n_e, iontype=Equations.IonSpecies.ION_TYPE_ION)

# Runaway settings
ds.eqsys.runaway.setEnabled(True)
ds.eqsys.hotTailGrid.setEnabled(True)

# Solver and time stepper
ds.solver.setType(DREAMSettings.Solver.LINEAR_IMPLICIT)
ds.timestep.setType(DREAMSettings.TimeStepper.CONSTANT)

# Output
ds.output.setFilename('output.h5')

# Run simulation
runiface(ds, 'output.h5', quiet=False)
```