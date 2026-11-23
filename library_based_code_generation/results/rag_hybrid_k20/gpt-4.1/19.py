```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.ElectronDensity as ElectronDensity
import DREAM.Settings.Equations.ElectronTemperature as ElectronTemperature
import DREAM.Settings.Equations.RunawayElectronDensity as RunawayElectronDensity
import DREAM.Settings.Equations.CurrentDensity as CurrentDensity
import DREAM.Settings.Equations.RunawayElectronDistribution as RunawayElectronDistribution
import DREAM.Settings.MomentumGrid as MomentumGrid

import numpy as np

ds = DREAMSettings()

# Radial grid: single point (0D)
ds.radialgrid.setB0(5)  # Tesla, arbitrary
ds.radialgrid.setMinorRadius(1)  # m, arbitrary
ds.radialgrid.setNr(1)

# Momentum grid
me_c = 9.10938356e-31 * 2.99792458e8  # m_e * c [kg*m/s]
pMax = 1.0  # in units of m_e*c
ds.eqsys.f_hot.setMomentumGrid(
    pmax=pMax,
    np=300,
    nxi=20
)

# Physical parameters
E_field = 6.0  # V/m
n_e = 5e19     # m^-3
T_e = 100      # eV

ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.n_cold.setPrescribedData(n_e)
ds.eqsys.T_cold.setPrescribedData(T_e)

# Runaway electron density equation (fluid model)
ds.eqsys.n_re.setType(RunawayElectronDensity.TYPE_FLUID)

# Current density
ds.eqsys.j_ohm.setType(CurrentDensity.TYPE_SELFCONSISTENT)

# Time settings
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(20)

# Solver and time stepper
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setType(TimeStepper.LINEAR)

# Save output
ds.output.setFilename('output.h5')

# Run simulation
runiface(ds, 'output.h5', quiet=False)
```