```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.RunawayElectrons as RunawayElectrons
import DREAM.Settings.Equations.ElectronDensity as ElectronDensity
import DREAM.Settings.Equations.ElectronTemperature as ElectronTemperature
import numpy as np

ds = DREAMSettings()

# Radial grid (single point)
ds.radialgrid.setMinorRadius(0.5)
ds.radialgrid.setNr(1)
ds.radialgrid.setWallRadius(0.51)

# Momentum grid
me_c = 9.10938356e-31 * 2.99792458e8  # m_e * c [kg*m/s]
ds.runawaygrid.setNxi(20)
ds.runawaygrid.setNp(300)
ds.runawaygrid.setPmax(1.0)

# Time grid
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(20)

# Physical parameters
ds.eqsys.E_field.setPrescribedData(6.0)         # V/m
ds.eqsys.n_cold.setPrescribedData(5e19)         # m^-3
ds.eqsys.T_cold.setPrescribedData(100.0)        # eV

# Runaway electrons: solve for runaway current
ds.eqsys.n_re.setType(Equations.RunawayElectrons.TYPE_FLUID)

# Solver and time stepper
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setType(TimeStepper.LINEAR)

# Save output
ds.output.setFilename('output.h5')

# Run simulation
runiface(ds, 'output.h5', quiet=False)
```