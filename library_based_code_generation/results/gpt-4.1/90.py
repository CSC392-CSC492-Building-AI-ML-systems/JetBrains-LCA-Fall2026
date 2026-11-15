```python
import DREAM
from DREAM import runiface
from DREAM.Settings import DREAMSettings
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.Equations.Ions as Ions
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.T_cold as T_cold
import DREAM.Settings.Equations.HotTailGrid as HotTailGrid
import DREAM.Settings.Equations.RunawayGrid as RunawayGrid

# Initial simulation parameters
tmax = 1e-3
nt = 100
nr = 5
r0 = 0
r1 = 0.22

# Create DREAM settings object
ds = DREAMSettings()

# Set up radial grid
ds.radialgrid.setB0(5)  # Tesla
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)
ds.radialgrid.setMajorRadius(1.65)

# Set up time stepper
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)
ds.timestep.setType(TimeStepper.TYPE_LINEAR)

# Add ions
ds.eqsys.n_i.addIon(name='D', Z=1, n=2e19, iontype=Ions.IONS_DYNAMIC)

# Set electric field: evolve self-consistently
ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(2.0)  # V/m

# Set cold electron temperature: evolve self-consistently
ds.eqsys.T_cold.setType(T_cold.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setInitialProfile(1e3)  # eV

# Set up hot tail grid (required, but not solved)
ds.hottailgrid.setEnabled(False)
ds.hottailgrid.setNxi(0)
ds.hottailgrid.setNp(0)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Set solver type and parameters
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.setLinearSolver(Solver.LINEARSOLVER_MKL)
ds.solver.setTolerance(1e-6)

# Other necessary settings
ds.eqsys.n_re.setAvalanche(False)
ds.eqsys.n_re.setDreicer(False)

# Save settings to HDF5 and run simulation
ds.save('settings_run0.h5')
runiface('settings_run0.h5', 'output_run0.h5', quiet=False)

# ---- First restart ----
# Load settings from previous output
ds1 = DREAMSettings('output_run0.h5')

# Adjust time stepper for restart
ds1.timestep.setTmax(tmax*2)
ds1.timestep.setNt(nt)

# Save new settings and run
ds1.save('settings_run1.h5')
runiface('settings_run1.h5', 'output_run1.h5', quiet=False)

# ---- Second restart ----
ds2 = DREAMSettings('output_run1.h5')
ds2.timestep.setTmax(tmax*3)
ds2.timestep.setNt(nt)

ds2.save('settings_run2.h5')
runiface('settings_run2.h5', 'output_run2.h5', quiet=False)
```