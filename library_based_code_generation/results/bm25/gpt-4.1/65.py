```python
import numpy as np
import DREAM
from DREAM import DREAMSettings, runiface
from DREAM.Settings.Equations import IonSpecies, ColdElectronTemperature, E_field, TransportSettings
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.TimeStepper import TimeStepper
from DREAM.Settings.HotElectronDistribution import HotElectronDistribution

# Simulation parameters
nr = 10
r0 = 0.0
r1 = 0.2
nt = 50
t0 = 0.0
t1 = 1e-3
Te0 = 10  # Initial cold electron temperature [eV]
Te1 = 1000  # Final cold electron temperature [eV]
Efield = 0.0  # Electric field [V/m]
ion_species = [{'name': 'D', 'Z': 1, 'n': 1e20}]
D_scalar = 1.0  # Constant diffusion coefficient [m^2/s]
transport_mode = 'diffusive'  # Options: 'diffusive', 'advective', 'off'
enable_hot_tail = True  # Set to False to disable hot tail grid

# Create DREAM settings object
ds = DREAMSettings()

# Set up radial grid
ds.radialgrid.setB0(2.0)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)

# Set up time stepper
times = np.linspace(t0, t1, nt)
ds.timestep.setTimes(times)

# Set up ions
for ion in ion_species:
    ds.eqsys.n_i.addIon(ion['name'], Z=ion['Z'], n=ion['n'])

# Set up cold electron temperature profile (linear ramp)
Te_profile = np.linspace(Te0, Te1, nt)
ds.eqsys.T_cold.setPrescribedData(Te_profile, times=times)

# Set up electric field (constant)
ds.eqsys.E_field.setPrescribedData(Efield)

# Enable/disable hot tail grid
if enable_hot_tail:
    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(8)
    ds.hottailgrid.setNp(40)
    ds.hottailgrid.setPmax(1.0)
else:
    ds.hottailgrid.setEnabled(False)

# Set up transport settings
if transport_mode == 'diffusive':
    ds.transport.setBoundaryCondition(ds.transport.BC_F0)
    ds.transport.setDiffusion(D_scalar)
    ds.transport.setAdvection(0.0)
elif transport_mode == 'advective':
    ds.transport.setBoundaryCondition(ds.transport.BC_F0)
    ds.transport.setDiffusion(0.0)
    ds.transport.setAdvection(D_scalar)
elif transport_mode == 'off':
    ds.transport.setBoundaryCondition(ds.transport.BC_F0)
    ds.transport.setDiffusion(0.0)
    ds.transport.setAdvection(0.0)
else:
    raise ValueError("Unknown transport mode: {}".format(transport_mode))

# Run simulation
runiface(ds, 'output.h5', quiet=False)
```