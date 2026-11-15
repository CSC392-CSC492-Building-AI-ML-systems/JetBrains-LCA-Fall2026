```python
import numpy as np
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Equations as Equations
import DREAM.Settings.Transport as Transport
import DREAM.Settings.Ions as Ions
import DREAM.Settings.HotElectronDistribution as HotElectronDistribution

# Simulation parameters
nr = 10
r0 = 0.0
r1 = 0.2
nt = 50
tmax = 1e-3
t = np.linspace(0, tmax, nt)
T0 = 10  # Initial cold electron temperature [eV]
Tf = 1000  # Final cold electron temperature [eV]
E_field = 2e5  # Electric field [V/m]
D_scalar = 0.5  # Constant diffusion coefficient [m^2/s]
ion_species = ['D', 'Ar']  # Example: Deuterium and Argon
ion_Z = [1, 18]
ion_n = [1e20, 1e18]  # Densities [m^-3]

# Transport mode and hot tail grid toggle
transport_mode = 'all_charged_states'  # Options: 'all_charged_states', 'single_charge_state'
enable_hot_tail = True

# Set up DREAM settings
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(2.5)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)

# Time stepper
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)

# Ions
for i, species in enumerate(ion_species):
    ds.eqsys.ions.addIon(name=species, Z=ion_Z[i], n=ion_n[i], iontype=Ions.IONS_DYNAMIC)

# Electric field
ds.eqsys.E_field.setPrescribedData(E_field)

# Cold electron temperature (linear ramp)
Te_profile = np.linspace(T0, Tf, nt)
ds.eqsys.T_cold.setPrescribedData(Te_profile, times=t)

# Hot tail grid
if enable_hot_tail:
    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(10)
    ds.hottailgrid.setNp(40)
    ds.hottailgrid.setPmax(1.0)
else:
    ds.hottailgrid.setEnabled(False)

# Runaway electron density
ds.eqsys.n_re.setInitialProfile(1e10*np.ones(nr))

# Transport settings
if transport_mode == 'all_charged_states':
    ds.transport.n_re.setDiffusion(Transport.DIFFUSION_MODE_ALL_CHARGED, D_scalar)
elif transport_mode == 'single_charge_state':
    ds.transport.n_re.setDiffusion(Transport.DIFFUSION_MODE_SINGLE_CHARGE, D_scalar)
else:
    raise ValueError("Unknown transport mode: {}".format(transport_mode))

# Optionally, set advection to zero
ds.transport.n_re.setAdvection(0)

# Run simulation
runiface(ds, 'output_radial_transport.h5', quiet=False)
```