```python
import DREAM
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Transport as Transport
import DREAM.Settings.TimeStepping as TimeStepping
import DREAM.Settings.Equilibrium as Equilibrium
import DREAM.Settings.Ions as Ions
import numpy as np

# Simulation parameters
tMax = 1e-3
dt = 1e-5
nr = 10
r0 = 0.0
r1 = 0.2
T0 = 10  # Initial temperature [eV]
Tf = 1000  # Final temperature [eV]
E_field = 0.0  # Electric field [V/m]
Te_cold = 10  # Cold electron temperature [eV]
ion_species = [{'name': 'D', 'Z': 1, 'n': 1e20}]
D_scalar = 1.0  # Diffusion coefficient [m^2/s]
transport_mode = 'DIFFUSION'  # Options: 'DIFFUSION', 'CONVECTION', 'NONE'
enable_hot_tail = True

# Create DREAM settings object
ds = DREAM.Settings.DREAMSettings()

# Radial grid
ds.radialgrid.setB0(2.5)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(r1*1.1)

# Time stepper
ds.timestep.setTmax(tMax)
ds.timestep.setNt(int(tMax/dt))

# Ions
for ion in ion_species:
    ds.eqsys.n_i.addIon(ion['name'], ion['Z'], n=ion['n'], Z0=ion['Z'])

# Set E_field and cold electron temperature
ds.eqsys.E_field.setPrescribedData(E_field)
ds.eqsys.T_cold.setPrescribedData(Te_cold)

# Enable hot tail grid
if enable_hot_tail:
    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(10)
    ds.hottailgrid.setNp(20)
    ds.hottailgrid.setPmax(1.0)
else:
    ds.hottailgrid.setEnabled(False)

# Set up transport settings
if transport_mode == 'DIFFUSION':
    ds.transport.setTransport(Transport.BC_TYPE_ZERO, D=D_scalar)
elif transport_mode == 'CONVECTION':
    ds.transport.setTransport(Transport.BC_TYPE_ZERO, D=0.0, Vp=0.1)
elif transport_mode == 'NONE':
    ds.transport.setTransport(Transport.BC_TYPE_ZERO, D=0.0, Vp=0.0)
else:
    raise ValueError("Unknown transport mode: {}".format(transport_mode))

# Set initial and final temperatures (for demonstration, ramp linearly)
times = np.linspace(0, tMax, int(tMax/dt)+1)
Te_profile = np.linspace(T0, Tf, len(times))
ds.eqsys.T_cold.setPrescribedData(Te_profile, times=times)

# Set initial n_re profile
n_re_init = np.linspace(1e18, 1e19, nr)
ds.eqsys.n_re.setInitialProfile(n_re_init)

# Run simulation
do = DREAM.runiface(ds, 'output_radial_transport.h5', quiet=False)
```