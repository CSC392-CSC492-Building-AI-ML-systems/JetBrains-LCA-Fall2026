```python
import DREAM
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Transport as Transport
import DREAM.Settings.TimeStepping as TimeStepping
import DREAM.Settings.Equations.Ions as Ions
import DREAM.Settings.Equations.HotElectronDistribution as HotElectronDistribution
import DREAM.Settings.Equations.ColdElectronTemperature as ColdElectronTemperature
import DREAM.Settings.Equations.ElectricField as ElectricField
import numpy as np

# Simulation parameters
nr = 10
r0 = 0
r1 = 0.2
nt = 50
tmax = 1e-3
T0 = 10  # Initial cold electron temperature [eV]
Tf = 1000  # Final cold electron temperature [eV]
E0 = 0.1  # Initial electric field [V/m]
ion_species = 'D'
Z = 1
n_ion = 1e20  # [m^-3]
D_scalar = 1.0  # Constant diffusion coefficient [m^2/s]
hotTailGridEnabled = True  # Set to False to disable hot tail grid
transport_mode = 'all'  # 'all' or 'single'

# Create DREAM settings object
ds = DREAM.Settings.DREAMSettings()

# Set up radial grid
ds.radialgrid.setB0(5)  # [T]
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)

# Set up time stepper
timesteps = np.linspace(0, tmax, nt)
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)

# Set up ions
ds.eqsys.n_i.addIon(ion_species, Z=Z, n=n_ion, Z0=Z, iontype=Ions.IONS_DYNAMIC)

# Set up cold electron temperature profile (linear ramp)
def Te_profile(t):
    return T0 + (Tf - T0) * t / tmax

ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_PRESCRIBED)
ds.eqsys.T_cold.setPrescribedData(Te_profile(timesteps), times=timesteps)

# Set up electric field (constant)
ds.eqsys.E_field.setType(ElectricField.TYPE_PRESCRIBED)
ds.eqsys.E_field.setPrescribedData(E0 * np.ones(nt), times=timesteps)

# Enable/disable hot tail grid
if hotTailGridEnabled:
    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(10)
    ds.hottailgrid.setNp(40)
    ds.hottailgrid.setPmax(1)
else:
    ds.hottailgrid.setEnabled(False)

# Set up runaway electron density equation
ds.eqsys.n_re.setInitialProfile(0)
ds.eqsys.n_re.setBoundaryCondition(Equations.BC_TYPE_ZERO)

# Set up transport settings
if transport_mode == 'all':
    # All charged states contribute to transport
    ds.transport.setTransportCoefficient(Transport.COEF_D, D_scalar, mode=Transport.DIFFUSION_MODE_ALL)
elif transport_mode == 'single':
    # Only single charge state contributes
    ds.transport.setTransportCoefficient(Transport.COEF_D, D_scalar, mode=Transport.DIFFUSION_MODE_SINGLE)
else:
    raise ValueError("Invalid transport_mode: choose 'all' or 'single'.")

# Enable radial transport for n_re
ds.eqsys.n_re.setTransport(True)

# Run simulation
do = DREAM.runiface(ds, 'output_radial_transport.h5', quiet=False)
```