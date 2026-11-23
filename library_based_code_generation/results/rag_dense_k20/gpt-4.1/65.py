```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations as Equations
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Transport as Transport
import DREAM.Settings.Ions as Ions
import numpy as np

# Simulation parameters
tMax = 1e-3
dt = 1e-5
nr = 10
nTimeSteps = int(tMax/dt)
r0 = 0.0
r1 = 0.2
T0 = 10  # Initial temperature [eV]
Tf = 1000  # Final temperature [eV]
E_field = 0.1  # Electric field [V/m]
Te_cold = 10  # Cold electron temperature [eV]
ion_species = [{'name': 'D', 'Z': 1, 'n': 1e20}]
Drr = 1.0  # Constant scalar diffusion coefficient [m^2/s]

# Transport modes and hot tail grid options
transport_modes = ['prescribed', 'neoclassical']
hot_tail_options = [False, True]

for mode in transport_modes:
    for enable_hot_tail in hot_tail_options:
        ds = DREAMSettings()

        # Radial grid
        ds.radialgrid.setB0(2.0)
        ds.radialgrid.setMinorRadius(r1)
        ds.radialgrid.setNr(nr)

        # Time stepper
        ds.timestep.setTmax(tMax)
        ds.timestep.setNt(nTimeSteps)

        # Ions
        for ion in ion_species:
            ds.eqsys.ions.addIon(name=ion['name'], Z=ion['Z'], n=ion['n'], iontype=Ions.IONS_DYNAMIC)

        # Initial and boundary conditions for n_re
        ds.eqsys.n_re.setInitialProfile(0.0)
        ds.eqsys.n_re.setBoundaryCondition(Equations.BC_TYPE_DIRICHLET)

        # Electric field
        ds.eqsys.E_field.setPrescribedData(E_field)

        # Cold electron temperature
        ds.eqsys.T_cold.setPrescribedData(Te_cold)

        # Hot tail grid
        if enable_hot_tail:
            ds.hottailgrid.setEnabled(True)
            ds.hottailgrid.setNxi(10)
            ds.hottailgrid.setNp(20)
            ds.hottailgrid.setPmax(1.0)
        else:
            ds.hottailgrid.setEnabled(False)

        # Transport settings
        if mode == 'prescribed':
            ds.transport.n_re.setType(Transport.TRANSPORT_PRESCRIBED)
            ds.transport.n_re.setDiffusion(Drr)
        elif mode == 'neoclassical':
            ds.transport.n_re.setType(Transport.TRANSPORT_NEOCASSICAL)
            ds.transport.n_re.setDiffusion(Drr)
        else:
            raise ValueError("Unknown transport mode: {}".format(mode))

        # Run simulation
        output = runiface(ds, 'output_radial_transport_{}_hot_{}.h5'.format(mode, enable_hot_tail), quiet=False)
```