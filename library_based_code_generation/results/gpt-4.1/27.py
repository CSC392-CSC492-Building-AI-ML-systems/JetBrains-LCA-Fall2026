```python
import numpy as np
import matplotlib.pyplot as plt
import DREAM
from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
from DREAM.Settings.Equations.Ions import Ions
from DREAM.Settings.RadialGrid import RadialGrid
from DREAM.Settings.Equations.ElectricField import ElectricField
from DREAM.Settings.Equations.Temperature import Temperature
from DREAM.Settings.Equations.RunawayElectrons import RunawayElectrons
from DREAM.Settings.Equations.HotTail import HotTail

# Parameters
nr = 10
r0 = 0
r1 = 0.5
nt = 10
tmax = 1e-3
Te0 = 100  # eV
ne0 = 1e19  # m^-3
E0 = 0.1  # V/m

# Temperature range for plotting
Te_range = np.linspace(10, 2000, 100)  # eV

# Stage 1: Initialization
ds = DREAMSettings()

# Radial grid
ds.radialgrid.setB0(2.0)
ds.radialgrid.setMinorRadius(r1)
ds.radialgrid.setNr(nr)

# Time steps
ds.timestep.setTmax(tmax)
ds.timestep.setNt(nt)

# Add ions (e.g., deuterium)
Z = 1
A = 2
ds.eqsys.n_i.addIon(name='D', Z=Z, n=ne0/2, iontype=Ions.IONTYPE_LITE, Z0=0, nZ0=ne0/2, impurity=False)

# Set up temperature and electric field
ds.eqsys.T_cold.setInitialProfile(Te0)
ds.eqsys.n_cold.setInitialProfile(ne0)
ds.eqsys.E_field.setInitialProfile(E0)

# Disable runaway and hot-tail grid
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

# Save and run initialization
ds.save('settings_stage1_init.h5')
runiface(ds, 'output_stage1_init.h5', quiet=False)

# Stage 2: Ionization
ds2 = DREAMSettings(ds)
ds2.eqsys.n_i.setIonization(True)
ds2.save('settings_stage2_ionization.h5')
runiface(ds2, 'output_stage2_ionization.h5', quiet=False)

# Stage 3: Equilibration
ds3 = DREAMSettings(ds2)
ds3.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds3.save('settings_stage3_equilibration.h5')
runiface(ds3, 'output_stage3_equilibration.h5', quiet=False)

# Stage 4: Radiation
ds4 = DREAMSettings(ds3)
ds4.eqsys.n_i.setPartialScreening(True)
ds4.save('settings_stage4_radiation.h5')
runiface(ds4, 'output_stage4_radiation.h5', quiet=False)

# Calculate ohmic heating and radiative losses as a function of temperature
ohmic_heating = []
radiative_losses = []

for Te in Te_range:
    ds_tmp = DREAMSettings(ds4)
    ds_tmp.eqsys.T_cold.setInitialProfile(Te)
    ds_tmp.eqsys.T_cold.setType(Temperature.TYPE_PRESCRIBED)
    ds_tmp.timestep.setTmax(1e-8)
    ds_tmp.timestep.setNt(1)
    ds_tmp.save('settings_tmp.h5')
    runiface(ds_tmp, 'output_tmp.h5', quiet=True)
    import h5py
    with h5py.File('output_tmp.h5', 'r') as f:
        ohmic = f['eqsys']['P_ohm'][0,0]
        rad = f['eqsys']['P_rad'][0,0]
        ohmic_heating.append(ohmic)
        radiative_losses.append(rad)

ohmic_heating = np.array(ohmic_heating)
radiative_losses = np.array(radiative_losses)

plt.figure(figsize=(8,6))
plt.plot(Te_range, ohmic_heating, label='Ohmic Heating')
plt.plot(Te_range, radiative_losses, label='Radiative Losses')
plt.xlabel('Electron Temperature [eV]')
plt.ylabel('Power Density [W/m$^3$]')
plt.title('Ohmic Heating and Radiative Losses vs Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```