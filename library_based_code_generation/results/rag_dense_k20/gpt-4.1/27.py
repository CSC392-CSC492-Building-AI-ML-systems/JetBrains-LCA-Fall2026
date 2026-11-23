```python
import numpy as np
import matplotlib.pyplot as plt
import DREAM
from DREAM import DREAMSettings, runiface
from DREAM.Settings.Equations import IonSpecies
from DREAM.Settings.RadialGrid import TYPE_CENTRAL
from DREAM.Settings.Equations import Temperature, ElectricField

# Stage 1: Initialization
ds = DREAMSettings()

# Radial grid
nr = 10
r = np.linspace(0, 0.2, nr)
ds.radialgrid.setB0(2.5)
ds.radialgrid.setMinorRadius(0.2)
ds.radialgrid.setNr(nr)
ds.radialgrid.setWallRadius(0.22)
ds.radialgrid.setType(TYPE_CENTRAL)

# Time steps
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(10)

# Add ions (e.g., deuterium plasma)
Z = 1
n_D = 5e19  # m^-3
ds.eqsys.n_i.addIon(name='D', Z=Z, n=n_D, iontype=IonSpecies.IONS_DYNAMIC)

# Set up temperature and electric field
Te0 = 100  # eV
ds.eqsys.T_cold.setInitialProfile(Te0 * np.ones(nr))
ds.eqsys.E_field.setInitialProfile(0.5 * np.ones(nr))

# Disable runaway and hot-tail grid
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)
ds.eqsys.n_re.setEnabled(False)

# Save settings and run initialization
ds.save('settings_init.h5')
runiface(ds, 'output_init.h5', quiet=True)

# Stage 2: Ionization
ds2 = DREAMSettings('settings_init.h5')
ds2.eqsys.n_i.setIonization(True)
ds2.save('settings_ionization.h5')
runiface(ds2, 'output_ionization.h5', quiet=True)

# Stage 3: Equilibration
ds3 = DREAMSettings('settings_ionization.h5')
ds3.eqsys.n_i.setEquilibrium(True)
ds3.save('settings_equilibration.h5')
runiface(ds3, 'output_equilibration.h5', quiet=True)

# Stage 4: Radiation
ds4 = DREAMSettings('settings_equilibration.h5')
ds4.eqsys.n_i.setIncludeRadiation(True)
ds4.save('settings_radiation.h5')
runiface(ds4, 'output_radiation.h5', quiet=True)

# Load results
import DREAM.Output
do = DREAM.Output('output_radiation.h5')

# Plot energy balance as a function of temperature
temperatures = np.linspace(10, 2000, 100)  # eV
ohmic_heating = []
radiative_losses = []

for Te in temperatures:
    ds_tmp = DREAMSettings('settings_radiation.h5')
    ds_tmp.eqsys.T_cold.setInitialProfile(Te * np.ones(nr))
    ds_tmp.timestep.setTmax(1e-6)
    ds_tmp.timestep.setNt(1)
    ds_tmp.save('settings_tmp.h5')
    runiface(ds_tmp, 'output_tmp.h5', quiet=True)
    do_tmp = DREAM.Output('output_tmp.h5')
    ohmic = do_tmp.eqsys.W_ohmic[-1, :]
    rad = do_tmp.eqsys.W_rad_tot[-1, :]
    ohmic_heating.append(np.mean(ohmic))
    radiative_losses.append(np.mean(rad))

plt.figure(figsize=(8,6))
plt.plot(temperatures, ohmic_heating, label='Ohmic Heating')
plt.plot(temperatures, radiative_losses, label='Radiative Losses')
plt.xlabel('Temperature (eV)')
plt.ylabel('Power Density (W/m$^3$)')
plt.title('Energy Balance in Plasma at Equilibrium Ionization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```