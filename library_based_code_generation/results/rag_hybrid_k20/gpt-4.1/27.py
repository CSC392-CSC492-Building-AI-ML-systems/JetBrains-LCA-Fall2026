```python
import DREAM
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations.Ions as Ions
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.HotElectronDistributionFunction as HotTail
import DREAM.Settings.Equations.RunawayElectronDistributionFunction as Runaway
import numpy as np
import matplotlib.pyplot as plt

# --- Initialization Stage ---
ds = DREAMSettings()

# Radial grid
nr = 10
ds.radialgrid.setB0(5)  # Tesla
ds.radialgrid.setMinorRadius(0.5)  # m
ds.radialgrid.setNr(nr)

# Time steps
ds.timestep.setTmax(1e-3)
ds.timestep.setNt(10)

# Add ions (example: deuterium plasma with some argon impurity)
ds.eqsys.n_i.addIon(name='D', Z=1, n=1e20, iontype=Ions.IONS_DYNAMIC)
ds.eqsys.n_i.addIon(name='Ar', Z=18, n=1e18, iontype=Ions.IONS_DYNAMIC)

# Set up temperature and electric field
Te0 = 1e3  # eV
ds.eqsys.T_cold.setInitialProfile(Te0 * np.ones(nr))
ds.eqsys.E_field.setInitialProfile(0.1 * np.ones(nr))  # V/m

# Disable runaway and hot-tail grids
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

# Save settings and run initialization
ds.save('settings_init.h5')
runiface(ds, 'output_init.h5', quiet=False)

# --- Ionization Stage ---
ds = DREAMSettings('settings_init.h5')
ds.eqsys.n_i.setIonization(True)
ds.save('settings_ionization.h5')
runiface(ds, 'output_ionization.h5', quiet=False)

# --- Equilibration Stage ---
ds = DREAMSettings('settings_ionization.h5')
ds.eqsys.n_i.setEquilibrium(True)
ds.save('settings_equilibration.h5')
runiface(ds, 'output_equilibration.h5', quiet=False)

# --- Radiation Stage ---
ds = DREAMSettings('settings_equilibration.h5')
ds.eqsys.n_i.setIncludeRadiation(True)
ds.save('settings_radiation.h5')
runiface(ds, 'output_radiation.h5', quiet=False)

# --- Plotting Energy Balance ---
from DREAM.DREAMOutput import DREAMOutput

do = DREAMOutput('output_radiation.h5')

# Extract temperature profile at final time
Te = do.eqsys.T_cold[-1, :]
r = do.grid.r

# Calculate ohmic heating and radiative losses as a function of temperature
ohmic_heating = do.other.energy_ohmic[-1, :]
radiative_losses = do.other.energy_rad[-1, :]

plt.figure(figsize=(8,6))
plt.plot(Te, ohmic_heating, label='Ohmic Heating')
plt.plot(Te, radiative_losses, label='Radiative Losses')
plt.xlabel('Electron Temperature [eV]')
plt.ylabel('Power Density [W/m$^3$]')
plt.title('Energy Balance in Plasma at Equilibrium Ionization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```