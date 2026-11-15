```python
import DREAM
from DREAM import runSimulation
from DREAM.Settings.Equations import IonSpecies, Temperature, RadialGrid, ElectricField
from DREAM.Settings.Solver import Solver
from DREAM.Output import DREAMOutput
import matplotlib.pyplot as plt

# Stage 1: Initialization
ds = DREAM.Settings.DREAMSettings()

# Radial grid
ds.radialgrid.setB0(5)                # Magnetic field [T]
ds.radialgrid.setMinorRadius(0.5)     # Plasma minor radius [m]
ds.radialgrid.setNr(10)               # Number of radial grid points

# Time steps
ds.timestep.setTmax(1e-3)             # Total simulation time [s]
ds.timestep.setNt(10)                 # Number of time steps

# Add ions (e.g., deuterium and argon)
ds.eqsys.n_i.addIon(name='D', Z=1, n=2e19)
ds.eqsys.n_i.addIon(name='Ar', Z=18, n=1e17)

# Set up temperature and electric field
ds.eqsys.T_cold.setInitialProfile(1e3)    # Initial electron temperature [eV]
ds.eqsys.E_field.setInitialProfile(0.1)   # Initial electric field [V/m]

# Disable runaway and hot-tail grid
ds.hottailgrid.setEnabled(False)
ds.runawaygrid.setEnabled(False)

# Save settings and run initialization
ds.save('settings_stage1_init.h5')
do1 = runSimulation(ds, 'output_stage1_init.h5')

# Stage 2: Ionization
ds.eqsys.n_i.setIonization(True)
ds.save('settings_stage2_ionization.h5')
do2 = runSimulation(ds, 'output_stage2_ionization.h5')

# Stage 3: Equilibration
ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
ds.save('settings_stage3_equilibration.h5')
do3 = runSimulation(ds, 'output_stage3_equilibration.h5')

# Stage 4: Radiation
ds.eqsys.n_i.setPartialIonization(True)
ds.save('settings_stage4_radiation.h5')
do4 = runSimulation(ds, 'output_stage4_radiation.h5')

# Extract results for plotting
output = DREAMOutput('output_stage4_radiation.h5')
T = output.eqsys.T_cold[:,0,:]  # [time, radius]
t = output.grid.t
r = output.grid.r

# Ohmic heating and radiative losses
ohmic = output.other.energy_ohmic[:,0,:]      # [time, radius]
radiation = output.other.energy_rad[:,0,:]    # [time, radius]

# Plot at final time step, center radius
T_plot = T[-1,:]
ohmic_plot = ohmic[-1,:]
radiation_plot = radiation[-1,:]

plt.figure(figsize=(8,6))
plt.plot(T_plot, ohmic_plot, label='Ohmic Heating')
plt.plot(T_plot, radiation_plot, label='Radiative Losses')
plt.xlabel('Temperature [eV]')
plt.ylabel('Power density [W/m$^3$]')
plt.title('Ohmic Heating and Radiative Losses vs Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```