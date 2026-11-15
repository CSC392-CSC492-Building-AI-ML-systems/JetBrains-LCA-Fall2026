```python
import numpy as np
import matplotlib.pyplot as plt
from burnman import minerals, solutionmodels, seismic, composition, seismicmodels, tools

# 1. Two minerals mixed in simple mole fractions
olivine = minerals.SLB_2011.olivine()
wadsleyite = minerals.SLB_2011.wadsleyite()
minerals_2 = [olivine, wadsleyite]
mole_fractions_2 = [0.6, 0.4]

# 2. Mix of three minerals
ringwoodite = minerals.SLB_2011.ringwoodite()
minerals_3 = [olivine, wadsleyite, ringwoodite]
mole_fractions_3 = [0.5, 0.3, 0.2]

# 3. Using preset solutions
pyroxene_solution = solutionmodels.SLB_2011.pyroxene()
garnet_solution = solutionmodels.SLB_2011.garnet()
minerals_preset = [pyroxene_solution, garnet_solution]
mole_fractions_preset = [0.7, 0.3]

# 4. Defining a custom solution
custom_components = [minerals.SLB_2011.olivine(), minerals.SLB_2011.wadsleyite()]
custom_mole_fractions = [0.8, 0.2]
custom_solution = solutionmodels.IdealSolution(custom_components, custom_mole_fractions)
minerals_custom = [custom_solution]
mole_fractions_custom = [1.0]

# Choose one composite for demonstration (e.g., three-mineral mix)
minerals_list = minerals_3
mole_fractions = mole_fractions_3

# Set up pressure and temperature arrays
pressures = np.linspace(10e9, 25e9, 30)  # Pa
temperatures = 1600. + 0.5 * (pressures - 10e9) / 1e9 * 100  # K, simple geotherm

# Compute seismic velocities and properties
vs = []
vphi = []
density = []
for p, t in zip(pressures, temperatures):
    for m in minerals_list:
        m.set_state(pressure=p, temperature=t)
    mixture = composition.Composite(minerals_list, mole_fractions)
    mixture.set_state(pressure=p, temperature=t)
    density.append(mixture.density())
    vs.append(seismic.seismic_Vs(mixture))
    vphi.append(seismic.seismic_Vphi(mixture))

vs = np.array(vs)
vphi = np.array(vphi)
density = np.array(density)

# Reference seismic model (e.g., PREM)
prem = seismicmodels.PREM()
ref_vs = []
ref_vphi = []
ref_density = []
for p, t in zip(pressures, temperatures):
    depth = prem.depth_from_pressure(p)
    ref_vs.append(prem.Vs(depth))
    ref_vphi.append(prem.Vphi(depth))
    ref_density.append(prem.density(depth))

ref_vs = np.array(ref_vs)
ref_vphi = np.array(ref_vphi)
ref_density = np.array(ref_density)

# Calculate misfit
misfit_vs = np.sqrt(np.mean((vs - ref_vs) ** 2))
misfit_vphi = np.sqrt(np.mean((vphi - ref_vphi) ** 2))
misfit_density = np.sqrt(np.mean((density - ref_density) ** 2))

print(f"RMS misfit Vs: {misfit_vs:.3f} km/s")
print(f"RMS misfit Vphi: {misfit_vphi:.3f} km/s")
print(f"RMS misfit density: {misfit_density:.3f} kg/m^3")

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(pressures / 1e9, vs, label='Computed')
axs[0, 0].plot(pressures / 1e9, ref_vs, label='PREM')
axs[0, 0].set_xlabel('Pressure (GPa)')
axs[0, 0].set_ylabel('Vs (km/s)')
axs[0, 0].legend()
axs[0, 0].set_title('Shear Velocity (Vs)')

axs[0, 1].plot(pressures / 1e9, vphi, label='Computed')
axs[0, 1].plot(pressures / 1e9, ref_vphi, label='PREM')
axs[0, 1].set_xlabel('Pressure (GPa)')
axs[0, 1].set_ylabel('Vphi (km/s)')
axs[0, 1].legend()
axs[0, 1].set_title('Compressional Velocity (Vphi)')

axs[1, 0].plot(pressures / 1e9, density, label='Computed')
axs[1, 0].plot(pressures / 1e9, ref_density, label='PREM')
axs[1, 0].set_xlabel('Pressure (GPa)')
axs[1, 0].set_ylabel('Density (kg/m^3)')
axs[1, 0].legend()
axs[1, 0].set_title('Density')

axs[1, 1].plot(pressures / 1e9, temperatures, label='Geotherm')
axs[1, 1].set_xlabel('Pressure (GPa)')
axs[1, 1].set_ylabel('Temperature (K)')
axs[1, 1].legend()
axs[1, 1].set_title('Geotherm')

plt.tight_layout()
plt.savefig('seismic_properties_vs_prem.png')
plt.close()
```