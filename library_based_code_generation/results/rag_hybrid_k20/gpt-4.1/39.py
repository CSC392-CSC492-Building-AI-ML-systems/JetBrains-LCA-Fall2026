```python
import numpy as np
import matplotlib.pyplot as plt
from burnman import minerals, solution_model, seismic, composition, seismic as bm_seismic
from burnman.seismic import Seismic1DModel

# 1. Two minerals mixed in simple mole fractions
olivine = minerals.SLB_2011.olivine()
wadsleyite = minerals.SLB_2011.wadsleyite()
mole_fractions_2 = [0.6, 0.4]
minerals_2 = [olivine, wadsleyite]

# 2. Mix of three minerals
ringwoodite = minerals.SLB_2011.ringwoodite()
mole_fractions_3 = [0.5, 0.3, 0.2]
minerals_3 = [olivine, wadsleyite, ringwoodite]

# 3. Using preset solutions
pyroxene_solution = solution_model.SLB_2011.pyroxene()
garnet_solution = solution_model.SLB_2011.garnet()
mole_fractions_sol = [0.7, 0.3]
minerals_sol = [pyroxene_solution, garnet_solution]

# 4. Defining a custom solution
custom_endmembers = [minerals.SLB_2011.olivine(), minerals.SLB_2011.ringwoodite()]
custom_mole_fractions = [0.8, 0.2]
custom_solution = solution_model.IdealSolution(custom_endmembers)
minerals_custom = [custom_solution]
mole_fractions_custom = [1.0]

# Pressure and temperature arrays
pressures = np.linspace(10e9, 25e9, 30)  # Pa
temperatures = 1873. + 0.5 * (pressures - 10e9) / 1e9 * 100  # K, simple geotherm

# Reference seismic model (e.g., PREM)
reference_model = Seismic1DModel('PREM')
ref_pressures = pressures / 1e9  # GPa for PREM
ref_vs = reference_model.vs(ref_pressures)
ref_vphi = reference_model.vphi(ref_pressures)
ref_rho = reference_model.rho(ref_pressures)
ref_temp = 1873. + 0.5 * (pressures - 10e9) / 1e9 * 100  # K

def compute_properties(minerals_list, mole_fractions, pressures, temperatures):
    vs, vphi, rho = [], [], []
    for p, t in zip(pressures, temperatures):
        assemblage = seismic.Composite(minerals_list, mole_fractions)
        assemblage.set_state(p, t)
        v = assemblage.seismic_velocities()
        vs.append(v['vs'])
        vphi.append(v['vphi'])
        rho.append(assemblage.density())
    return np.array(vs), np.array(vphi), np.array(rho)

# Compute for each example
vs_2, vphi_2, rho_2 = compute_properties(minerals_2, mole_fractions_2, pressures, temperatures)
vs_3, vphi_3, rho_3 = compute_properties(minerals_3, mole_fractions_3, pressures, temperatures)
vs_sol, vphi_sol, rho_sol = compute_properties(minerals_sol, mole_fractions_sol, pressures, temperatures)
vs_custom, vphi_custom, rho_custom = compute_properties(minerals_custom, mole_fractions_custom, pressures, temperatures)

# Compute misfits
def misfit(computed, reference):
    return np.sqrt(np.mean((computed - reference) ** 2))

misfit_vs_2 = misfit(vs_2, ref_vs)
misfit_vs_3 = misfit(vs_3, ref_vs)
misfit_vs_sol = misfit(vs_sol, ref_vs)
misfit_vs_custom = misfit(vs_custom, ref_vs)

misfit_vphi_2 = misfit(vphi_2, ref_vphi)
misfit_vphi_3 = misfit(vphi_3, ref_vphi)
misfit_vphi_sol = misfit(vphi_sol, ref_vphi)
misfit_vphi_custom = misfit(vphi_custom, ref_vphi)

misfit_rho_2 = misfit(rho_2, ref_rho)
misfit_rho_3 = misfit(rho_3, ref_rho)
misfit_rho_sol = misfit(rho_sol, ref_rho)
misfit_rho_custom = misfit(rho_custom, ref_rho)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Vs
axs[0, 0].plot(pressures/1e9, vs_2, label='2 minerals')
axs[0, 0].plot(pressures/1e9, vs_3, label='3 minerals')
axs[0, 0].plot(pressures/1e9, vs_sol, label='Preset solution')
axs[0, 0].plot(pressures/1e9, vs_custom, label='Custom solution')
axs[0, 0].plot(pressures/1e9, ref_vs, 'k--', label='Reference')
axs[0, 0].set_xlabel('Pressure (GPa)')
axs[0, 0].set_ylabel('Vs (km/s)')
axs[0, 0].set_title('Shear Velocity (Vs)')
axs[0, 0].legend()

# Vphi
axs[0, 1].plot(pressures/1e9, vphi_2, label='2 minerals')
axs[0, 1].plot(pressures/1e9, vphi_3, label='3 minerals')
axs[0, 1].plot(pressures/1e9, vphi_sol, label='Preset solution')
axs[0, 1].plot(pressures/1e9, vphi_custom, label='Custom solution')
axs[0, 1].plot(pressures/1e9, ref_vphi, 'k--', label='Reference')
axs[0, 1].set_xlabel('Pressure (GPa)')
axs[0, 1].set_ylabel('Vphi (km/s)')
axs[0, 1].set_title('Compressional Velocity (Vphi)')
axs[0, 1].legend()

# Density
axs[1, 0].plot(pressures/1e9, rho_2, label='2 minerals')
axs[1, 0].plot(pressures/1e9, rho_3, label='3 minerals')
axs[1, 0].plot(pressures/1e9, rho_sol, label='Preset solution')
axs[1, 0].plot(pressures/1e9, rho_custom, label='Custom solution')
axs[1, 0].plot(pressures/1e9, ref_rho, 'k--', label='Reference')
axs[1, 0].set_xlabel('Pressure (GPa)')
axs[1, 0].set_ylabel('Density (kg/m³)')
axs[1, 0].set_title('Density')
axs[1, 0].legend()

# Geotherm
axs[1, 1].plot(pressures/1e9, temperatures, label='Geotherm')
axs[1, 1].plot(pressures/1e9, ref_temp, 'k--', label='Reference')
axs[1, 1].set_xlabel('Pressure (GPa)')
axs[1, 1].set_ylabel('Temperature (K)')
axs[1, 1].set_title('Geotherm')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('seismic_comparison.png')
```