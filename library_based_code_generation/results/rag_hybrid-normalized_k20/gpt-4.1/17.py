```python
import numpy as np
import matplotlib.pyplot as plt
from bluemira.geometry import make_CS_coil, make_enclosure_2d
from bluemira.mesh import MeshOptions, create_mesh, convert_mesh_to_fenics
from bluemira.components import MagneticComponent, UniverseComponent
from bluemira.solver import MagnetostaticSolver
from bluemira.sources import CoilCurrentSource
from bluemira.utils import theoretical_Bz_on_axis, theoretical_Br_off_axis

# Parameters
coil_radius = 0.2         # meters
coil_current = 1e4        # Amperes
coil_center = (0.0, 0.0)  # (x, y)
coil_width = 0.02         # meters
enclosure_radius = 0.5    # meters

# Geometry
coil = make_CS_coil(center=coil_center, radius=coil_radius, width=coil_width)
enclosure = make_enclosure_2d(center=coil_center, radius=enclosure_radius)

# Mesh options
coil_mesh_opts = MeshOptions(element_size=0.005)
enclosure_mesh_opts = MeshOptions(element_size=0.02)

# Components
universe = UniverseComponent()
enclosure_comp = MagneticComponent(geometry=enclosure, mesh_options=enclosure_mesh_opts, name="enclosure")
coil_comp = MagneticComponent(geometry=coil, mesh_options=coil_mesh_opts, name="coil", current=coil_current)

# Mesh creation and conversion
mesh = create_mesh([enclosure_comp, coil_comp], universe)
fenics_mesh, subdomains, boundaries = convert_mesh_to_fenics(mesh)

# Magnetostatic solver setup
solver = MagnetostaticSolver(fenics_mesh, subdomains, boundaries)
source = CoilCurrentSource(coil_comp, current=coil_current)
solver.add_source(source)

# Plot source term
plt.figure()
solver.plot_source_term()
plt.title("Source Term (Current Density)")
plt.show()

# Solve magnetostatic problem
A = solver.solve()
B = solver.compute_magnetic_field(A)

# Theoretical and calculated Bz along z-axis (r=0)
z_vals = np.linspace(-0.3, 0.3, 100)
r_vals = np.zeros_like(z_vals)
Bz_calc = []
Bz_theory = []
for z in z_vals:
    point = np.array([0.0, 0.0, z])
    Bz_calc.append(solver.evaluate_Bz_at_point(point))
    Bz_theory.append(theoretical_Bz_on_axis(coil_radius, coil_current, z))

plt.figure()
plt.plot(z_vals, Bz_calc, label='Calculated Bz')
plt.plot(z_vals, Bz_theory, '--', label='Theoretical Bz')
plt.xlabel('z (m)')
plt.ylabel('Bz (T)')
plt.title('Bz along z-axis (r=0)')
plt.legend()
plt.show()

plt.figure()
plt.plot(z_vals, np.array(Bz_calc) - np.array(Bz_theory))
plt.xlabel('z (m)')
plt.ylabel('Bz Error (T)')
plt.title('Difference: Calculated - Theoretical Bz along z-axis')
plt.show()

# Theoretical and calculated Br along radial path at z = 0.05 m
z_offset = 0.05
r_vals = np.linspace(0, 0.4, 100)
Br_calc = []
Br_theory = []
for r in r_vals:
    point = np.array([r, 0.0, z_offset])
    Br_calc.append(solver.evaluate_Br_at_point(point))
    Br_theory.append(theoretical_Br_off_axis(coil_radius, coil_current, r, z_offset))

plt.figure()
plt.plot(r_vals, Br_calc, label='Calculated Br')
plt.plot(r_vals, Br_theory, '--', label='Theoretical Br')
plt.xlabel('r (m)')
plt.ylabel('Br (T)')
plt.title(f'Br along r at z={z_offset} m')
plt.legend()
plt.show()

plt.figure()
plt.plot(r_vals, np.array(Br_calc) - np.array(Br_theory))
plt.xlabel('r (m)')
plt.ylabel('Br Error (T)')
plt.title(f'Difference: Calculated - Theoretical Br at z={z_offset} m')
plt.show()
```