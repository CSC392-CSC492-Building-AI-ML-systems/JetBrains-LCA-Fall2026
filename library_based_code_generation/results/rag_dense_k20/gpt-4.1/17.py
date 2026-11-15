```python
import numpy as np
import matplotlib.pyplot as plt
from bluemira.geometry import Rectangle, Circle, GeometryCollection
from bluemira.mesh import MeshOptions, generate_mesh
from bluemira.components import MagneticComponent, UniverseComponent
from bluemira.fem import FemMagnetostatic2d
from bluemira.fem.mesh_conversion import mesh_to_fenics
from bluemira.fem.sources import CurrentDensity2D
from bluemira.fem.utils import plot_function

# Parameters
coil_center = (0.0, 0.0)
coil_radius = 0.1
coil_width = 0.02
enclosure_size = 1.0
current = 1000.0  # Amperes
coil_area = np.pi * ((coil_radius + coil_width/2)**2 - (coil_radius - coil_width/2)**2)
current_density = current / coil_area

# Geometry
coil_outer = Circle(center=coil_center, radius=coil_radius + coil_width/2)
coil_inner = Circle(center=coil_center, radius=coil_radius - coil_width/2)
coil_geom = coil_outer - coil_inner

enclosure_geom = Rectangle(
    x_min=-enclosure_size/2, x_max=enclosure_size/2,
    y_min=-enclosure_size/2, y_max=enclosure_size/2
)

geometry = GeometryCollection([enclosure_geom, coil_geom])

# Mesh options
enclosure_mesh_opts = MeshOptions(max_element_size=0.05)
coil_mesh_opts = MeshOptions(max_element_size=0.01)

# Components
universe = UniverseComponent(geometry=enclosure_geom, mesh_options=enclosure_mesh_opts)
enclosure = MagneticComponent(geometry=enclosure_geom, mesh_options=enclosure_mesh_opts, name="enclosure")
coil = MagneticComponent(geometry=coil_geom, mesh_options=coil_mesh_opts, name="coil", current=current)

# Mesh generation
mesh = generate_mesh([enclosure, coil], universe=universe)
fenics_mesh, subdomains, boundaries = mesh_to_fenics(mesh)

# Magnetostatic solver setup
solver = FemMagnetostatic2d(fenics_mesh, subdomains=subdomains, boundaries=boundaries)

# Source term (current density in coil region)
source = CurrentDensity2D(subdomains, coil.subdomain_id, current_density)
plot_function(source.function, title="Current Density Source Term")

# Solve magnetostatic problem
A = solver.solve(source=source)
B = solver.compute_B(A)

# Theoretical B field along z-axis (center of coil, infinite solenoid approx)
mu0 = 4 * np.pi * 1e-7
B_theory_center = mu0 * current / (2 * coil_radius)

# Compare along z-axis (y=0)
z_vals = np.linspace(-coil_radius, coil_radius, 100)
B_calc_z = []
B_theory_z = []
for z in z_vals:
    pt = (0.0, z)
    B_vec = B(pt)
    B_calc_z.append(B_vec[1])  # y-component (since coil is in x-y plane)
    # Theoretical: on axis of a circular loop
    R = coil_radius
    denom = (R**2 + z**2)**(1.5)
    Bz = mu0 * current * R**2 / (2 * denom)
    B_theory_z.append(Bz)

B_calc_z = np.array(B_calc_z)
B_theory_z = np.array(B_theory_z)
diff_z = B_calc_z - B_theory_z

plt.figure()
plt.plot(z_vals, B_calc_z, label='Calculated Bz')
plt.plot(z_vals, B_theory_z, label='Theoretical Bz', linestyle='--')
plt.xlabel('z (m)')
plt.ylabel('Bz (T)')
plt.title('Magnetic Field along z-axis')
plt.legend()
plt.show()

plt.figure()
plt.plot(z_vals, diff_z)
plt.xlabel('z (m)')
plt.ylabel('Difference (T)')
plt.title('Difference: Calculated - Theoretical Bz along z-axis')
plt.show()

# Compare along radial path at z = 0.05 m
z_offset = 0.05
r_vals = np.linspace(0, coil_radius*1.5, 100)
B_calc_r = []
B_theory_r = []
for r in r_vals:
    pt = (r, z_offset)
    B_vec = B(pt)
    B_calc_r.append(B_vec[1])  # y-component
    # Theoretical: off-axis field of a loop (approximate with Biot-Savart)
    R = coil_radius
    z = z_offset
    denom = (R**2 + z**2)**(1.5)
    Bz = mu0 * current * R**2 / (2 * denom)
    B_theory_r.append(Bz)

B_calc_r = np.array(B_calc_r)
B_theory_r = np.array(B_theory_r)
diff_r = B_calc_r - B_theory_r

plt.figure()
plt.plot(r_vals, B_calc_r, label='Calculated Bz')
plt.plot(r_vals, B_theory_r, label='Theoretical Bz', linestyle='--')
plt.xlabel('r (m)')
plt.ylabel('Bz (T)')
plt.title(f'Magnetic Field along r at z={z_offset} m')
plt.legend()
plt.show()

plt.figure()
plt.plot(r_vals, diff_r)
plt.xlabel('r (m)')
plt.ylabel('Difference (T)')
plt.title(f'Difference: Calculated - Theoretical Bz along r at z={z_offset} m')
plt.show()
```