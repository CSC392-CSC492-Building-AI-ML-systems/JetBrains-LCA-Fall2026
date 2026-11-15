```python
import numpy as np
import matplotlib.pyplot as plt
from bluemira.geometry import Geometry, Coil, Enclosure
from bluemira.component import Component
from bluemira.mesh import Mesh
from bluemira.fem import MagnetostaticSolver
from bluemira.utils import convert_mesh_to_fenics
import dolfin as df

# Parameters
coil_radius = 0.05  # meters
coil_length = 0.01  # meters
coil_current = 10.0  # Amperes
coil_turns = 100
enclosure_radius = 0.2  # meters
enclosure_length = 0.2  # meters

# Create geometry
geometry = Geometry()
coil = Coil(
    center=[0, 0, 0],
    radius=coil_radius,
    length=coil_length,
    current=coil_current,
    turns=coil_turns,
    axis=[0, 0, 1]
)
enclosure = Enclosure(
    center=[0, 0, 0],
    radius=enclosure_radius,
    length=enclosure_length,
    axis=[0, 0, 1]
)
geometry.add(coil)
geometry.add(enclosure)

# Set mesh options
coil_mesh_opts = {'resolution': 0.005}
enclosure_mesh_opts = {'resolution': 0.02}
geometry.set_mesh_options(coil, coil_mesh_opts)
geometry.set_mesh_options(enclosure, enclosure_mesh_opts)

# Create components
universe = Component('universe')
enclosure_comp = Component('enclosure', geometry=enclosure)
coil_comp = Component('coil', geometry=coil)

# Create mesh and convert for FEniCS
mesh = Mesh(geometry)
mesh.generate()
fenics_mesh, subdomains, boundaries = convert_mesh_to_fenics(mesh)

# Magnetostatic solver setup
solver = MagnetostaticSolver(fenics_mesh, subdomains=subdomains, boundaries=boundaries)

# Define source term (current density in coil region)
V = df.FunctionSpace(fenics_mesh, 'CG', 1)
coil_marker = 2  # Assuming coil is marked as 2 in subdomains
J = df.Function(V)
J_vec = df.Function(df.VectorFunctionSpace(fenics_mesh, 'CG', 1))

# Set current density in coil region
coil_area = np.pi * coil_radius**2
current_density = coil_current * coil_turns / coil_area
J_array = J.vector().get_local()
dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))
for i, x in enumerate(dof_coords):
    r = np.sqrt(x[0]**2 + x[1]**2)
    z = x[2]
    if (r <= coil_radius) and (np.abs(z) <= coil_length/2):
        J_array[i] = current_density
    else:
        J_array[i] = 0.0
J.vector().set_local(J_array)
J.vector().apply('insert')

# Set vector current density (along z)
J_vec_array = J_vec.vector().get_local().reshape((-1, 3))
for i, x in enumerate(dof_coords):
    r = np.sqrt(x[0]**2 + x[1]**2)
    z = x[2]
    if (r <= coil_radius) and (np.abs(z) <= coil_length/2):
        J_vec_array[i, 2] = current_density
J_vec.vector().set_local(J_vec_array.flatten())
J_vec.vector().apply('insert')

# Plot source term
plt.figure()
p = df.plot(J, title='Current Density in Coil')
plt.colorbar(p)
plt.show()

# Solve magnetostatic problem
A = solver.solve(J_vec)

# Calculate magnetic field B = curl(A)
W = df.VectorFunctionSpace(fenics_mesh, 'CG', 1)
B = df.project(df.curl(A), W)

# Theoretical field along z-axis (on axis of solenoid)
mu0 = 4 * np.pi * 1e-7
def Bz_theoretical(z):
    R = coil_radius
    L = coil_length
    N = coil_turns
    I = coil_current
    z1 = z - L/2
    z2 = z + L/2
    return (mu0 * N * I / 2) * ( (z2/np.sqrt(R**2 + z2**2)) - (z1/np.sqrt(R**2 + z1**2)) ) / L

# Sample points along z-axis (r=0)
z_points = np.linspace(-0.1, 0.1, 100)
Bz_calc = []
Bz_theory = []
for z in z_points:
    point = df.Point(0, 0, z)
    try:
        B_vec = B(point)
        Bz_calc.append(B_vec[2])
    except:
        Bz_calc.append(np.nan)
    Bz_theory.append(Bz_theoretical(z))

Bz_calc = np.array(Bz_calc)
Bz_theory = np.array(Bz_theory)

plt.figure()
plt.plot(z_points, Bz_calc, label='Calculated $B_z$')
plt.plot(z_points, Bz_theory, '--', label='Theoretical $B_z$')
plt.xlabel('z [m]')
plt.ylabel('$B_z$ [T]')
plt.legend()
plt.title('Magnetic Field Along z-axis')
plt.show()

plt.figure()
plt.plot(z_points, Bz_calc - Bz_theory)
plt.xlabel('z [m]')
plt.ylabel('Difference [T]')
plt.title('Difference: Calculated - Theoretical $B_z$ Along z-axis')
plt.show()

# Radial path at z = 0 (midplane)
r_points = np.linspace(0, 0.1, 100)
Br_calc = []
Br_theory = []
z_offset = 0.0
for r in r_points:
    point = df.Point(r, 0, z_offset)
    try:
        B_vec = B(point)
        Br_calc.append(B_vec[2])  # On axis, Bz is dominant
    except:
        Br_calc.append(np.nan)
    Br_theory.append(Bz_theoretical(z_offset))  # Approximate as on axis

Br_calc = np.array(Br_calc)
Br_theory = np.array(Br_theory)

plt.figure()
plt.plot(r_points, Br_calc, label='Calculated $B_z$ (radial path)')
plt.plot(r_points, Br_theory, '--', label='Theoretical $B_z$ (on axis)')
plt.xlabel('r [m]')
plt.ylabel('$B_z$ [T]')
plt.legend()
plt.title('Magnetic Field Along Radial Path at z=0')
plt.show()

plt.figure()
plt.plot(r_points, Br_calc - Br_theory)
plt.xlabel('r [m]')
plt.ylabel('Difference [T]')
plt.title('Difference: Calculated - Theoretical $B_z$ Along Radial Path at z=0')
plt.show()
```