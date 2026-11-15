```python
import numpy as np
import matplotlib.pyplot as plt
from bluemira.geometry import Coil, Enclosure, Universe
from bluemira.mesh import Mesh
from bluemira.components import Component
from bluemira.solver import MagnetostaticSolver
from bluemira.utils import convert_mesh_to_fenics
from fenics import *

# --- Parameters ---
coil_radius = 0.05         # meters
coil_length = 0.1          # meters
coil_current = 10.0        # Amperes
coil_center = (0.0, 0.0, 0.0)
enclosure_radius = 0.2     # meters
enclosure_length = 0.4     # meters
mesh_size_coil = 0.005
mesh_size_enclosure = 0.02

# --- Geometry ---
coil = Coil(
    center=coil_center,
    radius=coil_radius,
    length=coil_length,
    current=coil_current
)
enclosure = Enclosure(
    center=coil_center,
    radius=enclosure_radius,
    length=enclosure_length
)

# --- Mesh options ---
coil_mesh_opts = {'mesh_size': mesh_size_coil}
enclosure_mesh_opts = {'mesh_size': mesh_size_enclosure}

# --- Components ---
universe = Universe()
enclosure_comp = Component(geometry=enclosure, mesh_options=enclosure_mesh_opts)
coil_comp = Component(geometry=coil, mesh_options=coil_mesh_opts)

universe.add_component(enclosure_comp)
universe.add_component(coil_comp)

# --- Mesh creation ---
mesh = Mesh(universe)
mesh.generate()
fenics_mesh = convert_mesh_to_fenics(mesh)

# --- Magnetostatic Solver ---
solver = MagnetostaticSolver(mesh=fenics_mesh)

# --- Source term (current density in coil) ---
V = FunctionSpace(fenics_mesh, 'P', 1)
x = SpatialCoordinate(fenics_mesh)

def current_density_expr(x, y, z):
    r = np.sqrt((x - coil_center[0])**2 + (y - coil_center[1])**2)
    zmin = coil_center[2] - coil_length/2
    zmax = coil_center[2] + coil_length/2
    if (r <= coil_radius) and (zmin <= z <= zmax):
        area = np.pi * coil_radius**2
        return coil_current / area
    else:
        return 0.0

J = Expression('((sqrt(pow(x[0],2) + pow(x[1],2)) <= R) && (x[2] >= ZMIN) && (x[2] <= ZMAX)) ? I/(pi*pow(R,2)) : 0.0',
               degree=2, R=coil_radius, ZMIN=coil_center[2]-coil_length/2, ZMAX=coil_center[2]+coil_length/2, I=coil_current)

# Plot source term
J_func = interpolate(J, V)
plot(J_func)
plt.title("Current Density Source Term")
plt.colorbar()
plt.show()

# --- Solve magnetostatic problem ---
A = solver.solve(J)

# --- Calculate magnetic field ---
B = solver.compute_B(A)

# --- Theoretical B field along z-axis (on axis of coil) ---
def theoretical_Bz_on_axis(z):
    mu0 = 4 * np.pi * 1e-7
    N = 1  # single turn
    R = coil_radius
    L = coil_length
    z1 = z - L/2
    z2 = z + L/2
    return (mu0 * coil_current * R**2 / 2) * (1.0/((R**2 + z1**2)**1.5) - 1.0/((R**2 + z2**2)**1.5))

z_vals = np.linspace(-0.15, 0.15, 100)
Bz_calc = []
Bz_theory = []

for z in z_vals:
    point = np.array([0.0, 0.0, z])
    B_vec = solver.evaluate_B_at_point(B, point)
    Bz_calc.append(B_vec[2])
    Bz_theory.append(theoretical_Bz_on_axis(z))

plt.figure()
plt.plot(z_vals, Bz_calc, label='Calculated $B_z$')
plt.plot(z_vals, Bz_theory, '--', label='Theoretical $B_z$')
plt.xlabel('z (m)')
plt.ylabel('$B_z$ (T)')
plt.legend()
plt.title('Magnetic Field Along z-axis')
plt.show()

plt.figure()
plt.plot(z_vals, np.array(Bz_calc) - np.array(Bz_theory))
plt.xlabel('z (m)')
plt.ylabel('Difference (T)')
plt.title('Difference: Calculated - Theoretical $B_z$ Along z-axis')
plt.show()

# --- Radial path at z-offset ---
z_offset = 0.0
r_vals = np.linspace(0, 0.15, 100)
Br_calc = []
Br_theory = []

def theoretical_Bz_off_axis(r, z):
    # Approximate using on-axis formula for small r, otherwise use Biot-Savart or elliptic integrals (not implemented here)
    return theoretical_Bz_on_axis(z) * (1 - (r**2)/(2*coil_radius**2))  # crude approximation

for r in r_vals:
    point = np.array([r, 0.0, z_offset])
    B_vec = solver.evaluate_B_at_point(B, point)
    Br_calc.append(B_vec[2])
    Br_theory.append(theoretical_Bz_off_axis(r, z_offset))

plt.figure()
plt.plot(r_vals, Br_calc, label='Calculated $B_z$')
plt.plot(r_vals, Br_theory, '--', label='Theoretical $B_z$ (approx)')
plt.xlabel('r (m)')
plt.ylabel('$B_z$ (T)')
plt.legend()
plt.title('Magnetic Field Along Radial Path at z=0')
plt.show()

plt.figure()
plt.plot(r_vals, np.array(Br_calc) - np.array(Br_theory))
plt.xlabel('r (m)')
plt.ylabel('Difference (T)')
plt.title('Difference: Calculated - Theoretical $B_z$ Along Radial Path')
plt.show()
```