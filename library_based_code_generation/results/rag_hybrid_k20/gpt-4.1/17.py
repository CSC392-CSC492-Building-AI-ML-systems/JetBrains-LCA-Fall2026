```python
import numpy as np
import matplotlib.pyplot as plt
from bluemira.geometry import BluemiraGeoEncoder, make_pf_coil_path
from bluemira.components import MagneticComponent
from bluemira.mesh import create_mesh, convert_mesh_to_fenics
from bluemira.solvers import systems_code_solver
from bluemira.constraints import MagneticConstraintSet, MagneticConstraint
from fenics import FunctionSpace, Function, plot, Point

# Parameters
coil_radius = 0.2         # meters
coil_current = 1e4        # Amperes
coil_center = (0.0, 0.0, 0.0)
coil_length = 0.05        # meters (thickness in z)
enclosure_radius = 0.5    # meters
enclosure_height = 0.5    # meters
mesh_size_coil = 0.01     # meters
mesh_size_enclosure = 0.05 # meters

# Create geometry
encoder = BluemiraGeoEncoder()
coil_path = make_pf_coil_path(radius=coil_radius, center=coil_center, length=coil_length)
coil_geom = encoder.encode_curve(coil_path)
enclosure_geom = encoder.encode_cylinder(center=(0,0,0), radius=enclosure_radius, height=enclosure_height)

# Set mesh options
coil_mesh_opts = {'mesh_size': mesh_size_coil}
enclosure_mesh_opts = {'mesh_size': mesh_size_enclosure}

# Create components
universe = MagneticComponent(name='universe')
enclosure = MagneticComponent(name='enclosure', geometry=enclosure_geom, mesh_options=enclosure_mesh_opts)
coil = MagneticComponent(name='coil', geometry=coil_geom, mesh_options=coil_mesh_opts, current=coil_current)

# Add components to universe
universe.add_child(enclosure)
enclosure.add_child(coil)

# Create mesh and convert for FEniCS
mesh = create_mesh(universe)
fenics_mesh = convert_mesh_to_fenics(mesh)

# Set up FEniCS function space
V = FunctionSpace(fenics_mesh, 'P', 1)

# Define source term (current density in coil region)
class CoilCurrentDensity:
    def __init__(self, center, radius, length, current):
        self.center = np.array(center)
        self.radius = radius
        self.length = length
        self.current = current
        self.mu0 = 4 * np.pi * 1e-7

    def __call__(self, x, y, z):
        r = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        z_rel = z - self.center[2]
        if r <= self.radius and abs(z_rel) <= self.length/2:
            area = np.pi * self.radius**2
            return self.current / area
        else:
            return 0.0

coil_j = CoilCurrentDensity(coil_center, coil_radius, coil_length, coil_current)

# Create FEniCS source term function
source = Function(V)
coords = fenics_mesh.coordinates()
source_values = np.array([coil_j(x, y, z) for x, y, z in coords])
source.vector()[:] = source_values

# Plot source term
plt.figure()
plot(source)
plt.title("Current Density Source Term")
plt.colorbar()
plt.show()

# Instantiate magnetostatic solver and solve
solver = systems_code_solver(universe, mesh=fenics_mesh, source=source)
A = solver.solve()  # Magnetic vector potential

# Calculate magnetic field B = curl(A)
B = solver.calculate_magnetic_field(A)

# Theoretical Bz on axis (z-axis)
def theoretical_Bz_on_axis(z, coil_radius, coil_current, coil_center, coil_length):
    mu0 = 4 * np.pi * 1e-7
    z_rel = z - coil_center[2]
    R = coil_radius
    I = coil_current
    # For a thin coil at z=0, Bz = mu0*I*R^2 / (2*(R^2 + z^2)^(3/2))
    return mu0 * I * R**2 / (2 * (R**2 + z_rel**2)**(1.5))

z_vals = np.linspace(-0.2, 0.2, 100)
Bz_calc = []
Bz_theory = []
for z in z_vals:
    pt = Point(0.0, 0.0, z)
    B_vec = B(pt)
    Bz_calc.append(B_vec[2])
    Bz_theory.append(theoretical_Bz_on_axis(z, coil_radius, coil_current, coil_center, coil_length))

# Plot comparison along z-axis
plt.figure()
plt.plot(z_vals, Bz_calc, label='Calculated Bz')
plt.plot(z_vals, Bz_theory, label='Theoretical Bz', linestyle='--')
plt.xlabel('z (m)')
plt.ylabel('Bz (T)')
plt.title('Bz along z-axis')
plt.legend()
plt.show()

plt.figure()
plt.plot(z_vals, np.array(Bz_calc) - np.array(Bz_theory))
plt.xlabel('z (m)')
plt.ylabel('Bz Error (T)')
plt.title('Difference: Calculated - Theoretical Bz along z-axis')
plt.show()

# Radial path at z-offset
z_offset = 0.0
r_vals = np.linspace(0, 0.4, 100)
Br_calc = []
Br_theory = []
for r in r_vals:
    pt = Point(r, 0.0, z_offset)
    B_vec = B(pt)
    Br_calc.append(B_vec[0])
    # Theoretical Br for a thin coil is zero on axis, but off-axis is more complex.
    # For simplicity, use Biot-Savart for a circular loop in the plane z=0:
    # Br = mu0*I*z/(4*pi) * integral over phi of (cos(phi) - r/R) / ((R^2 + r^2 - 2*R*r*cos(phi) + z^2)^(3/2))
    # For z=0, Br=0 by symmetry, so we plot calculated only.
    Br_theory.append(0.0)

plt.figure()
plt.plot(r_vals, Br_calc, label='Calculated Br')
plt.plot(r_vals, Br_theory, label='Theoretical Br', linestyle='--')
plt.xlabel('r (m)')
plt.ylabel('Br (T)')
plt.title('Br along radial path at z=0')
plt.legend()
plt.show()

plt.figure()
plt.plot(r_vals, np.array(Br_calc) - np.array(Br_theory))
plt.xlabel('r (m)')
plt.ylabel('Br Error (T)')
plt.title('Difference: Calculated - Theoretical Br along radial path at z=0')
plt.show()
```