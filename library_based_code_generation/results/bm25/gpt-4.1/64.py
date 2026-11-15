```python
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core.post import read_binary
import numpy as np

# Launch MAPDL
mapdl = launch_mapdl()

# Set units to US Customary (inches)
mapdl.prep7()
mapdl.units('INCH')

# Define material properties (e.g., Steel)
mapdl.mp('EX', 1, 3e7)      # Young's modulus in psi
mapdl.mp('NUXY', 1, 0.3)    # Poisson's ratio

# Define element type (Solid185)
mapdl.et(1, 'SOLID185')

# Create geometry: Cylinder with hemispherical ends (pressure vessel)
# Parameters
radius = 12      # inches
length = 48      # inches (cylinder length, not including hemispheres)
thickness = 0.5  # inches

# Create the main cylinder
mapdl.cylind(radius, length)

# Create hemispherical ends
mapdl.cylind(radius, thickness, z1=0, z2=thickness, theta1=0, theta2=360)
mapdl.cylind(radius, thickness, z1=length, z2=length+thickness, theta1=0, theta2=360)

# Subtract inner volume to create shell
mapdl.cylind(radius-thickness, length)
mapdl.cylind(radius-thickness, thickness, z1=0, z2=thickness, theta1=0, theta2=360)
mapdl.cylind(radius-thickness, thickness, z1=length, z2=length+thickness, theta1=0, theta2=360)
mapdl.vsbv('ALL', 'ALL', 'ALL', 'ALL')

# Mesh the geometry
mapdl.esize(1)
mapdl.vmesh('ALL')

# Apply boundary conditions
# Fix one end (z=0)
mapdl.nsel('S', 'LOC', 'Z', 0)
mapdl.d('ALL', 'ALL')
mapdl.nsel('ALL')

# Apply internal pressure to inner surfaces
pressure = 150  # psi
mapdl.nsel('S', 'LOC', 'RADIUS', radius-thickness)
mapdl.sfe('ALL', 1, 'PRES', '', pressure)
mapdl.nsel('ALL')

# Solve
mapdl.finish()
mapdl.run('/SOLU')
mapdl.antype('STATIC')
mapdl.solve()
mapdl.finish()

# Post-processing: von-Mises stress
mapdl.post1()
mapdl.set(1, 1)
von_mises = mapdl.post_processing.nodal_stress('SEQV')

# Plot von-Mises stress
mapdl.post_processing.plot_nodal_stress('SEQV', show_edges=True, cpos='xy')

# Compare with legacy file reader
rst = read_binary(mapdl.result_file)
legacy_von_mises = rst.nodal_stress(0)[:, -1]
are_same = np.allclose(von_mises, legacy_von_mises, atol=1e-3)
print(f"Are the results the same? {are_same}")

# Stop MAPDL
mapdl.exit()
```