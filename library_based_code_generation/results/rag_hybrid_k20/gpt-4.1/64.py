```python
from ansys.mapdl.core import launch_mapdl
import numpy as np
import matplotlib.pyplot as plt

# Launch MAPDL
mapdl = launch_mapdl()

# Set units to US Customary (inches)
mapdl.prep7()
mapdl.units('BIN')  # BIN = British Inch

# Define material properties (e.g., Steel)
mapdl.mp('EX', 1, 3e7)      # Young's modulus in psi
mapdl.mp('NUXY', 1, 0.3)    # Poisson's ratio

# Define element type (Solid185)
mapdl.et(1, 'SOLID185')

# Geometry parameters (inches)
ri = 20      # inner radius
ro = 24      # outer radius
length = 60  # vessel length

# Create geometry: hollow cylinder (pressure vessel)
mapdl.cylind(ri, ro, 0, length)

# Mesh controls
mapdl.esize(2)
mapdl.vmesh('ALL')

# Apply boundary conditions
# Fix one end (all DOF at Z=0)
mapdl.nsel('S', 'LOC', 'Z', 0)
mapdl.d('ALL', 'ALL', 0)
mapdl.nsel('ALL')

# Apply internal pressure (e.g., 500 psi) to inner surface
mapdl.sf('ALL', 'PRES', 0)  # Clear previous
mapdl.esel('S', 'LOC', 'R', ri, ri+0.1)
mapdl.sfe('ALL', 1, 'PRES', 500)
mapdl.esel('ALL')

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
from ansys.mapdl.reader import read_binary
rst = read_binary(mapdl.result_file)
legacy_von_mises = rst.nodal_stress(0)[:, -1]

# Plot comparison
nodes = mapdl.mesh.nnum
plt.figure(figsize=(10,5))
plt.plot(nodes, von_mises, label='PyMAPDL')
plt.plot(nodes, legacy_von_mises, '--', label='Legacy Reader')
plt.xlabel('Node Number')
plt.ylabel('Von-Mises Stress (psi)')
plt.legend()
plt.title('Von-Mises Stress Comparison')
plt.show()

# Stop MAPDL
mapdl.exit()
```