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
mapdl.mp('EX', 1, 30e6)      # Young's modulus in psi
mapdl.mp('NUXY', 1, 0.3)     # Poisson's ratio

# Define element type (Solid185)
mapdl.et(1, 'SOLID185')

# Geometry parameters (in inches)
ri = 20      # inner radius
ro = 24      # outer radius
length = 60  # vessel length

# Create geometry: hollow cylinder (pressure vessel)
mapdl.cylind(ri, ro, length)

# Mesh controls
mapdl.esize(2)
mapdl.vmesh('ALL')

# Apply boundary conditions
# Fix one end (all DOF at z=0)
mapdl.nsel('S', 'LOC', 'Z', 0)
mapdl.d('ALL', 'ALL', 0)
mapdl.nsel('ALL')

# Apply internal pressure to inner surface
pressure = 150  # psi
mapdl.asel('S', 'LOC', 'RADIUS', ri)
mapdl.nsla('S', 1)
mapdl.esln('S')
mapdl.sfa('ALL', 1, 'PRES', pressure)
mapdl.asel('ALL')

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
nodes = mapdl.mesh.nodes

# Plot von-Mises stress
mapdl.post_processing.plot_nodal_stress('SEQV', show_edges=True, cmap='jet')

# Compare with legacy file reader
from ansys.mapdl.reader import read_binary
rst = read_binary(mapdl.result_file)
legacy_von_mises = rst.nodal_stress(0)[:, -1]  # von-Mises is last column

# Plot comparison
plt.figure(figsize=(8, 5))
plt.hist(von_mises, bins=40, alpha=0.5, label='PyMAPDL')
plt.hist(legacy_von_mises, bins=40, alpha=0.5, label='Legacy Reader')
plt.xlabel('Von-Mises Stress (psi)')
plt.ylabel('Number of Nodes')
plt.legend()
plt.title('Von-Mises Stress Distribution Comparison')
plt.show()

# Stop MAPDL
mapdl.exit()
```