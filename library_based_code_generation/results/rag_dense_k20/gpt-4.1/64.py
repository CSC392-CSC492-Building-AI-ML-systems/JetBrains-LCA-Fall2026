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
E = 3e7      # Young's modulus in psi
NU = 0.3     # Poisson's ratio
mapdl.mp('EX', 1, E)
mapdl.mp('PRXY', 1, NU)

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
pressure = 1500  # psi
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
mapdl.etable('SVM', 'S', 'EQV')
nnum, svm = mapdl.post_processing.nodal_solution('S', 'EQV')

# Plot von-Mises stress
mapdl.post_processing.plot_nodal_solution('S', 'EQV', cmap='jet', show_edges=True)

# Compare with legacy file reader
from ansys.mapdl.reader import read_binary
rst = read_binary(mapdl.result_file)
legacy_nnum, legacy_svm = rst.nodal_stress(0, principal=False, von_mises=True)
# Interpolate legacy results to match MAPDL node order
legacy_svm_map = dict(zip(legacy_nnum, legacy_svm))
svm_legacy = np.array([legacy_svm_map.get(n, np.nan) for n in nnum])

# Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(nnum, svm, 'b.', label='PyMAPDL')
plt.plot(nnum, svm_legacy, 'r.', label='Legacy Reader', alpha=0.5)
plt.xlabel('Node Number')
plt.ylabel('Von-Mises Stress (psi)')
plt.legend()
plt.title('Von-Mises Stress Comparison')
plt.show()

# Stop MAPDL
mapdl.exit()
```