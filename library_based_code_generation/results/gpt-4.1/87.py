```python
from ansys.mapdl.core import launch_mapdl
import numpy as np

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create first block (lower block)
block1 = mapdl.block(0, 50, 0, 50, 0, 20)

# Mesh first block with tetrahedral elements (SOLID185)
mapdl.et(1, 185)
mapdl.esize(5)
mapdl.vsweep(1)
mapdl.vmesh(1)

# Create second block (upper block), above the first, with a gap
gap = 5
block2 = mapdl.block(0, 50, 0, 50, 25, 45)  # 5 units gap between blocks

# Mesh second block with quadratic hexahedral elements (SOLID186)
mapdl.et(2, 186)
mapdl.vsweep(2)
mapdl.vmesh(2)

# Select all elements at the bottom face of the upper block (Z=25)
mapdl.nsel('S', 'LOC', 'Z', 25)
mapdl.esln('S', '1')
bottom_face_elems = mapdl.mesh.elem.selected

# Select all elements at the top face of the lower block (Z=20)
mapdl.nsel('S', 'LOC', 'Z', 20)
mapdl.esln('S', '1')
top_face_elems = mapdl.mesh.elem.selected

# Combine both faces for contact pair creation
mapdl.allsel('ALL')

# Define contact and target element types
mapdl.et(10, 170)  # CONTA174
mapdl.et(11, 171)  # TARGE170

# Define real constants for contact and target
mapdl.r(10)
mapdl.r(11)

# Define contact and target surfaces
mapdl.nsel('S', 'LOC', 'Z', 20)
mapdl.cm('TOPFACE', 'NODE')
mapdl.nsel('S', 'LOC', 'Z', 25)
mapdl.cm('BOTFACE', 'NODE')
mapdl.allsel('ALL')

# Create contact and target elements
mapdl.smat('CLEAR')
mapdl.smat('NEW', 'CONTA')
mapdl.smat('ADD', 'CONTA', 'TOPFACE', 10, 10)
mapdl.smat('ADD', 'CONTA', 'BOTFACE', 11, 11)
mapdl.smat('GEN', 'CONTA')

# Plot contact element pairs as wire-frame
mapdl.eplot(show_edges=True, quality='wireframe', element_types=[10, 11])

# Stop MAPDL
mapdl.exit()
```