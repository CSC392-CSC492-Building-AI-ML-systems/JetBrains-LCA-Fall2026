```python
from ansys.mapdl.core import launch_mapdl

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create first block (lower block)
block1 = mapdl.block(0, 10, 0, 10, 0, 10)

# Mesh first block with tetrahedral elements
mapdl.et(1, 285)  # 285 = SOLID285 (tetrahedral)
mapdl.vsweep('ALL')
mapdl.esize(2)
mapdl.vmesh('ALL')

# Create second block (upper block, not touching the first)
block2 = mapdl.block(0, 10, 0, 10, 12, 22)

# Mesh second block with quadratic hexahedral elements
mapdl.et(2, 186)  # 186 = SOLID186 (20-node hexahedral)
mapdl.vsel('S', 'LOC', 'Z', 12, 22)
mapdl.esize(2)
mapdl.vmesh('ALL')
mapdl.allsel()

# Create contact and target elements at the intersection (gap) between blocks
# Select faces on top of block1 (Z=10)
mapdl.nsel('S', 'LOC', 'Z', 10)
mapdl.esln('S', 'NODE')
mapdl.cm('contact_faces', 'ELEM')
mapdl.allsel()

# Select faces on bottom of block2 (Z=12)
mapdl.nsel('S', 'LOC', 'Z', 12)
mapdl.esln('S', 'NODE')
mapdl.cm('target_faces', 'ELEM')
mapdl.allsel()

# Define contact and target elements
mapdl.et(3, 170)  # CONTA170
mapdl.et(4, 174)  # TARGE170

# Create contact elements on block1 top face
mapdl.esel('S', 'NAME', '', 'contact_faces')
mapdl.type(3)
mapdl.real(1)
mapdl.mat(1)
mapdl.esurf()

# Create target elements on block2 bottom face
mapdl.esel('S', 'NAME', '', 'target_faces')
mapdl.type(4)
mapdl.real(2)
mapdl.mat(1)
mapdl.esurf()
mapdl.allsel()

# Plot contact element pairs as wire-frame
mapdl.eplot(style='wire', show_edges=True, element_types=[3, 4])

# Stop MAPDL
mapdl.exit()
```