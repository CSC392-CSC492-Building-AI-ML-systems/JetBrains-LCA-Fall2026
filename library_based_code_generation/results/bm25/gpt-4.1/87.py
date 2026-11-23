```python
from ansys.mapdl.core import launch_mapdl
import numpy as np

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create first block (lower block)
block1_zmin = 0
block1_zmax = 10
block1 = mapdl.block(0, 10, 0, 10, block1_zmin, block1_zmax)

# Mesh first block with tetrahedral elements
mapdl.et(1, 187)  # SOLID187: 10-node tetrahedral
mapdl.vsel('S', 'VOLU', '', 1)
mapdl.vmesh('ALL')

# Create second block (upper block), above the first, with a gap
gap = 2
block2_zmin = block1_zmax + gap
block2_zmax = block2_zmin + 10
block2 = mapdl.block(0, 10, 0, 10, block2_zmin, block2_zmax)

# Mesh second block with quadratic hexahedral elements
mapdl.et(2, 186)  # SOLID186: 20-node hexahedral
mapdl.vsel('S', 'VOLU', '', 2)
mapdl.vmesh('ALL')

# Select all elements at the intersection (top face of block1, bottom face of block2)
# Select elements on top face of block1
mapdl.esel('ALL')
mapdl.nsel('S', 'LOC', 'Z', block1_zmax)
mapdl.esln('S', 'NODE')
mapdl.cm('TOP1', 'ELEM')

# Select elements on bottom face of block2
mapdl.nsel('S', 'LOC', 'Z', block2_zmin)
mapdl.esln('S', 'NODE')
mapdl.cm('BOT2', 'ELEM')

# Create contact and target elements for general contact
mapdl.et(3, 170)  # CONTA170: 3D node-to-surface contact
mapdl.et(4, 174)  # TARGE170: 3D target

# Create contact pair: assign contact to top of block1, target to bottom of block2
mapdl.esel('S', 'ELEM', '', '', '', '', 'TOP1')
mapdl.type(3)
mapdl.real(1)
mapdl.mat(1)
mapdl.emodif('ALL', 'TYPE', 3)

mapdl.esel('S', 'ELEM', '', '', '', '', 'BOT2')
mapdl.type(4)
mapdl.real(1)
mapdl.mat(1)
mapdl.emodif('ALL', 'TYPE', 4)

# Define contact pair
mapdl.seltol(0.1)
mapdl.allsel()
mapdl.contact('ALL', 'ALL', 'ALL', 'ALL')

# Plot contact element pairs as wire-frame
mapdl.eplot(style='wire', show_edges=True, color='contact')

# Stop MAPDL
mapdl.exit()
```