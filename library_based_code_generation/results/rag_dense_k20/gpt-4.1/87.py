```python
from ansys.mapdl.core import launch_mapdl

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create first block (lower block)
block1_x = [0, 50]
block1_y = [0, 50]
block1_z = [0, 20]
mapdl.block(block1_x[0], block1_x[1], block1_y[0], block1_y[1], block1_z[0], block1_z[1])

# Mesh first block with tetrahedral elements (SOLID185)
mapdl.et(1, 185)
mapdl.esize(5)
mapdl.vsel('S', 'VOLU', '', 1)
mapdl.vmesh('ALL')

# Create second block (upper block), above the first, not touching
gap = 5
block2_z = [block1_z[1] + gap, block1_z[1] + gap + 20]
mapdl.block(block1_x[0], block1_x[1], block1_y[0], block1_y[1], block2_z[0], block2_z[1])

# Mesh second block with quadratic hexahedral elements (SOLID186)
mapdl.et(2, 186)
mapdl.keyopt(2, 3, 2)  # 20-node brick
mapdl.vsel('S', 'VOLU', '', 2)
mapdl.type(2)
mapdl.esize(5)
mapdl.vmesh('ALL')

# Select all elements at the bottom face of the upper block (potential contact)
mapdl.esel('ALL')
mapdl.nsel('S', 'LOC', 'Z', block2_z[0])
mapdl.esln('S', 'NODE')
mapdl.cm('CONTACT_NODES', 'NODE')
mapdl.nsel('ALL')

# Select all elements at the top face of the lower block (potential target)
mapdl.nsel('S', 'LOC', 'Z', block1_z[1])
mapdl.esln('S', 'NODE')
mapdl.cm('TARGET_NODES', 'NODE')
mapdl.nsel('ALL')

# Create contact and target surfaces
mapdl.et(3, 170)  # CONTA170
mapdl.et(4, 174)  # TARGE174

# Create contact elements on upper block's bottom face
mapdl.type(3)
mapdl.real(1)
mapdl.esel('S', 'NODE', '', '', '', '', 'CONTACT_NODES')
mapdl.nsle()
mapdl.sfe('ALL', 1, 'CONTA', '')

# Create target elements on lower block's top face
mapdl.type(4)
mapdl.real(2)
mapdl.esel('S', 'NODE', '', '', '', '', 'TARGET_NODES')
mapdl.nsle()
mapdl.sfe('ALL', 1, 'TARGE', '')

# Select contact and target elements for plotting
mapdl.esel('S', 'TYPE', '', 3)
mapdl.esel('A', 'TYPE', '', 4)

# Plot contact element pairs as wire-frame
mapdl.eplot(style='wire')

# Stop MAPDL
mapdl.exit()
```