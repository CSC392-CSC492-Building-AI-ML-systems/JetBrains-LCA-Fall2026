```python
from pymapdl import launch_mapdl

# Launch MAPDL
mapdl = launch_mapdl()

# Enter pre-processor
mapdl.prep7()

# Create first block (lower block)
block1 = mapdl.block(0, 10, 0, 10, 0, 5)

# Mesh first block with tetrahedral elements (SOLID185)
mapdl.et(1, 185)
mapdl.vsweep('ALL')
mapdl.esize(1)
mapdl.vmesh('ALL')

# Create second block (upper block), above the first, with a gap
block2 = mapdl.block(0, 10, 0, 10, 6, 11)

# Mesh second block with quadratic hexahedral elements (SOLID186)
mapdl.et(2, 186)
mapdl.vsel('S', 'VOLU', '', 2)
mapdl.esize(1)
mapdl.vmesh('ALL')
mapdl.allsel('ALL')

# Select faces of lower block (top face at Z=5)
mapdl.nsel('S', 'LOC', 'Z', 5)
mapdl.esln('S')
mapdl.cm('CONTACT_NODES', 'NODE')
mapdl.cm('CONTACT_ELEMS', 'ELEM')
mapdl.allsel('ALL')

# Select faces of upper block (bottom face at Z=6)
mapdl.nsel('S', 'LOC', 'Z', 6)
mapdl.esln('S')
mapdl.cm('TARGET_NODES', 'NODE')
mapdl.cm('TARGET_ELEMS', 'ELEM')
mapdl.allsel('ALL')

# Create contact and target elements
mapdl.et(3, 170)  # CONTA170
mapdl.et(4, 174)  # TARGE170

# Create contact elements on lower block top face
mapdl.type(3)
mapdl.real(1)
mapdl.mat(1)
mapdl.esel('S', 'ELEM', '', '', '', '', 'CONTACT_ELEMS')
mapdl.nsle()
mapdl.nsel('R', 'NODE', '', '', '', '', 'CONTACT_NODES')
mapdl.sfe('ALL', 1, 'CONTA', '')

# Create target elements on upper block bottom face
mapdl.type(4)
mapdl.real(2)
mapdl.mat(1)
mapdl.esel('S', 'ELEM', '', '', '', '', 'TARGET_ELEMS')
mapdl.nsle()
mapdl.nsel('R', 'NODE', '', '', '', '', 'TARGET_NODES')
mapdl.sfe('ALL', 1, 'TARGE', '')

mapdl.allsel('ALL')

# Plot contact element pairs as wire-frame
mapdl.eplot(style='wire', show_edges=True, element_types=[3, 4])

# Stop MAPDL
mapdl.exit()
```