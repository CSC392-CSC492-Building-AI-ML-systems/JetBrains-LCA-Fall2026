```python
import numpy as np
from pyscf.pbc import gto, scf, cc, tools
from pyscf.pbc.tools import k2gamma

# Build the cell
cell = gto.Cell()
cell.atom = '''
C 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.685068
'''
cell.a = '''
3.5668 0.0000 0.0000
0.0000 3.5668 0.0000
0.0000 0.0000 3.5668
'''
cell.unit = 'B'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

# 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

# KHF calculation
khf = scf.KRHF(cell, kpts)
khf.kernel()

# KCCSD calculation
kcc = cc.KCCSD(khf)
kcc.kernel()
print("KCCSD (2x2x2 k-points) total energy per unit cell: ", kcc.e_tot)

# Single k-point calculation
kpts_single = cell.make_kpts([1, 1, 1])

# KHF single k-point
khf1 = scf.KRHF(cell, kpts_single)
khf1.kernel()

# KCCSD single k-point
kcc1 = cc.KCCSD(khf1)
kcc1.kernel()
print("KCCSD (single k-point) total energy per unit cell: ", kcc1.e_tot)

# RHF single k-point
rhf = scf.RHF(cell)
rhf.kernel()

# RCCSD
rcc = cc.CCSD(rhf)
rcc.kernel()
print("RCCSD (single k-point) total energy per unit cell: ", rcc.e_tot)

# RCCSD energy from density matrices
dm1 = rcc.make_rdm1()
dm2 = rcc.make_rdm2()
e_rcc_dm = rcc.energy(dm1=dm1, dm2=dm2)
print("RCCSD (from density matrices) energy per unit cell: ", e_rcc_dm)

# Convert RHF to UHF
uhf = scf.addons.convert_to_uhf(rhf)
uhf.kernel()

# UCCSD
ucc = cc.UCCSD(uhf)
ucc.kernel()
print("UCCSD (single k-point) total energy per unit cell: ", ucc.e_tot)

# UCCSD energy from density matrices
dm1u = ucc.make_rdm1()
dm2u = ucc.make_rdm2()
e_ucc_dm = ucc.energy(dm1=dm1u, dm2=dm2u)
print("UCCSD (from density matrices) energy per unit cell: ", e_ucc_dm)

# Convert UHF to GHF
from pyscf.pbc.scf import addons as pbc_addons
ghf = pbc_addons.convert_to_ghf(uhf)
ghf.kernel()

# GCCSD
from pyscf.pbc.cc import gccsd
gcc = gccsd.GCCSD(ghf)
gcc.kernel()
print("GCCSD (single k-point) total energy per unit cell: ", gcc.e_tot)

# GCCSD energy from density matrices
dm1g = gcc.make_rdm1()
dm2g = gcc.make_rdm2()
e_gcc_dm = gcc.energy(dm1=dm1g, dm2=dm2g)
print("GCCSD (from density matrices) energy per unit cell: ", e_gcc_dm)
```