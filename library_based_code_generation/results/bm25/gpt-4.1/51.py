```python
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools import lattice

# Build the cell
cell = gto.Cell()
cell.atom = '''
C 0.000 0.000 0.000
C 1.685 1.685 1.685
'''
cell.a = '''
3.37 0.00 0.00
0.00 3.37 0.00
0.00 0.00 3.37
'''
cell.unit = 'Angstrom'
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
rhf = scf.RHF(cell, kpts=kpts_single)
rhf.kernel()

# RCCSD
rcc = cc.RCCSD(rhf)
rcc.kernel()
print("RCCSD (single k-point) total energy per unit cell: ", rcc.e_tot)

# RCCSD energy from density matrices
dm1, dm2 = rcc.make_rdm1(), rcc.make_rdm2()
rcc_energy_dm = rcc.energy(dm1=dm1, dm2=dm2)
print("RCCSD (single k-point) energy from density matrices: ", rcc_energy_dm)

# Convert RHF to UHF
uhf = rhf.to_uhf()
uhf.kernel()

# UCCSD
ucc = cc.UCCSD(uhf)
ucc.kernel()
print("UCCSD (single k-point) total energy per unit cell: ", ucc.e_tot)

# UCCSD energy from density matrices
udm1, udm2 = ucc.make_rdm1(), ucc.make_rdm2()
ucc_energy_dm = ucc.energy(dm1=udm1, dm2=udm2)
print("UCCSD (single k-point) energy from density matrices: ", ucc_energy_dm)

# Convert UHF to GHF
ghf = uhf.to_ghf()
ghf.kernel()

# GCCSD
gcc = cc.GCCSD(ghf)
gcc.kernel()
print("GCCSD (single k-point) total energy per unit cell: ", gcc.e_tot)

# GCCSD energy from density matrices
gdm1, gdm2 = gcc.make_rdm1(), gcc.make_rdm2()
gcc_energy_dm = gcc.energy(dm1=gdm1, dm2=gdm2)
print("GCCSD (single k-point) energy from density matrices: ", gcc_energy_dm)
```