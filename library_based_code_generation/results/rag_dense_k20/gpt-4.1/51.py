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
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
3.37 0.00 0.00
0.00 3.37 0.00
0.00 0.00 3.37
'''
cell.unit = 'Angstrom'
cell.verbose = 4
cell.build()

# 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh, space_group_symmetry=True)

# KHF calculation
khf = scf.KRHF(cell, kpts)
khf.kernel()

# KCCSD calculation
kcc = cc.KCCSD(khf)
kcc.kernel()
print("KCCSD (2x2x2 k-points) total energy per unit cell: ", kcc.e_tot)

# Single k-point calculation
kpts_single = cell.make_kpts([1, 1, 1])
khf1 = scf.KRHF(cell, kpts_single)
khf1.kernel()

kcc1 = cc.KCCSD(khf1)
kcc1.kernel()
print("KCCSD (single k-point) total energy per unit cell: ", kcc1.e_tot)

# RHF calculation at single k-point
rhf = scf.RHF(cell)
rhf.kernel()

rcc = cc.CCSD(rhf)
rcc.kernel()
print("RCCSD (RHF, single k-point) total energy per unit cell: ", rcc.e_tot)

# RCCSD energy from density matrices
rdm1, rdm2 = rcc.make_rdm1(), rcc.make_rdm2()
e_rcc_dm = rcc.energy(rdm1=rdm1, rdm2=rdm2)
print("RCCSD energy from density matrices: ", e_rcc_dm)

# Convert RHF to UHF
uhf = scf.addons.convert_to_uhf(rhf)
uhf.kernel()

ucc = cc.UCCSD(uhf)
ucc.kernel()
print("UCCSD (UHF, single k-point) total energy per unit cell: ", ucc.e_tot)

# UCCSD energy from density matrices
udm1, udm2 = ucc.make_rdm1(), ucc.make_rdm2()
e_ucc_dm = ucc.energy(rdm1=udm1, rdm2=udm2)
print("UCCSD energy from density matrices: ", e_ucc_dm)

# Convert UHF to GHF
from pyscf.pbc.scf import ghf
ghf_obj = ghf.GHF(cell)
ghf_obj.mo_coeff = uhf.mo_coeff
ghf_obj.mo_occ = uhf.mo_occ
ghf_obj.mo_energy = uhf.mo_energy
ghf_obj.kernel()

gcc = cc.GCCSD(ghf_obj)
gcc.kernel()
print("GCCSD (GHF, single k-point) total energy per unit cell: ", gcc.e_tot)

# GCCSD energy from density matrices
gdm1, gdm2 = gcc.make_rdm1(), gcc.make_rdm2()
e_gcc_dm = gcc.energy(rdm1=gdm1, rdm2=gdm2)
print("GCCSD energy from density matrices: ", e_gcc_dm)
```