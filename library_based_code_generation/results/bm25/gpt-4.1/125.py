```python
from pyscf import pbc
from pyscf.pbc import gto, scf, mp
import numpy as np

# Define cell parameters
cell = gto.Cell()
cell.atom = '''
H 0 0 0
H 0 0 0.74
'''
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = '''
0.0 1.0 1.0
1.0 0.0 1.0
1.0 1.0 0.0
'''
cell.unit = 'Bohr'
cell.verbose = 4
cell.build()

# KHF and KMP2 with 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
khf = scf.KRHF(cell, kpts=kpts)
khf.kernel()
kmp2 = mp.KMP2(khf)
emp2, _ = kmp2.kernel()
print("KMP2 energy per unit cell (2x2x2 kpts):", emp2)

# KHF and KMP2 with single k-point
kmesh1 = [1, 1, 1]
kpts1 = cell.make_kpts(kmesh1)
khf1 = scf.KRHF(cell, kpts=kpts1)
khf1.kernel()
kmp2_1 = mp.KMP2(khf1)
emp2_1, _ = kmp2_1.kernel()
print("KMP2 energy per unit cell (1x1x1 kpt):", emp2_1)

# RHF and RMP2 with single k-point
rhf = scf.RHF(cell)
rhf.kernel()
rmp2 = mp.MP2(rhf)
emp2_r, _ = rmp2.kernel()
print("RMP2 energy per unit cell (Gamma):", emp2_r)

# Generate 1st and 2nd order RDMs and calculate total energy
rdm1 = rmp2.make_rdm1()
rdm2 = rmp2.make_rdm2()
e_tot_rmp2 = mp.mp2.energy(rmp2, rdm1, rdm2)
print("RMP2 total energy from RDMs:", e_tot_rmp2)

# Convert RHF to UHF and GHF
uhf = rhf.to_uhf()
ghf = rhf.to_ghf()

# UMP2 calculation
ump2 = mp.UMP2(uhf)
emp2_u, _ = ump2.kernel()
print("UMP2 energy per unit cell (Gamma):", emp2_u)
rdm1_u = ump2.make_rdm1()
rdm2_u = ump2.make_rdm2()
e_tot_ump2 = mp.mp2.energy(ump2, rdm1_u, rdm2_u)
print("UMP2 total energy from RDMs:", e_tot_ump2)

# GMP2 calculation
gmp2 = mp.GMP2(ghf)
emp2_g, _ = gmp2.kernel()
print("GMP2 energy per unit cell (Gamma):", emp2_g)
rdm1_g = gmp2.make_rdm1()
rdm2_g = gmp2.make_rdm2()
e_tot_gmp2 = mp.mp2.energy(gmp2, rdm1_g, rdm2_g)
print("GMP2 total energy from RDMs:", e_tot_gmp2)
```