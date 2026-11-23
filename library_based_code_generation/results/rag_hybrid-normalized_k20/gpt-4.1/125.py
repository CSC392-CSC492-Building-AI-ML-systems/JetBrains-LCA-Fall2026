```python
from pyscf import pbc, gto, scf, mp
from pyscf.pbc import gto as pbcgto, scf as pbcscf, mp as pbcmp
import numpy as np

# Define cell parameters
cell = pbcgto.Cell()
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

# 2x2x2 k-points mesh
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

# KHF calculation
khf = pbcscf.KRHF(cell, kpts)
khf.kernel()

# KMP2 calculation
kmp2 = pbcmp.KMP2(khf)
emp2, t2 = kmp2.kernel()
print("KMP2 energy per unit cell (2x2x2 k-points):", emp2)

# Single k-point calculation (Gamma point)
kpts_gamma = cell.make_kpts([1, 1, 1])

# KHF at Gamma
khf_gamma = pbcscf.KRHF(cell, kpts_gamma)
khf_gamma.kernel()

# KMP2 at Gamma
kmp2_gamma = pbcmp.KMP2(khf_gamma)
emp2_gamma, t2_gamma = kmp2_gamma.kernel()
print("KMP2 energy per unit cell (Gamma):", emp2_gamma)

# RHF at Gamma
rhf = pbcscf.RHF(cell)
rhf.kernel()

# RMP2 at Gamma
rmp2 = mp.MP2(rhf)
emp2_rhf, t2_rhf = rmp2.kernel()
print("RMP2 energy per unit cell (Gamma):", emp2_rhf)

# 1st and 2nd order RDMs for RMP2
rdm1_rmp2 = rmp2.make_rdm1()
rdm2_rmp2 = rmp2.make_rdm2()
e_tot_rmp2 = rmp2.energy_tot(rdm1=rdm1_rmp2, rdm2=rdm2_rmp2)
print("RMP2 total energy from RDMs:", e_tot_rmp2)

# Convert RHF to UHF
uhf = scf.addons.convert_to_uhf(rhf)
uhf.kernel()

# UMP2 calculation
ump2 = mp.UMP2(uhf)
emp2_uhf, t2_uhf = ump2.kernel()
print("UMP2 energy per unit cell (Gamma):", emp2_uhf)

# 1st and 2nd order RDMs for UMP2
rdm1_ump2 = ump2.make_rdm1()
rdm2_ump2 = ump2.make_rdm2()
e_tot_ump2 = ump2.energy_tot(rdm1=rdm1_ump2, rdm2=rdm2_ump2)
print("UMP2 total energy from RDMs:", e_tot_ump2)

# Convert RHF to GHF
ghf = scf.addons.convert_to_ghf(rhf)
ghf.kernel()

# GMP2 calculation
gmp2 = mp.GMP2(ghf)
emp2_ghf, t2_ghf = gmp2.kernel()
print("GMP2 energy per unit cell (Gamma):", emp2_ghf)

# 1st and 2nd order RDMs for GMP2
rdm1_gmp2 = gmp2.make_rdm1()
rdm2_gmp2 = gmp2.make_rdm2()
e_tot_gmp2 = gmp2.energy_tot(rdm1=rdm1_gmp2, rdm2=rdm2_gmp2)
print("GMP2 total energy from RDMs:", e_tot_gmp2)
```