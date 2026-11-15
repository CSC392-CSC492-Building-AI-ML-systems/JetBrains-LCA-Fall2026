```python
from pyscf.pbc import gto, scf, mp
import numpy as np

# Define cell parameters
atom = '''
H 0 0 0
H 0 0 0.74
'''
a = np.eye(3) * 2.0  # cubic cell, 2 Angstrom per side
basis = 'gth-szv'
pseudo = 'gth-pade'
unit = 'B'
verbosity = 4

# Create cell
cell = gto.Cell()
cell.atom = atom
cell.a = a
cell.basis = basis
cell.pseudo = pseudo
cell.unit = unit
cell.verbose = verbosity
cell.build()

# KHF and KMP2 with 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
khf = scf.KRHF(cell, kpts=kpts)
khf.kernel()
kmp2 = mp.KMP2(khf)
emp2, t2 = kmp2.kernel()
print("KMP2 energy per unit cell (2x2x2 k-points):", emp2)

# KHF and KMP2 with single k-point (Gamma)
kpts_gamma = cell.make_kpts([1, 1, 1])
khf_gamma = scf.KRHF(cell, kpts=kpts_gamma)
khf_gamma.kernel()
kmp2_gamma = mp.KMP2(khf_gamma)
emp2_gamma, t2_gamma = kmp2_gamma.kernel()
print("KMP2 energy per unit cell (Gamma point):", emp2_gamma)

# RHF and RMP2 at Gamma point
rhf = scf.RHF(cell)
rhf.kernel()
rmp2 = mp.MP2(rhf)
emp2_rhf, t2_rhf = rmp2.kernel()
print("RMP2 energy per unit cell (RHF, Gamma point):", emp2_rhf)

# Generate 1- and 2-RDMs and calculate total energy for RMP2
rdm1 = rmp2.make_rdm1()
rdm2 = rmp2.make_rdm2()
e_tot_rmp2 = rmp2.energy_tot(rdm1=rdm1, rdm2=rdm2)
print("RMP2 total energy from RDMs:", e_tot_rmp2)

# Convert RHF to UHF and GHF
uhf = scf.convert_to_uhf(rhf)
ghf = scf.convert_to_ghf(rhf)

# UMP2 calculation
ump2 = mp.UMP2(uhf)
emp2_uhf, t2_uhf = ump2.kernel()
print("UMP2 energy per unit cell (UHF, Gamma point):", emp2_uhf)
rdm1_uhf = ump2.make_rdm1()
rdm2_uhf = ump2.make_rdm2()
e_tot_ump2 = ump2.energy_tot(rdm1=rdm1_uhf, rdm2=rdm2_uhf)
print("UMP2 total energy from RDMs:", e_tot_ump2)

# GMP2 calculation
gmp2 = mp.GMP2(ghf)
emp2_ghf, t2_ghf = gmp2.kernel()
print("GMP2 energy per unit cell (GHF, Gamma point):", emp2_ghf)
rdm1_ghf = gmp2.make_rdm1()
rdm2_ghf = gmp2.make_rdm2()
e_tot_gmp2 = gmp2.energy_tot(rdm1=rdm1_ghf, rdm2=rdm2_ghf)
print("GMP2 total energy from RDMs:", e_tot_gmp2)
```