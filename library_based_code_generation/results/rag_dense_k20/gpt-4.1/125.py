```python
from pyscf.pbc import gto, scf, mp
import numpy as np

# Define cell parameters
atom = '''
H 0 0 0
H 0 0 0.74
'''
basis = 'sto-3g'
pseudo = 'gth-pade'
a = [[0.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0]]
unit = 'Bohr'
verbosity = 4

# Create cell
cell = gto.Cell()
cell.atom = atom
cell.basis = basis
cell.pseudo = pseudo
cell.a = a
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

# KHF and KMP2 with single k-point
kmesh1 = [1, 1, 1]
kpts1 = cell.make_kpts(kmesh1)
khf1 = scf.KRHF(cell, kpts=kpts1)
khf1.kernel()
kmp2_1 = mp.KMP2(khf1)
emp2_1, t2_1 = kmp2_1.kernel()
print("KMP2 energy per unit cell (single k-point):", emp2_1)

# RHF and RMP2 with single k-point
rhf = scf.RHF(cell)
rhf.kernel()
rmp2 = mp.MP2(rhf)
emp2_rhf, t2_rhf = rmp2.kernel()
print("RMP2 energy per unit cell (single k-point):", emp2_rhf)

# Generate 1- and 2-RDMs and calculate total energy for RMP2
rdm1 = rmp2.make_rdm1()
rdm2 = rmp2.make_rdm2()
h1 = rhf.get_hcore()
eri = rhf.with_df.ao2mo()
e1 = np.einsum('ij,ji', h1, rdm1)
e2 = 0.5 * np.einsum('ijkl,ijkl', eri, rdm2)
e_tot_rmp2 = e1 + e2 + cell.energy_nuc()
print("RMP2 total energy from RDMs:", e_tot_rmp2)

# Convert RHF to UHF and GHF
uhf = scf.addons.convert_to_uhf(rhf)
ghf = scf.addons.convert_to_ghf(rhf)

# UMP2 calculation
ump2 = mp.UMP2(uhf)
emp2_uhf, t2_uhf = ump2.kernel()
print("UMP2 energy per unit cell (single k-point):", emp2_uhf)
rdm1_uhf = ump2.make_rdm1()
rdm2_uhf = ump2.make_rdm2()
h1_uhf = uhf.get_hcore()
eri_uhf = uhf.with_df.ao2mo()
e1_uhf = np.einsum('ij,ji', h1_uhf, rdm1_uhf)
e2_uhf = 0.5 * np.einsum('ijkl,ijkl', eri_uhf, rdm2_uhf)
e_tot_ump2 = e1_uhf + e2_uhf + cell.energy_nuc()
print("UMP2 total energy from RDMs:", e_tot_ump2)

# GMP2 calculation
from pyscf.pbc.mp import GMP2
gmp2 = GMP2(ghf)
emp2_ghf, t2_ghf = gmp2.kernel()
print("GMP2 energy per unit cell (single k-point):", emp2_ghf)
rdm1_ghf = gmp2.make_rdm1()
rdm2_ghf = gmp2.make_rdm2()
h1_ghf = ghf.get_hcore()
eri_ghf = ghf.with_df.ao2mo()
e1_ghf = np.einsum('ij,ji', h1_ghf, rdm1_ghf)
e2_ghf = 0.5 * np.einsum('ijkl,ijkl', eri_ghf, rdm2_ghf)
e_tot_gmp2 = e1_ghf + e2_ghf + cell.energy_nuc()
print("GMP2 total energy from RDMs:", e_tot_gmp2)
```