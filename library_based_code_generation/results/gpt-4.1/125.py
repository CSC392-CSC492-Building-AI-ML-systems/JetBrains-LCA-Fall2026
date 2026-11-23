```python
import numpy as np
from pyscf import pbc
from pyscf.pbc import gto, scf, mp
from pyscf.pbc.tools import lattice

# Define cell parameters
cell = gto.Cell()
cell.atom = '''
H 0.0 0.0 0.0
H 0.0 0.0 0.74
'''
cell.basis = 'sto-3g'
cell.pseudo = 'gth-pade'
cell.a = lattice.hexagonal(1.0, 1.5)
cell.unit = 'B'
cell.verbosity = 4
cell.build()

# KHF and KMP2 with 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts=kpts)
kmf.kernel()
kmp2 = mp.KMP2(kmf)
emp2, t2 = kmp2.kernel()
print("KMP2 energy per unit cell (2x2x2 k-points):", emp2)

# KHF and KMP2 with single k-point (Gamma)
kpts_gamma = cell.make_kpts([1, 1, 1])
kmf_gamma = scf.KRHF(cell, kpts=kpts_gamma)
kmf_gamma.kernel()
kmp2_gamma = mp.KMP2(kmf_gamma)
emp2_gamma, t2_gamma = kmp2_gamma.kernel()
print("KMP2 energy per unit cell (Gamma):", emp2_gamma)

# RHF and RMP2 at Gamma
mf = scf.RHF(cell)
mf.kernel()
mp2 = mp.MP2(mf)
emp2_rhf, t2_rhf = mp2.kernel()
print("RMP2 energy per unit cell (Gamma):", emp2_rhf)

# Generate 1-RDM and 2-RDM, calculate total energy
rdm1 = mp2.make_rdm1()
rdm2 = mp2.make_rdm2()
h1 = mf.get_hcore()
eri = mf._eri
e1 = np.einsum('ij,ji', h1, rdm1)
e2 = 0.5 * np.einsum('ijkl,ijkl', eri, rdm2)
enuc = cell.energy_nuc()
etot_rmp2 = e1 + e2 + enuc
print("RMP2 total energy from RDMs (Gamma):", etot_rmp2)

# Convert RHF to UHF and GHF
uhf = mf.to_uhf()
ghf = mf.to_ghf()

# UMP2 calculation
ump2 = mp.UMP2(uhf)
emp2_uhf, t2_uhf = ump2.kernel()
print("UMP2 energy per unit cell (Gamma):", emp2_uhf)
rdm1u = ump2.make_rdm1()
rdm2u = ump2.make_rdm2()
h1u = uhf.get_hcore()
e1u = np.einsum('ij,ji', h1u[0], rdm1u[0]) + np.einsum('ij,ji', h1u[1], rdm1u[1])
eri_aa = uhf._eri[0]
eri_ab = uhf._eri[1]
eri_bb = uhf._eri[2]
e2u = 0.5 * (np.einsum('ijkl,ijkl', eri_aa, rdm2u[0]) +
             np.einsum('ijkl,ijkl', eri_ab, rdm2u[1]) +
             np.einsum('ijkl,ijkl', eri_bb, rdm2u[2]))
etot_ump2 = e1u + e2u + enuc
print("UMP2 total energy from RDMs (Gamma):", etot_ump2)

# GMP2 calculation
gmp2 = mp.GMP2(ghf)
emp2_ghf, t2_ghf = gmp2.kernel()
print("GMP2 energy per unit cell (Gamma):", emp2_ghf)
rdm1g = gmp2.make_rdm1()
rdm2g = gmp2.make_rdm2()
h1g = ghf.get_hcore()
e1g = np.einsum('ij,ji', h1g, rdm1g)
eri_g = ghf._eri
e2g = 0.5 * np.einsum('ijkl,ijkl', eri_g, rdm2g)
etot_gmp2 = e1g + e2g + enuc
print("GMP2 total energy from RDMs (Gamma):", etot_gmp2)
```