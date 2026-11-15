```python
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools import lattice

# Build the cell
cell = gto.Cell()
cell.atom = '''
C 0.0 0.0 0.0
C 1.4 0.0 0.0
'''
cell.a = '''
2.8 0.0 0.0
0.0 2.8 0.0
0.0 0.0 2.8
'''
cell.unit = 'Angstrom'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

# 2x2x2 k-points
kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh, wrap_around=True)

# KHF calculation
kmf = scf.KRHF(cell, kpts)
ehf_k = kmf.kernel()
print("KHF (2x2x2) total energy per unit cell: ", ehf_k)

# KCCSD calculation
kcc = cc.KCCSD(kmf)
ecc_k, t1_k, t2_k = kcc.kernel()
print("KCCSD (2x2x2) total energy per unit cell: ", ecc_k + ehf_k)

# Single k-point calculation
kpts1 = cell.make_kpts([1, 1, 1], wrap_around=True)
kmf1 = scf.KRHF(cell, kpts1)
ehf1 = kmf1.kernel()
print("KHF (1x1x1) total energy per unit cell: ", ehf1)

kcc1 = cc.KCCSD(kmf1)
ecc1, t1_1, t2_1 = kcc1.kernel()
print("KCCSD (1x1x1) total energy per unit cell: ", ecc1 + ehf1)

# RHF calculation at single k-point
mf_rhf = scf.RHF(cell, kpts=kpts1)
ehf_rhf = mf_rhf.kernel()
print("RHF (single k-point) total energy per unit cell: ", ehf_rhf)

cc_rccsd = cc.CCSD(mf_rhf)
ecc_rccsd, t1_r, t2_r = cc_rccsd.kernel()
print("RCCSD (single k-point) total energy per unit cell: ", ecc_rccsd + ehf_rhf)

# RCCSD energy from density matrices
rdm1_r, rdm2_r = cc_rccsd.make_rdm1(), cc_rccsd.make_rdm2()
e_rccsd_dm = mf_rhf.energy_tot(dm=rdm1_r)
print("RCCSD (single k-point) energy from CCSD density matrices: ", e_rccsd_dm)

# Convert RHF to UHF
mf_uhf = mf_rhf.to_uhf()
ehf_uhf = mf_uhf.kernel()
print("UHF (single k-point) total energy per unit cell: ", ehf_uhf)

cc_uccsd = cc.UCCSD(mf_uhf)
ecc_uccsd, t1_u, t2_u = cc_uccsd.kernel()
print("UCCSD (single k-point) total energy per unit cell: ", ecc_uccsd + ehf_uhf)

# UCCSD energy from density matrices
rdm1_u, rdm2_u = cc_uccsd.make_rdm1(), cc_uccsd.make_rdm2()
e_uccsd_dm = mf_uhf.energy_tot(dm=rdm1_u)
print("UCCSD (single k-point) energy from CCSD density matrices: ", e_uccsd_dm)

# Convert UHF to GHF
mf_ghf = mf_uhf.to_ghf()
ehf_ghf = mf_ghf.kernel()
print("GHF (single k-point) total energy per unit cell: ", ehf_ghf)

cc_gccsd = cc.GCCSD(mf_ghf)
ecc_gccsd, t1_g, t2_g = cc_gccsd.kernel()
print("GCCSD (single k-point) total energy per unit cell: ", ecc_gccsd + ehf_ghf)

# GCCSD energy from density matrices
rdm1_g, rdm2_g = cc_gccsd.make_rdm1(), cc_gccsd.make_rdm2()
e_gccsd_dm = mf_ghf.energy_tot(dm=rdm1_g)
print("GCCSD (single k-point) energy from CCSD density matrices: ", e_gccsd_dm)
```