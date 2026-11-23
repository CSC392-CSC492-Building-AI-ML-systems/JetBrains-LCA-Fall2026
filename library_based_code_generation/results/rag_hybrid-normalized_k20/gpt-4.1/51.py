```python
import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools import k2gamma

# Build the cell
cell = gto.Cell()
cell.atom = '''
C 0.0 0.0 0.0
C 1.4 0.0 0.0
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = [
    [2.8, 0.0, 0.0],
    [0.0, 2.8, 0.0],
    [0.0, 0.0, 2.8]
]
cell.unit = 'Angstrom'
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
print("KCCSD (2x2x2 k-points) total energy per unit cell: %.12f" % kcc.e_tot)

# Single k-point calculation
kpts_single = cell.make_kpts([1, 1, 1])

# KHF single k-point
khf1 = scf.KRHF(cell, kpts_single)
khf1.kernel()

# KCCSD single k-point
kcc1 = cc.KCCSD(khf1)
kcc1.kernel()
print("KCCSD (1x1x1 k-point) total energy per unit cell: %.12f" % kcc1.e_tot)

# RHF single k-point
rhf = scf.RHF(cell)
rhf.kernel()

# RCCSD
rcc = cc.CCSD(rhf)
rcc.kernel()
print("RCCSD (RHF, single k-point) total energy per unit cell: %.12f" % rcc.e_tot)

# RCCSD energy from density matrices
rdm1 = rcc.make_rdm1()
rdm2 = rcc.make_rdm2()
energy_rccsd_dm = rcc.energy_elec(rdm1, rdm2)[0] + cell.energy_nuc()
print("RCCSD (from density matrices) total energy per unit cell: %.12f" % energy_rccsd_dm)

# Convert RHF to UHF
uhf = rhf.to_uhf()
uhf.kernel()

# UCCSD
ucc = cc.UCCSD(uhf)
ucc.kernel()
print("UCCSD (UHF, single k-point) total energy per unit cell: %.12f" % ucc.e_tot)

# UCCSD energy from density matrices
udm1 = ucc.make_rdm1()
udm2 = ucc.make_rdm2()
energy_uccsd_dm = ucc.energy_elec(udm1, udm2)[0] + cell.energy_nuc()
print("UCCSD (from density matrices) total energy per unit cell: %.12f" % energy_uccsd_dm)

# Convert UHF to GHF
from pyscf.pbc.scf import ghf
ghf_obj = ghf.GHF(cell)
ghf_obj.mo_coeff = uhf.mo_coeff
ghf_obj.mo_occ = uhf.mo_occ
ghf_obj.mo_energy = uhf.mo_energy
ghf_obj.kernel()

# GCCSD
from pyscf.pbc.cc import gccsd
gcc = gccsd.GCCSD(ghf_obj)
gcc.kernel()
print("GCCSD (GHF, single k-point) total energy per unit cell: %.12f" % gcc.e_tot)

# GCCSD energy from density matrices
gdm1 = gcc.make_rdm1()
gdm2 = gcc.make_rdm2()
energy_gccsd_dm = gcc.energy_elec(gdm1, gdm2)[0] + cell.energy_nuc()
print("GCCSD (from density matrices) total energy per unit cell: %.12f" % energy_gccsd_dm)
```