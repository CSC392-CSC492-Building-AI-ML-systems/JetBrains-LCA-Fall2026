```python
import numpy as np
import os
from pyscf import gto, dft, scf, ao2mo, lib

# Define molecule and basis
mol = gto.Mole()
mol.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 -0.757160 0.586260
H 0.000000 0.757160 0.586260
'''
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()

# State 1: DFT calculation, store to chkfile1
mf1 = dft.RKS(mol)
mf1.xc = 'b3lyp'
chkfile1 = 'state1.chk'
mf1.chkfile = chkfile1
mf1.kernel()

# State 2: DFT calculation with slightly modified geometry, store to chkfile2
mol2 = mol.copy()
mol2.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 -0.757160 0.586260
H 0.000000 0.857160 0.586260
'''
mol2.build()
mf2 = dft.RKS(mol2)
mf2.xc = 'b3lyp'
chkfile2 = 'state2.chk'
mf2.chkfile = chkfile2
mf2.kernel()

# Read MO coefficients and occupations
from pyscf import lib
mo1 = lib.chkfile.load(chkfile1, 'scf/mo_coeff')
occ1 = lib.chkfile.load(chkfile1, 'scf/mo_occ')
mo2 = lib.chkfile.load(chkfile2, 'scf/mo_coeff')
occ2 = lib.chkfile.load(chkfile2, 'scf/mo_occ')

# Get number of electrons and occupied orbitals
nelec = mol.nelectron
nocc = np.count_nonzero(occ1 > 0.9)

# AO overlap between two geometries
s12 = gto.intor_cross('int1e_ovlp', mol, mol2)

# Overlap between two determinants
mo1_occ = mo1[:, occ1 > 0.9]
mo2_occ = mo2[:, occ2 > 0.9]
S_det = np.linalg.det(np.dot(mo1_occ.T, np.dot(s12, mo2_occ)))

# Construct density matrices
dm1 = mf1.make_rdm1()
dm2 = mf2.make_rdm1()

# One-electron integrals
h1e_1 = mf1.get_hcore()
h1e_2 = mf2.get_hcore()

# Two-electron integrals (Coulomb and exchange)
eri1 = mol.intor('int2e', aosym='s8')
eri2 = mol2.intor('int2e', aosym='s8')

# One-electron part of coupling
h12 = np.dot(mo1_occ.T, np.dot(h1e_1, mo2_occ))
s12_mo = np.dot(mo1_occ.T, np.dot(s12, mo2_occ))
H1e_coupling = np.trace(h12 @ np.linalg.inv(s12_mo))

# Two-electron part (approximate, using density matrices)
J12 = np.einsum('pqrs,pr,qs->', eri1, dm1, dm2)
K12 = np.einsum('prqs,pr,qs->', eri1, dm1, dm2)
H2e_coupling = 0.5 * (J12 - K12)

# Total coupling matrix element
H12 = H1e_coupling + H2e_coupling

# Diagonal energies
E1 = mf1.e_tot
E2 = mf2.e_tot

# Effective electronic coupling (assuming symmetric, two-state model)
V_eff = (H12 - S_det * (E1 + E2) / 2) / (1 - S_det**2)

print(f"Determinant overlap S12: {S_det:.6e}")
print(f"One-electron coupling: {H1e_coupling:.6e} Hartree")
print(f"Two-electron coupling: {H2e_coupling:.6e} Hartree")
print(f"Total coupling H12: {H12:.6e} Hartree")
print(f"State 1 energy: {E1:.6e} Hartree")
print(f"State 2 energy: {E2:.6e} Hartree")
print(f"Effective electronic coupling V_eff: {V_eff:.6e} Hartree")

# Remove chkfiles
os.remove(chkfile1)
os.remove(chkfile2)
```