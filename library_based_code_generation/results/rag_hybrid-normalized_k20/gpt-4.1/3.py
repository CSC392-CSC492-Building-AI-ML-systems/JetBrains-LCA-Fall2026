```python
import os
import numpy as np
from pyscf import gto, dft, scf, lib

# Define molecule
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

# Perform DFT calculation for diabatic state 1
mf1 = dft.RKS(mol)
mf1.xc = 'b3lyp'
mf1.chkfile = 'state1.chk'
mf1.kernel()

# Perform DFT calculation for diabatic state 2 (swap H atoms for diabaticity)
mol2 = gto.Mole()
mol2.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 0.757160 0.586260
H 0.000000 -0.757160 0.586260
'''
mol2.basis = 'sto-3g'
mol2.charge = 0
mol2.spin = 0
mol2.build()

mf2 = dft.RKS(mol2)
mf2.xc = 'b3lyp'
mf2.chkfile = 'state2.chk'
mf2.kernel()

# Read MO coefficients and occupation numbers
mo1 = lib.chkfile.load('state1.chk', 'scf/mo_coeff')
occ1 = lib.chkfile.load('state1.chk', 'scf/mo_occ')
mo2 = lib.chkfile.load('state2.chk', 'scf/mo_coeff')
occ2 = lib.chkfile.load('state2.chk', 'scf/mo_occ')

# Overlap matrix in AO basis
s12 = gto.intor_cross('int1e_ovlp', mol, mol2)

# Get occupied orbitals
occ_idx1 = np.where(occ1 > 0)[0]
occ_idx2 = np.where(occ2 > 0)[0]
mo1_occ = mo1[:, occ_idx1]
mo2_occ = mo2[:, occ_idx2]

# Overlap between determinants (Slater-Condon rule)
S_mo = np.dot(mo1_occ.T, np.dot(s12, mo2_occ))
det_overlap = np.linalg.det(S_mo)

# Construct density matrices
dm1 = mf1.make_rdm1()
dm2 = mf2.make_rdm1()

# One-electron integrals
h1e_12 = gto.intor_cross('int1e_kin', mol, mol2) + gto.intor_cross('int1e_nuc', mol, mol2)
h1e_21 = gto.intor_cross('int1e_kin', mol2, mol) + gto.intor_cross('int1e_nuc', mol2, mol)

# Two-electron integrals (approximate using AO density matrices)
eri_12 = gto.intor_cross('int2e', mol, mol2)
J12 = np.einsum('pqrs,rs->pq', eri_12, dm2)
K12 = np.einsum('prqs,rs->pq', eri_12, dm2)
E2e_12 = 0.5 * (np.trace(np.dot(dm1, J12)) - 0.5 * np.trace(np.dot(dm1, K12)))

# One-electron part
E1e_12 = np.trace(np.dot(dm1, h1e_12))

# Total off-diagonal Hamiltonian element
H12 = E1e_12 + E2e_12

# Effective electronic coupling
if abs(det_overlap) > 1e-8:
    V_eff = H12 / det_overlap
else:
    V_eff = 0.0

print("Determinant overlap S12:", det_overlap)
print("One-electron contribution H1e_12:", E1e_12)
print("Two-electron contribution H2e_12:", E2e_12)
print("Total off-diagonal Hamiltonian H12:", H12)
print("Effective electronic coupling V_eff:", V_eff)

# Remove chkfiles
os.remove('state1.chk')
os.remove('state2.chk')
```