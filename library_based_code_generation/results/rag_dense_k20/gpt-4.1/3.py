```python
import os
import numpy as np
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

# Perform DFT calculation for diabatic state 1
mf1 = dft.RKS(mol)
mf1.xc = 'b3lyp'
mf1.chkfile = 'diabatic1.chk'
mf1.kernel()

# Perform DFT calculation for diabatic state 2 (e.g., with permuted H atoms)
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
mf2.chkfile = 'diabatic2.chk'
mf2.kernel()

# Read MO coefficients and occupation numbers
mo1 = scf.chkfile.load('diabatic1.chk', 'mo_coeff')
occ1 = scf.chkfile.load('diabatic1.chk', 'mo_occ')
mo2 = scf.chkfile.load('diabatic2.chk', 'mo_coeff')
occ2 = scf.chkfile.load('diabatic2.chk', 'mo_occ')

# Get overlap matrix
s12 = gto.intor_cross('int1e_ovlp', mol, mol2)

# Select occupied orbitals
occ_idx1 = np.where(occ1 > 0)[0]
occ_idx2 = np.where(occ2 > 0)[0]
mo1_occ = mo1[:, occ_idx1]
mo2_occ = mo2[:, occ_idx2]

# Calculate determinant overlap
S = np.dot(mo1_occ.T, np.dot(s12, mo2_occ))
det_overlap = np.linalg.det(S)

# Construct density matrices
dm1 = np.dot(mo1_occ, mo1_occ.T) * 2
dm2 = np.dot(mo2_occ, mo2_occ.T) * 2

# One-electron integrals
h1e_1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
h1e_2 = mol2.intor('int1e_kin') + mol2.intor('int1e_nuc')

# Two-electron integrals
eri1 = mol.intor('int2e', aosym='s8')
eri2 = mol2.intor('int2e', aosym='s8')

# One-electron contributions
E1_1e = np.einsum('ij,ji', h1e_1, dm1)
E2_1e = np.einsum('ij,ji', h1e_2, dm2)

# Two-electron contributions
E1_2e = np.einsum('ijkl,ij,kl', eri1, dm1, dm1) * 0.5 * 0.5
E2_2e = np.einsum('ijkl,ij,kl', eri2, dm2, dm2) * 0.5 * 0.5

# Total energies
E1 = E1_1e + E1_2e + mol.energy_nuc()
E2 = E2_1e + E2_2e + mol2.energy_nuc()

# Off-diagonal Hamiltonian element (effective coupling)
# One-electron part
h12 = mol.intor_cross('int1e_kin', mol, mol2) + mol.intor_cross('int1e_nuc', mol, mol2)
H1e_12 = np.einsum('ij,ji', h12, np.dot(mo1_occ, mo2_occ.T)) * det_overlap

# Two-electron part (approximate, using density matrices)
eri12 = mol.intor_cross('int2e', mol, mol2)
H2e_12 = 0.5 * np.einsum('ijkl,ij,kl', eri12, np.dot(mo1_occ, mo2_occ.T), np.dot(mo1_occ, mo2_occ.T)) * det_overlap

H12 = H1e_12 + H2e_12

# Effective electronic coupling
H11 = E1
H22 = E2
S12 = det_overlap
J = (H12 - S12 * (H11 + H22) / 2) / (1 - S12**2)

print(f"Determinant overlap S12: {S12:.6e}")
print(f"H11: {H11:.6f} Eh")
print(f"H22: {H22:.6f} Eh")
print(f"H12: {H12:.6e} Eh")
print(f"Effective electronic coupling J: {J:.6e} Eh")

# Remove chkfiles
os.remove('diabatic1.chk')
os.remove('diabatic2.chk')
```