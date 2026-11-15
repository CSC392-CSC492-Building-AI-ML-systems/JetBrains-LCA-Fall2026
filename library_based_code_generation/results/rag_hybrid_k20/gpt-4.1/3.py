```python
import os
import numpy as np
from pyscf import gto, dft, scf, lib

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

# Read MO coefficients and occupation numbers from chkfiles
mo1 = lib.chkfile.load('state1.chk', 'scf/mo_coeff')
occ1 = lib.chkfile.load('state1.chk', 'scf/mo_occ')
mo2 = lib.chkfile.load('state2.chk', 'scf/mo_coeff')
occ2 = lib.chkfile.load('state2.chk', 'scf/mo_occ')

# Calculate AO overlap matrix
s12 = gto.intor_cross('int1e_ovlp', mol, mol2)

# Calculate determinant overlap
# For closed-shell, use occupied orbitals only
occ_idx1 = np.where(occ1 > 0)[0]
occ_idx2 = np.where(occ2 > 0)[0]
mo1_occ = mo1[:, occ_idx1]
mo2_occ = mo2[:, occ_idx2]
S_det = np.linalg.det(np.dot(mo1_occ.T, np.dot(s12, mo2_occ)))

# Construct density matrices
dm1 = mf1.make_rdm1(mo_coeff=mo1, mo_occ=occ1)
dm2 = mf2.make_rdm1(mo_coeff=mo2, mo_occ=occ2)

# One-electron integrals
h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

# Two-electron integrals
eri = mol.intor('int2e', aosym='s8')

# One-electron part
E1 = np.einsum('ij,ji', h1e, dm1)
E2 = np.einsum('ij,ji', h1e, dm2)

# Two-electron part
def get_2e(dm, eri):
    # (ij|kl) dm_ik dm_jl - 0.5 * (ij|kl) dm_il dm_jk
    J = np.einsum('ijkl,ik,jl->', eri, dm, dm)
    K = np.einsum('ijkl,il,jk->', eri, dm, dm)
    return J - 0.5 * K

E1_2e = 0.5 * get_2e(dm1, eri)
E2_2e = 0.5 * get_2e(dm2, eri)

# Total energies (should match SCF energies)
E1_tot = E1 + E1_2e
E2_tot = E2 + E2_2e

# Off-diagonal Hamiltonian element (effective coupling)
# H12 = <D1|H|D2>
# One-electron part
h12 = np.einsum('ij,ji', h1e, 0.5*(dm1+dm2))
# Two-electron part (approximate with average density)
dm_avg = 0.5*(dm1+dm2)
H12_2e = 0.5 * get_2e(dm_avg, eri)
H12 = h12 + H12_2e

# Effective electronic coupling
# V = (H12 - S_det*(E1_tot + E2_tot)/2) / (1 - S_det**2)
V = (H12 - S_det * (E1_tot + E2_tot) / 2) / (1 - S_det**2)

print("Determinant overlap S12:", S_det)
print("State 1 energy:", E1_tot)
print("State 2 energy:", E2_tot)
print("Off-diagonal Hamiltonian element H12:", H12)
print("Effective electronic coupling V:", V)

# Remove chkfiles
os.remove('state1.chk')
os.remove('state2.chk')
```