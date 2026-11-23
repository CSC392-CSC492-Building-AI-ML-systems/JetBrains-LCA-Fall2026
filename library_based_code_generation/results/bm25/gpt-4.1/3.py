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

# State 1: DFT calculation, save to chkfile1
mf1 = dft.RKS(mol)
mf1.xc = 'b3lyp'
mf1.kernel()
chkfile1 = 'state1.chk'
mf1.chkfile = chkfile1
mf1.dump_chk()

# State 2: Slightly modify geometry for diabatic state, DFT calculation, save to chkfile2
mol2 = mol.copy()
mol2.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 -0.757160 0.586260
H 0.000000 0.757160 0.786260
'''
mol2.build()
mf2 = dft.RKS(mol2)
mf2.xc = 'b3lyp'
mf2.kernel()
chkfile2 = 'state2.chk'
mf2.chkfile = chkfile2
mf2.dump_chk()

# Read MO coefficients and occupations from chkfiles
mo1 = scf.chkfile.load(chkfile1, 'scf/mo_coeff')
occ1 = scf.chkfile.load(chkfile1, 'scf/mo_occ')
mo2 = scf.chkfile.load(chkfile2, 'scf/mo_coeff')
occ2 = scf.chkfile.load(chkfile2, 'scf/mo_occ')

# Get number of electrons and occupied orbitals
nelec = mol.nelectron
nocc = np.count_nonzero(occ1 > 0)

# Calculate overlap matrix S
s = mol.intor('int1e_ovlp')

# Calculate determinant overlap
# For closed-shell, use occupied orbitals only
mo1_occ = mo1[:, occ1 > 0]
mo2_occ = mo2[:, occ2 > 0]
S12 = np.dot(mo1_occ.T, np.dot(s, mo2_occ))
det_overlap = np.linalg.det(S12)

# Construct density matrices
dm1 = np.dot(mo1_occ, mo1_occ.T) * 2
dm2 = np.dot(mo2_occ, mo2_occ.T) * 2

# One-electron integrals
h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

# Two-electron integrals (chemist's notation)
eri = mol.intor('int2e', aosym='s8')

# Calculate one-electron part
E1 = np.sum(h1 * dm1)
E2 = np.sum(h1 * dm2)
E12 = np.sum(h1 * (dm1 + dm2) / 2)

# Calculate two-electron part
def get_2e(dm, eri):
    # (ij|kl) dm_ik dm_jl
    return 0.5 * np.einsum('pqrs,pq,rs->', eri, dm, dm)

E1_2e = get_2e(dm1, eri)
E2_2e = get_2e(dm2, eri)
E12_2e = get_2e((dm1 + dm2) / 2, eri)

# Total energies
E1_tot = E1 + E1_2e
E2_tot = E2 + E2_2e
E12_tot = E12 + E12_2e

# Effective electronic coupling (assuming symmetric diabatic states)
Hab = (E12_tot - (E1_tot + E2_tot) / 2) / det_overlap

print("Determinant overlap S12:", det_overlap)
print("E1 (state 1 energy):", E1_tot)
print("E2 (state 2 energy):", E2_tot)
print("E12 (off-diagonal energy):", E12_tot)
print("Effective electronic coupling (Hab):", Hab)

# Remove chkfiles
os.remove(chkfile1)
os.remove(chkfile2)
```