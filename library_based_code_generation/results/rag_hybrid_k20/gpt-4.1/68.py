```python
from pyscf import gto, scf, agf2, df, lib
import numpy as np

# Define molecule
mol = gto.Mole()
mol.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 -0.757160 0.586260
H 0.000000 0.757160 0.586260
'''
mol.basis = 'cc-pvdz'
mol.charge = 0
mol.spin = 0
mol.build()

# RHF with density fitting
auxbasis = 'cc-pvdz-jkfit'
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
mf.conv_tol = 1e-10
mf.kernel()

# AGF2 calculation with density fitting
agf2_calc = agf2.AGF2(mf, verbose=0, auxbasis='cc-pvdz-ri')
agf2_calc.conv_tol = 1e-7
agf2_calc.kernel()

# Print first three IPs and EAs
ips = agf2_calc.ipagf2(nroots=3)
eas = agf2_calc.eaagf2(nroots=3)
print("First three ionization potentials (eV):")
for ip in ips:
    print(ip * 27.2114)
print("First three electron affinities (eV):")
for ea in eas:
    print(ea * 27.2114)

# MO-basis density matrix
dm_mo = agf2_calc.make_rdm1()

# AO to MO transformation for dipole integrals
dip_ao = mol.intor_symmetric('int1e_r', comp=3)
mo_coeff = mf.mo_coeff
dip_mo = np.einsum('xij,ia,jb->xab', dip_ao, mo_coeff, mo_coeff)

# Dipole moment: electronic + nuclear
el_dip = -np.einsum('xab,ba->x', dip_mo, dm_mo)
nuc_dip = mol.atom_charges().dot(mol.atom_coords())
total_dip = el_dip + nuc_dip

print("Dipole moment (Debye):")
print(np.linalg.norm(total_dip) * 2.541746)
```