```python
from pyscf import gto, scf, agf2, df
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
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
mf.conv_tol = 1e-10
mf.kernel()

# AGF2 calculation with density fitting
agf2_calc = agf2.AGF2(mf, verbose=0)
agf2_calc.conv_tol = 1e-7
agf2_calc.kernel()

# Print first three IPs and EAs
ips = agf2_calc.ipagf2(nroots=3)[0]
eas = agf2_calc.eaagf2(nroots=3)[0]
print("First three ionization potentials (eV):")
for i, ip in enumerate(ips):
    print(f"IP {i+1}: {ip*27.2114:.6f}")
print("First three electron affinities (eV):")
for i, ea in enumerate(eas):
    print(f"EA {i+1}: {ea*27.2114:.6f}")

# MO-basis density matrix
dm_mo = agf2_calc.make_rdm1()

# AO to MO transformation for dipole integrals
dip_ao = mol.intor_symmetric('int1e_r', comp=3)
mo_coeff = mf.mo_coeff
dip_mo = np.einsum('xij,ia,jb->xab', dip_ao, mo_coeff, mo_coeff)

# Dipole moment (electronic part)
mu_elec = -np.einsum('xab,ba->x', dip_mo, dm_mo)

# Nuclear dipole
charges = mol.atom_charges()
coords = mol.atom_coords()
mu_nuc = np.dot(charges, coords)

# Total dipole moment (in Debye)
mu_tot = mu_elec + mu_nuc
debye = 2.541746
mu_tot_debye = np.linalg.norm(mu_tot) * debye

print(f"Calculated dipole moment: {mu_tot_debye:.6f} Debye")
```