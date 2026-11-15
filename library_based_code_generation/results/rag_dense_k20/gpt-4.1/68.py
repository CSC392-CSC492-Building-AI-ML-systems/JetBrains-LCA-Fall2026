```python
from pyscf import gto, scf, df, agf2
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

# RHF calculation with density fitting and specified convergence tolerance
auxbasis = 'cc-pvdz-ri'
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
mf.conv_tol = 1e-10
mf.kernel()

# AGF2 calculation with density fitting and specified convergence tolerance
agf2_calc = agf2.RAGF2(mf)
agf2_calc.conv_tol = 1e-7
agf2_calc.kernel()

# Print first three ionization potentials and electron affinities
ip = agf2_calc.ipagf2()[:3]
ea = agf2_calc.eaagf2()[:3]
print("First three ionization potentials (eV):")
for i, val in enumerate(ip):
    print(f"IP {i+1}: {val*27.2114:.6f}")
print("First three electron affinities (eV):")
for i, val in enumerate(ea):
    print(f"EA {i+1}: {val*27.2114:.6f}")

# Calculate MO-basis density matrix
dm = agf2_calc.make_rdm1()

# AO dipole integrals
dip_ao = mol.intor_symmetric('int1e_r', comp=3)

# Transform dipole integrals to MO basis
mo_coeff = mf.mo_coeff
dip_mo = np.einsum('xij,ia,jb->xab', dip_ao, mo_coeff, mo_coeff)

# Electronic dipole moment (in a.u.)
mu_elec = -np.einsum('xab,ba->x', dip_mo, dm)

# Nuclear dipole moment (in a.u.)
charges = mol.atom_charges()
coords = mol.atom_coords()
mu_nuc = np.dot(charges, coords)

# Total dipole moment (in a.u.)
mu_tot = mu_elec + mu_nuc

# Convert to Debye
debye = 2.541746
mu_tot_debye = np.linalg.norm(mu_tot) * debye

print(f"Calculated dipole moment: {mu_tot_debye:.6f} Debye")
```