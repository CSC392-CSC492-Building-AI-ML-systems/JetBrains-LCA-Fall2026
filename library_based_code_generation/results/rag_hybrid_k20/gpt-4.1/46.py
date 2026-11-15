```python
import numpy as np
from pyscf import gto, scf, ao2mo, fci
from pyscf.fci.addons import transform_ci_for_orbital_rotation, states_transform_ci_for_orbital_rotation
from pyscf.fci.addons import to_fci, from_fci

# Task 1: Transform FCI wavefunction with respect to orbital rotation/transformation

# Molecule 1
mol1 = gto.Mole()
mol1.atom = 'H 0 0 0; H 0 0 0.74'
mol1.basis = 'sto-3g'
mol1.spin = 0
mol1.build()
mf1 = scf.RHF(mol1).run()
norb1 = mf1.mo_coeff.shape[1]
nelec1 = mol1.nelectron

# Molecule 2 (different atomic configuration)
mol2 = gto.Mole()
mol2.atom = 'H 0 0 0; H 0 0 1.0'
mol2.basis = 'sto-3g'
mol2.spin = 0
mol2.build()
mf2 = scf.RHF(mol2).run()
norb2 = mf2.mo_coeff.shape[1]
nelec2 = mol2.nelectron

# FCI calculation for both molecules
cisolver1 = fci.FCI(mol1, mf1.mo_coeff)
e1, ci1 = cisolver1.kernel()
cisolver2 = fci.FCI(mol2, mf2.mo_coeff)
e2, ci2 = cisolver2.kernel()

# Transform ci1 from mf1.mo_coeff to mf2.mo_coeff
mo1 = mf1.mo_coeff
mo2 = mf2.mo_coeff
orb_rot = np.dot(mo1.T, mo2)
ci1_trans = transform_ci_for_orbital_rotation(ci1, norb1, nelec1, orb_rot)

# Compare transformed ci1 with ci2 (up to a sign)
ci2_proj = np.dot(ci2.ravel(), ci1_trans.ravel())
print("FCI energy molecule 1:", e1)
print("FCI energy molecule 2:", e2)
print("Overlap between transformed ci1 and ci2:", ci2_proj)
print("Are transformed ci1 and ci2 close?", np.allclose(np.abs(ci1_trans), np.abs(ci2), atol=1e-6))

# Alternative transformation using states_transform_ci_for_orbital_rotation
ci1_trans_alt = states_transform_ci_for_orbital_rotation([ci1], norb1, nelec1, orb_rot)[0]
print("Are both transformation methods close?", np.allclose(np.abs(ci1_trans), np.abs(ci1_trans_alt), atol=1e-8))

# Task 2: Transfer FCI wavefunction from a smaller orbital space to a larger one

# Small molecule (minimal basis)
mol_small = gto.Mole()
mol_small.atom = 'H 0 0 0; H 0 0 0.74'
mol_small.basis = 'sto-3g'
mol_small.spin = 0
mol_small.build()
mf_small = scf.RHF(mol_small).run()
norb_small = mf_small.mo_coeff.shape[1]
nelec_small = mol_small.nelectron

cisolver_small = fci.FCI(mol_small, mf_small.mo_coeff)
e_small, ci_small = cisolver_small.kernel()

# Large molecule (same geometry, larger basis)
mol_large = gto.Mole()
mol_large.atom = 'H 0 0 0; H 0 0 0.74'
mol_large.basis = '6-31g'
mol_large.spin = 0
mol_large.build()
mf_large = scf.RHF(mol_large).run()
norb_large = mf_large.mo_coeff.shape[1]
nelec_large = mol_large.nelectron

# Expand ci_small to larger orbital space
ci_large = to_fci(ci_small, norb_small, nelec_small, norb_large, nelec_large)

# FCI calculation in large space
cisolver_large = fci.FCI(mol_large, mf_large.mo_coeff)
e_large, ci_large_ref = cisolver_large.kernel()

# Compare expanded wavefunction with FCI solver result (projected to the same space)
# Project ci_large_ref to small space for comparison
ci_large_proj = from_fci(ci_large_ref, norb_large, nelec_large, norb_small, nelec_small)
overlap = np.dot(ci_small.ravel(), ci_large_proj.ravel())
print("FCI energy small basis:", e_small)
print("FCI energy large basis:", e_large)
print("Overlap between ci_small and projected ci_large_ref:", overlap)
print("Are expanded ci_small and ci_large_ref close (in small space)?", np.allclose(np.abs(ci_small), np.abs(ci_large_proj), atol=1e-6))

# Alternative expansion using from_fci and to_fci
ci_large_alt = to_fci(ci_small, norb_small, nelec_small, norb_large, nelec_large)
print("Are both expansion methods close?", np.allclose(np.abs(ci_large), np.abs(ci_large_alt), atol=1e-8))
```