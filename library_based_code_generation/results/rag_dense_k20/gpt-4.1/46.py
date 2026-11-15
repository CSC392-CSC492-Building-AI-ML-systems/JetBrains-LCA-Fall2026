```python
import numpy as np
from pyscf import gto, scf, fci, ao2mo
from pyscf.fci.addons import transform_ci_for_orbital_rotation, states_transform_ci_for_orbital_rotation

# Task 1: Transform FCI wavefunction with respect to orbital rotation/transformation

# Molecule 1
mol1 = gto.Mole()
mol1.atom = 'H 0 0 0; H 0 0 0.74'
mol1.basis = 'sto-3g'
mol1.build()
mf1 = scf.RHF(mol1).run()
norb1 = mf1.mo_coeff.shape[1]
nelec1 = mol1.nelectron

cisolver1 = fci.FCI(mol1, mf1.mo_coeff)
e_fci1, ci1 = cisolver1.kernel()

# Molecule 2 (rotated/translated)
mol2 = gto.Mole()
mol2.atom = 'H 0 0 0; H 0 0 1.0'
mol2.basis = 'sto-3g'
mol2.build()
mf2 = scf.RHF(mol2).run()
norb2 = mf2.mo_coeff.shape[1]
nelec2 = mol2.nelectron

cisolver2 = fci.FCI(mol2, mf2.mo_coeff)
e_fci2, ci2 = cisolver2.kernel()

# Orbital rotation matrix from mol1 to mol2
# Project mol1's MOs onto mol2's AO basis
S12 = np.dot(mol1.intor('int1e_ovlp'), mol2.intor('int1e_ovlp').T)
U = np.dot(mf1.mo_coeff.T, np.dot(S12, mf2.mo_coeff))
# Orthonormalize U
U, _ = np.linalg.qr(U)

# Transform ci1 to mol2's MO basis
ci1_trans = transform_ci_for_orbital_rotation(ci1, norb1, nelec1, U)

# Compare with ci2 (up to a sign/phase)
ci1_trans_norm = np.linalg.norm(ci1_trans)
ci2_norm = np.linalg.norm(ci2)
overlap = np.dot(ci1_trans.ravel(), ci2.ravel()) / (ci1_trans_norm * ci2_norm)
print("Task 1: Overlap between transformed ci1 and ci2:", overlap)
print("Task 1: Are they close?", np.allclose(np.abs(overlap), 1.0, atol=1e-6))

# Alternative transformation using states_transform_ci_for_orbital_rotation
ci1_trans_alt = states_transform_ci_for_orbital_rotation([ci1], norb1, nelec1, U)[0]
overlap_alt = np.dot(ci1_trans_alt.ravel(), ci2.ravel()) / (np.linalg.norm(ci1_trans_alt) * ci2_norm)
print("Task 1: Overlap (alternative method):", overlap_alt)
print("Task 1: Are they close (alt)?", np.allclose(np.abs(overlap_alt), 1.0, atol=1e-6))


# Task 2: Transfer FCI wavefunction from a smaller orbital space to a larger one

# Small basis molecule
mol_small = gto.Mole()
mol_small.atom = 'H 0 0 0; H 0 0 0.74'
mol_small.basis = 'sto-3g'
mol_small.build()
mf_small = scf.RHF(mol_small).run()
norb_small = mf_small.mo_coeff.shape[1]
nelec_small = mol_small.nelectron

cisolver_small = fci.FCI(mol_small, mf_small.mo_coeff)
e_fci_small, ci_small = cisolver_small.kernel()

# Large basis molecule (same geometry, larger basis)
mol_large = gto.Mole()
mol_large.atom = 'H 0 0 0; H 0 0 0.74'
mol_large.basis = '6-31g'
mol_large.build()
mf_large = scf.RHF(mol_large).run()
norb_large = mf_large.mo_coeff.shape[1]
nelec_large = mol_large.nelectron

cisolver_large = fci.FCI(mol_large, mf_large.mo_coeff)
e_fci_large, ci_large = cisolver_large.kernel()

# Project small basis MOs into large basis AO space
S = gto.intor_cross('int1e_ovlp', mol_small, mol_large)
U_expand = np.dot(mf_small.mo_coeff.T, np.dot(S, mf_large.mo_coeff))
# Orthonormalize U_expand
U_expand, _ = np.linalg.qr(U_expand)

# Expand ci_small to large orbital space
# Pad with zeros for the extra orbitals
def expand_ci(ci, norb_from, norb_to, nelec):
    from pyscf.fci.addons import _unpack_nelec
    nelec = _unpack_nelec(nelec)
    from pyscf.fci import direct_spin1
    ci_expanded = np.zeros([direct_spin1.num_strings(norb_to, nelec[0]),
                            direct_spin1.num_strings(norb_to, nelec[1])])
    for i in range(ci.shape[0]):
        for j in range(ci.shape[1]):
            ci_expanded[i, j] = ci[i, j]
    return ci_expanded

ci_small_expanded = expand_ci(ci_small, norb_small, norb_large, nelec_large)

# Transform expanded ci_small to large basis MO space
ci_small_trans = transform_ci_for_orbital_rotation(ci_small_expanded, norb_large, nelec_large, U_expand)

# Compare with ci_large
ci_small_trans_norm = np.linalg.norm(ci_small_trans)
ci_large_norm = np.linalg.norm(ci_large)
overlap2 = np.dot(ci_small_trans.ravel(), ci_large.ravel()) / (ci_small_trans_norm * ci_large_norm)
print("Task 2: Overlap between transformed ci_small and ci_large:", overlap2)
print("Task 2: Are they close?", np.allclose(np.abs(overlap2), 1.0, atol=1e-6))

# Alternative transformation using states_transform_ci_for_orbital_rotation
ci_small_trans_alt = states_transform_ci_for_orbital_rotation([ci_small_expanded], norb_large, nelec_large, U_expand)[0]
overlap2_alt = np.dot(ci_small_trans_alt.ravel(), ci_large.ravel()) / (np.linalg.norm(ci_small_trans_alt) * ci_large_norm)
print("Task 2: Overlap (alternative method):", overlap2_alt)
print("Task 2: Are they close (alt)?", np.allclose(np.abs(overlap2_alt), 1.0, atol=1e-6))
```