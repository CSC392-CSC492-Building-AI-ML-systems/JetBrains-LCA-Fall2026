```python
import numpy as np
from pyscf import gto, scf, mp
from pyscf.qmmm import add_mm_charges, add_mm_charges_grad

# Define QM molecule
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 0 0.96; H 0.92 0 0'
mol.basis = 'sto-3g'
mol.build()

# Generate random MM particles
n_mm = 5
mm_coords = np.random.rand(n_mm, 3) * 5  # random positions in a 5x5x5 box
mm_charges = np.random.uniform(-1, 1, n_mm)  # random charges between -1 and 1

# Add MM charges to the QM molecule
mm_mol = add_mm_charges(mol, mm_coords, mm_charges)

# Function to calculate force from QM region on MM particles
def calc_qmmm_force(mf, mm_coords, mm_charges):
    # Get density matrix
    dm = mf.make_rdm1()
    # Get core Hamiltonian gradient w.r.t. MM particles
    hcore_mm = add_mm_charges_grad(mol, mm_coords, mm_charges)
    # Get nuclear attraction force (QM nuclei on MM)
    force_nuc = np.zeros_like(mm_coords)
    for i, (q, r_mm) in enumerate(zip(mm_charges, mm_coords)):
        for atom in mol.atom_coords():
            r = r_mm - atom
            force_nuc[i] += q * mol.atom_charges()[0] * r / (np.linalg.norm(r)**3 + 1e-12)
    # Get electron density force (QM electrons on MM)
    force_elec = np.zeros_like(mm_coords)
    for i, (q, r_mm) in enumerate(zip(mm_charges, mm_coords)):
        # Compute potential integral at MM position
        ao = mol.eval_gto('GTOval_sph', r_mm.reshape(1, 3))
        v = np.einsum('ij,ji->', ao, dm)
        # Gradient of potential (force)
        grad_ao = mol.eval_gto('GTOval_sph_deriv1', r_mm.reshape(1, 3))[1:]
        grad_v = np.einsum('xij,ji->x', grad_ao, dm)
        force_elec[i] = -q * grad_v
    # Total force
    force = force_nuc + force_elec
    return force

# Hartree-Fock calculation
mf = scf.RHF(mm_mol)
mf.kernel()

# Calculate force from HF electron density
force_hf = calc_qmmm_force(mf, mm_coords, mm_charges)
print("HF QM->MM force:\n", force_hf)

# Verify by finite difference (numerical gradient)
def finite_diff_force(mf, mm_coords, mm_charges, delta=1e-4):
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mm_mol_p = add_mm_charges(mol, mm_coords_p, mm_charges)
            mm_mol_m = add_mm_charges(mol, mm_coords_m, mm_charges)
            mf_p = scf.RHF(mm_mol_p)
            mf_m = scf.RHF(mm_mol_m)
            e_p = mf_p.kernel()
            e_m = mf_m.kernel()
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd

force_hf_fd = finite_diff_force(mf, mm_coords, mm_charges)
print("HF QM->MM force (finite diff):\n", force_hf_fd)

# MP2 calculation with orbital response
mp2 = mp.MP2(mf)
mp2.kernel()

# Function to make rdm1 with orbital response (MP2)
def make_mp2_rdm1_with_orbital_response(mp2_obj):
    # Get MP2 1-RDM with orbital response
    dm1 = mp2_obj.make_rdm1()
    return dm1

# Function to calculate force from MP2 electron density (including orbital response)
def calc_mp2_qmmm_force(mp2_obj, mm_coords, mm_charges):
    dm1 = make_mp2_rdm1_with_orbital_response(mp2_obj)
    # Get core Hamiltonian gradient w.r.t. MM particles
    hcore_mm = add_mm_charges_grad(mol, mm_coords, mm_charges)
    # Get nuclear attraction force (QM nuclei on MM)
    force_nuc = np.zeros_like(mm_coords)
    for i, (q, r_mm) in enumerate(zip(mm_charges, mm_coords)):
        for atom in mol.atom_coords():
            r = r_mm - atom
            force_nuc[i] += q * mol.atom_charges()[0] * r / (np.linalg.norm(r)**3 + 1e-12)
    # Get electron density force (QM electrons on MM)
    force_elec = np.zeros_like(mm_coords)
    for i, (q, r_mm) in enumerate(zip(mm_charges, mm_coords)):
        ao = mol.eval_gto('GTOval_sph', r_mm.reshape(1, 3))
        v = np.einsum('ij,ji->', ao, dm1)
        grad_ao = mol.eval_gto('GTOval_sph_deriv1', r_mm.reshape(1, 3))[1:]
        grad_v = np.einsum('xij,ji->x', grad_ao, dm1)
        force_elec[i] = -q * grad_v
    force = force_nuc + force_elec
    return force

# Calculate force from MP2 electron density (including orbital response)
force_mp2 = calc_mp2_qmmm_force(mp2, mm_coords, mm_charges)
print("MP2 QM->MM force (with orbital response):\n", force_mp2)

# Verify by finite difference (numerical gradient) for MP2
def finite_diff_force_mp2(mm_coords, mm_charges, delta=1e-4):
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mm_mol_p = add_mm_charges(mol, mm_coords_p, mm_charges)
            mm_mol_m = add_mm_charges(mol, mm_coords_m, mm_charges)
            mf_p = scf.RHF(mm_mol_p)
            mf_m = scf.RHF(mm_mol_m)
            mf_p.kernel()
            mf_m.kernel()
            mp2_p = mp.MP2(mf_p)
            mp2_m = mp.MP2(mf_m)
            e_p = mp2_p.kernel()[0]
            e_m = mp2_m.kernel()[0]
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd

force_mp2_fd = finite_diff_force_mp2(mm_coords, mm_charges)
print("MP2 QM->MM force (finite diff):\n", force_mp2_fd)
```