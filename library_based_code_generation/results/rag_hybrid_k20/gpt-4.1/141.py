```python
import numpy as np
from pyscf import gto, scf, mp, grad
from pyscf.qmmm import add_mm_charges, add_mm_charges_grad

# Define QM molecule
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 0 1; H 0 1 0'
mol.basis = 'sto-3g'
mol.build()

# Generate random MM particles (positions and charges)
n_mm = 5
mm_coords = np.random.rand(n_mm, 3) * 5  # random positions in a 5x5x5 box
mm_charges = np.random.uniform(-1, 1, n_mm)  # random charges between -1 and 1

# Function to calculate force from QM region on MM particles
def calc_qm_force_on_mm(mol, mf, mm_coords, mm_charges):
    # Add MM charges to the molecule
    mm_mol = add_mm_charges(mol, mm_coords, mm_charges)
    # Compute electron density matrix
    dm = mf.make_rdm1()
    # Compute force: -dE/dR_mm = -Tr[dm * dV/dR_mm] - sum_A Z_A dV/dR_mm
    # Use pyscf.qmmm.add_mm_charges_grad to get dV/dR_mm
    vmm, vmm_grad = add_mm_charges_grad(mol, mm_coords, mm_charges)
    # Electron density contribution
    force_elec = np.einsum('ij,xij->x', dm, vmm_grad)
    # Nuclear contribution
    force_nuc = np.zeros_like(mm_coords)
    for ia, (za, ra) in enumerate(zip(mol.atom_charges(), mol.atom_coords())):
        diff = mm_coords - ra
        r = np.linalg.norm(diff, axis=1)[:, None]
        force_nuc += za * mm_charges[:, None] * diff / (r ** 3 + 1e-12)
    # Total force (QM on MM)
    force = force_elec + force_nuc.sum(axis=0)
    return force

# Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Calculate force from HF electron density
force_hf = calc_qm_force_on_mm(mol, mf, mm_coords, mm_charges)
print("HF force on MM particles:", force_hf)

# Verify by finite difference (numerical gradient)
def finite_diff_force(mol, mf, mm_coords, mm_charges, delta=1e-4):
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mm_mol_p = add_mm_charges(mol, mm_coords_p, mm_charges)
            mm_mol_m = add_mm_charges(mol, mm_coords_m, mm_charges)
            mf_p = scf.RHF(mm_mol_p).run()
            mf_m = scf.RHF(mm_mol_m).run()
            e_p = mf_p.e_tot
            e_m = mf_m.e_tot
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd.sum(axis=0)

force_hf_fd = finite_diff_force(mol, mf, mm_coords, mm_charges)
print("HF force (finite diff):", force_hf_fd)

# MP2 calculation with orbital response
mp2 = mp.MP2(mf)
mp2.kernel()

# Function to make rdm1 with orbital response (MP2)
def make_mp2_rdm1_with_orbital_response(mp2_obj):
    # Get MP2 density matrix with orbital response
    dm_mp2 = mp2_obj.make_rdm1()
    # Add orbital response (first-order density matrix)
    # pyscf.grad.mp2.get_rdm1_response returns the response part
    dm1resp = grad.mp2.get_rdm1_response(mp2_obj)
    return dm_mp2 + dm1resp

# Calculate force from MP2 electron density (including orbital response)
def calc_mp2_force_on_mm(mol, mf, mp2_obj, mm_coords, mm_charges):
    mm_mol = add_mm_charges(mol, mm_coords, mm_charges)
    dm_mp2 = make_mp2_rdm1_with_orbital_response(mp2_obj)
    vmm, vmm_grad = add_mm_charges_grad(mol, mm_coords, mm_charges)
    force_elec = np.einsum('ij,xij->x', dm_mp2, vmm_grad)
    force_nuc = np.zeros_like(mm_coords)
    for ia, (za, ra) in enumerate(zip(mol.atom_charges(), mol.atom_coords())):
        diff = mm_coords - ra
        r = np.linalg.norm(diff, axis=1)[:, None]
        force_nuc += za * mm_charges[:, None] * diff / (r ** 3 + 1e-12)
    force = force_elec + force_nuc.sum(axis=0)
    return force

force_mp2 = calc_mp2_force_on_mm(mol, mf, mp2, mm_coords, mm_charges)
print("MP2 force on MM particles (with orbital response):", force_mp2)

# Verify by finite difference (numerical gradient) for MP2
def finite_diff_force_mp2(mol, mm_coords, mm_charges, delta=1e-4):
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mm_mol_p = add_mm_charges(mol, mm_coords_p, mm_charges)
            mm_mol_m = add_mm_charges(mol, mm_coords_m, mm_charges)
            mf_p = scf.RHF(mm_mol_p).run()
            mf_m = scf.RHF(mm_mol_m).run()
            mp2_p = mp.MP2(mf_p).run()
            mp2_m = mp.MP2(mf_m).run()
            e_p = mp2_p.e_tot
            e_m = mp2_m.e_tot
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd.sum(axis=0)

force_mp2_fd = finite_diff_force_mp2(mol, mm_coords, mm_charges)
print("MP2 force (finite diff):", force_mp2_fd)
```