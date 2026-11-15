```python
import numpy as np
from pyscf import gto, scf, mp
from pyscf.qmmm import qmmm_for_scf, qmmm_grad_for_scf

# Define QM molecule
mol = gto.Mole()
mol.atom = '''
O 0.0 0.0 0.0
H 0.0 -0.757 0.587
H 0.0 0.757 0.587
'''
mol.basis = 'sto-3g'
mol.build()

# Generate random MM particles (positions and charges)
n_mm = 5
mm_coords = np.random.uniform(-5, 5, (n_mm, 3))
mm_charges = np.random.uniform(-1, 1, n_mm)

# Attach MM particles to QM molecule
mol.set_common_orig((0,0,0))
mf = scf.RHF(mol)
mf = qmmm_for_scf(mf, mm_coords, mm_charges)

# SCF calculation
mf.kernel()

# Function to calculate force from QM region on MM particles
def calc_qmmm_force(mf, mm_coords, mm_charges):
    # Get QM atom coords and charges
    qm_coords = mf.mol.atom_coords()
    qm_charges = mf.mol.atom_charges()
    # Get electron density matrix
    dm = mf.make_rdm1()
    # Get AO basis at MM positions
    ao_mm = mf.mol.eval_gto('GTOval', mm_coords)
    # Nuclear force on MM: F = q_mm * q_qm * (r_mm - r_qm) / |r_mm - r_qm|^3
    force_nuc = np.zeros_like(mm_coords)
    for i, (mmc, mmq) in enumerate(zip(mm_coords, mm_charges)):
        for qc, qq in zip(qm_coords, qm_charges):
            r = mmc - qc
            dist = np.linalg.norm(r)
            if dist > 1e-10:
                force_nuc[i] += mmq * qq * r / dist**3
    # Electron force on MM: F = -q_mm * <rho|grad(1/|r-r_mm|)>
    # Use pyscf.qmmm.grad_hcore_mm to get electron force
    force_elec = qmmm_grad_for_scf(mf, mm_coords, mm_charges, dm)
    # Total force
    force_total = force_nuc + force_elec
    return force_total

# Calculate and print force from HF electron density
force_hf = calc_qmmm_force(mf, mm_coords, mm_charges)
print("Force from HF electron density on MM particles:\n", force_hf)

# Verify by finite difference (numerical gradient)
def finite_diff_force(mf, mm_coords, mm_charges, delta=1e-4):
    e0 = mf.energy_tot()
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mf_p = scf.RHF(mol)
            mf_p = qmmm_for_scf(mf_p, mm_coords_p, mm_charges)
            mf_p.kernel()
            e_p = mf_p.energy_tot()
            mf_m = scf.RHF(mol)
            mf_m = qmmm_for_scf(mf_m, mm_coords_m, mm_charges)
            mf_m.kernel()
            e_m = mf_m.energy_tot()
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd

force_fd = finite_diff_force(mf, mm_coords, mm_charges)
print("Finite difference force (HF):\n", force_fd)

# MP2 calculation
mp2 = mp.MP2(mf)
mp2.kernel()

# Function to make rdm1 with orbital response (MP2)
def make_mp2_rdm1_with_orbital_response(mp2):
    # Get MP2 rdm1 with orbital response
    dm1 = mp2.make_rdm1()
    return dm1

# Calculate force from MP2 electron density (including orbital response)
def calc_qmmm_force_mp2(mf, mp2, mm_coords, mm_charges):
    # Get QM atom coords and charges
    qm_coords = mf.mol.atom_coords()
    qm_charges = mf.mol.atom_charges()
    # Get MP2 electron density matrix with orbital response
    dm = make_mp2_rdm1_with_orbital_response(mp2)
    # Nuclear force on MM
    force_nuc = np.zeros_like(mm_coords)
    for i, (mmc, mmq) in enumerate(zip(mm_coords, mm_charges)):
        for qc, qq in zip(qm_coords, qm_charges):
            r = mmc - qc
            dist = np.linalg.norm(r)
            if dist > 1e-10:
                force_nuc[i] += mmq * qq * r / dist**3
    # Electron force on MM
    force_elec = qmmm_grad_for_scf(mf, mm_coords, mm_charges, dm)
    force_total = force_nuc + force_elec
    return force_total

force_mp2 = calc_qmmm_force_mp2(mf, mp2, mm_coords, mm_charges)
print("Force from MP2 electron density (with orbital response) on MM particles:\n", force_mp2)

# Verify by finite difference (numerical gradient) for MP2
def finite_diff_force_mp2(mf, mm_coords, mm_charges, delta=1e-4):
    mf.kernel()
    mp2 = mp.MP2(mf)
    mp2.kernel()
    e0 = mp2.e_tot
    force_fd = np.zeros_like(mm_coords)
    for i in range(len(mm_coords)):
        for j in range(3):
            mm_coords_p = mm_coords.copy()
            mm_coords_m = mm_coords.copy()
            mm_coords_p[i, j] += delta
            mm_coords_m[i, j] -= delta
            mf_p = scf.RHF(mf.mol)
            mf_p = qmmm_for_scf(mf_p, mm_coords_p, mm_charges)
            mf_p.kernel()
            mp2_p = mp.MP2(mf_p)
            mp2_p.kernel()
            e_p = mp2_p.e_tot
            mf_m = scf.RHF(mf.mol)
            mf_m = qmmm_for_scf(mf_m, mm_coords_m, mm_charges)
            mf_m.kernel()
            mp2_m = mp.MP2(mf_m)
            mp2_m.kernel()
            e_m = mp2_m.e_tot
            force_fd[i, j] = -(e_p - e_m) / (2 * delta)
    return force_fd

force_fd_mp2 = finite_diff_force_mp2(mf, mm_coords, mm_charges)
print("Finite difference force (MP2):\n", force_fd_mp2)
```