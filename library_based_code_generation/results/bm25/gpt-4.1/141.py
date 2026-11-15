```python
import numpy as np
from pyscf import gto, scf, grad, mp

# Define QM molecule
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 0 0.96; H 0 0.92 -0.24'
mol.basis = 'sto-3g'
mol.unit = 'Angstrom'
mol.build()

# Generate random MM particles: positions and charges
n_mm = 5
mm_coords = np.random.rand(n_mm, 3) * 5  # random positions within 5 Angstrom cube
mm_charges = np.random.uniform(-1, 1, n_mm)  # random charges between -1 and 1

# Add MM charges to molecule
mol.set_common_orig((0,0,0))
mol.set_rinv_orig((0,0,0))
mm_info = {'coords': mm_coords, 'charges': mm_charges}
mol.mm_coords = mm_coords
mol.mm_charges = mm_charges

# Function to calculate force from QM region on MM particles
def calc_qm_mm_force(mol, mf, mm_coords, mm_charges, dm=None):
    # Get nuclear coordinates and charges
    qm_coords = mol.atom_coords()
    qm_charges = mol.atom_charges()
    # Electron density matrix
    if dm is None:
        dm = mf.make_rdm1()
    # Get AO integrals
    ao_dip = mol.intor('int1e_r', comp=3)
    # Force on MM particles: F = -dE/dR_mm
    forces = np.zeros_like(mm_coords)
    for i, (r_mm, q_mm) in enumerate(zip(mm_coords, mm_charges)):
        # Interaction with QM nuclei
        dR = qm_coords - r_mm
        R = np.linalg.norm(dR, axis=1)
        f_nuc = np.sum(q_mm * qm_charges[:,None] * dR / (R[:,None]**3 + 1e-12), axis=0)
        # Interaction with electron density
        # Add MM charge at r_mm to mol, get potential at AO basis
        v_mm = mol.intor('int1e_nuc', hermi=1, aosym='s1', comp=3, origin=r_mm)
        f_elec = -q_mm * np.einsum('xij,ji->x', v_mm, dm)
        # Total force
        forces[i] = f_nuc + f_elec
    return forces

# Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Calculate force from HF electron density
forces_hf = calc_qm_mm_force(mol, mf, mm_coords, mm_charges)
print("HF QM->MM forces:\n", forces_hf)

# Verify with pyscf's grad_nuc_mm and grad_hcore_mm
# grad_nuc_mm: gradient of nuclear repulsion wrt MM positions
# grad_hcore_mm: gradient of electronic energy wrt MM positions
forces_nuc = grad.grad_nuc_mm(mol, mm_coords, mm_charges)
forces_hcore = grad.grad_hcore_mm(mol, mf.make_rdm1(), mm_coords, mm_charges)
forces_hf_verify = forces_nuc + forces_hcore
print("HF QM->MM forces (pyscf verify):\n", forces_hf_verify)

# MP2 calculation
mp2 = mp.MP2(mf)
mp2.kernel()

# Function to make rdm1 with orbital response (MP2)
def make_mp2_rdm1_with_orbital_response(mp2, mm_coords, mm_charges):
    # Get MP2 rdm1 with orbital response wrt MM charges
    # Use pyscf's grad.mp2.Gradients.make_rdm1
    mp2_grad = grad.mp2.Gradients(mp2)
    # The following is a hack to get the rdm1 response wrt MM charges
    # Use add_mm_charges_grad to get hcore gradient wrt MM
    hcore_mm = grad.grad_hcore_mm(mp2._scf.mol, mp2._scf.make_rdm1(), mm_coords, mm_charges)
    # Get orbital response part of rdm1
    rdm1_response = mp2_grad.make_rdm1(mp2._scf.mol, mo_coeff=mp2.mo_coeff, mo_occ=mp2.mo_occ)
    return rdm1_response

# Calculate force from MP2 electron density (including orbital response)
def calc_mp2_mm_force(mol, mp2, mm_coords, mm_charges):
    # Get rdm1 with orbital response
    rdm1_mp2 = make_mp2_rdm1_with_orbital_response(mp2, mm_coords, mm_charges)
    # Use same force calculation as before, but with MP2 rdm1
    mf = mp2._scf
    return calc_qm_mm_force(mol, mf, mm_coords, mm_charges, dm=rdm1_mp2)

forces_mp2 = calc_mp2_mm_force(mol, mp2, mm_coords, mm_charges)
print("MP2 QM->MM forces (with orbital response):\n", forces_mp2)

# Verify with pyscf's grad.mp2.Gradients
mp2_grad = grad.mp2.Gradients(mp2)
forces_mp2_verify = mp2_grad.grad_mm(mm_coords, mm_charges)
print("MP2 QM->MM forces (pyscf verify):\n", forces_mp2_verify)
```