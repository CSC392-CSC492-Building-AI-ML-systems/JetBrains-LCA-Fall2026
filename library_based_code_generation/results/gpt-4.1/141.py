```python
import numpy as np
from pyscf import gto, scf, mp

# Define QM molecule
mol = gto.Mole()
mol.atom = '''
O 0.000000 0.000000 0.000000
H 0.000000 -0.757000 0.587000
H 0.000000 0.757000 0.587000
'''
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()

# Generate random MM particles
n_mm = 5
np.random.seed(42)
mm_coords = np.random.uniform(-3, 3, (n_mm, 3))
mm_charges = np.random.uniform(-1, 1, n_mm)

# Function to calculate force from QM region on MM particles
def calc_qmmm_force(mol, mf, mm_coords, mm_charges, rdm1=None):
    # Get QM atom coords and charges
    qm_coords = mol.atom_coords()
    qm_charges = mol.atom_charges()
    n_mm = mm_coords.shape[0]
    n_qm = qm_coords.shape[0]
    # Nuclear (QM nuclei) - MM interaction
    force_nuc = np.zeros((n_mm, 3))
    for i in range(n_mm):
        for j in range(n_qm):
            R = mm_coords[i] - qm_coords[j]
            r = np.linalg.norm(R)
            if r > 1e-12:
                f = mm_charges[i] * qm_charges[j] * R / r**3
                force_nuc[i] += f
    # Electron density - MM interaction
    # Compute potential at MM positions due to electron density
    # The force is -q * grad(V), but grad(V) = sum_mu_nu D_mu_nu * grad(<mu|1/r|nu>)
    # pyscf can compute the necessary integrals
    if rdm1 is None:
        rdm1 = mf.make_rdm1()
    ao = mol.eval_gto('GTOval', mm_coords)
    # Compute electron density at MM positions
    # But for force, need gradient of potential at MM positions
    # Use pyscf's ao_potential and ao_potential_deriv
    from pyscf import lib
    force_elec = np.zeros((n_mm, 3))
    for i in range(n_mm):
        # <mu|1/|r-R_mm||nu>
        vmat = mol.intor('int1e_rinv', hermi=1, comp=3, grids=mm_coords[i:i+1])
        # vmat.shape = (3, nao, nao)
        # force = -q * Tr[rdm1 * grad(V)]
        f = -mm_charges[i] * np.einsum('xuv,uv->x', vmat, rdm1)
        force_elec[i] = f
    # Total force on MM particles from QM region
    force_total = force_nuc + force_elec
    return force_total

# Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Calculate force from HF electron density
force_hf = calc_qmmm_force(mol, mf, mm_coords, mm_charges)

# Verify: finite difference check for first MM particle
def finite_diff_force(mol, mf, mm_coords, mm_charges, idx=0, delta=1e-4):
    # Move MM particle idx by delta in each direction
    e0 = 0.0
    for j in range(mol.natm):
        R = mm_coords[idx] - mol.atom_coords()[j]
        r = np.linalg.norm(R)
        if r > 1e-12:
            e0 += mm_charges[idx] * mol.atom_charges()[j] / r
    # Electron density - MM
    vmat = mol.intor('int1e_rinv', hermi=1, grids=mm_coords[idx:idx+1])
    e0 -= mm_charges[idx] * np.einsum('uv,uv->', mf.make_rdm1(), vmat[0])
    grad_fd = np.zeros(3)
    for k in range(3):
        mm_coords_p = mm_coords.copy()
        mm_coords_p[idx, k] += delta
        e1 = 0.0
        for j in range(mol.natm):
            R = mm_coords_p[idx] - mol.atom_coords()[j]
            r = np.linalg.norm(R)
            if r > 1e-12:
                e1 += mm_charges[idx] * mol.atom_charges()[j] / r
        vmat = mol.intor('int1e_rinv', hermi=1, grids=mm_coords_p[idx:idx+1])
        e1 -= mm_charges[idx] * np.einsum('uv,uv->', mf.make_rdm1(), vmat[0])
        grad_fd[k] = (e1 - e0) / delta
    return grad_fd

fd_force = finite_diff_force(mol, mf, mm_coords, mm_charges, idx=0)
print("Analytical HF force on MM[0]:", force_hf[0])
print("Finite diff HF force on MM[0]:", fd_force)

# MP2 calculation
mp2 = mp.MP2(mf)
mp2.kernel()

# Function to make rdm1 with orbital response (MP2)
def make_mp2_rdm1_with_orbital_response(mp2obj):
    # Get canonical rdm1
    rdm1 = mp2obj.make_rdm1()
    # Add orbital response (z-vector)
    # Use pyscf's grad module to get z-vector
    from pyscf.grad import mp2 as mp2_grad
    gradobj = mp2_grad.Gradients(mp2obj)
    gradobj.kernel()
    # gradobj.z stores the orbital response
    # Build response part of rdm1
    mo_coeff = mp2obj._scf.mo_coeff
    mo_occ = mp2obj._scf.mo_occ
    nocc = np.count_nonzero(mo_occ > 0)
    nvir = mo_coeff.shape[1] - nocc
    zvec = gradobj.z
    # Orbital response contribution to rdm1
    # See Eq. (13) in JCP 118, 9095 (2003)
    rdm1_resp = np.zeros_like(rdm1)
    for p in range(nocc):
        for q in range(nvir):
            rdm1_resp += zvec[p, q] * (
                np.outer(mo_coeff[:, p], mo_coeff[:, q + nocc]) +
                np.outer(mo_coeff[:, q + nocc], mo_coeff[:, p])
            )
    rdm1_total = rdm1 + rdm1_resp
    return rdm1_total

# Calculate force from MP2 electron density (including orbital response)
rdm1_mp2 = make_mp2_rdm1_with_orbital_response(mp2)
force_mp2 = calc_qmmm_force(mol, mf, mm_coords, mm_charges, rdm1=rdm1_mp2)

# Verify: finite difference check for first MM particle at MP2 level
def finite_diff_force_mp2(mol, mf, mm_coords, mm_charges, idx=0, delta=1e-4):
    # Move MM particle idx by delta in each direction
    # Recompute MP2 energy
    e0 = 0.0
    for j in range(mol.natm):
        R = mm_coords[idx] - mol.atom_coords()[j]
        r = np.linalg.norm(R)
        if r > 1e-12:
            e0 += mm_charges[idx] * mol.atom_charges()[j] / r
    vmat = mol.intor('int1e_rinv', hermi=1, grids=mm_coords[idx:idx+1])
    mf_rdm1 = mf.make_rdm1()
    e0 -= mm_charges[idx] * np.einsum('uv,uv->', mf_rdm1, vmat[0])
    # Add MP2 correlation energy
    mp2 = mp.MP2(mf)
    mp2.kernel()
    e0 += mp2.e_corr
    grad_fd = np.zeros(3)
    for k in range(3):
        mm_coords_p = mm_coords.copy()
        mm_coords_p[idx, k] += delta
        e1 = 0.0
        for j in range(mol.natm):
            R = mm_coords_p[idx] - mol.atom_coords()[j]
            r = np.linalg.norm(R)
            if r > 1e-12:
                e1 += mm_charges[idx] * mol.atom_charges()[j] / r
        vmat = mol.intor('int1e_rinv', hermi=1, grids=mm_coords_p[idx:idx+1])
        mf_rdm1 = mf.make_rdm1()
        e1 -= mm_charges[idx] * np.einsum('uv,uv->', mf_rdm1, vmat[0])
        mp2 = mp.MP2(mf)
        mp2.kernel()
        e1 += mp2.e_corr
        grad_fd[k] = (e1 - e0) / delta
    return grad_fd

fd_force_mp2 = finite_diff_force_mp2(mol, mf, mm_coords, mm_charges, idx=0)
print("Analytical MP2 force on MM[0]: