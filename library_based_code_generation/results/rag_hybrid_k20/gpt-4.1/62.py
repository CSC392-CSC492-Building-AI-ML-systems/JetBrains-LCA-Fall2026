```python
import numpy as np
from pyscf import gto, scf, mcscf, dmrgscf, tools
from pyscf.nevpt2 import DMRG_NEVPT2

# Define Iron-Porphyrin molecule geometry (example geometry, replace with actual coordinates)
mol = gto.Mole()
mol.atom = '''
Fe      0.0000    0.0000    0.0000
N       1.9340    0.0000    0.0000
N      -1.9340    0.0000    0.0000
N       0.0000    1.9340    0.0000
N       0.0000   -1.9340    0.0000
C       2.8000    1.2000    0.0000
C      -2.8000    1.2000    0.0000
C       2.8000   -1.2000    0.0000
C      -2.8000   -1.2000    0.0000
C       1.2000    2.8000    0.0000
C      -1.2000    2.8000    0.0000
C       1.2000   -2.8000    0.0000
C      -1.2000   -2.8000    0.0000
'''
mol.basis = 'def2-svp'
mol.charge = 0
mol.spin = 4  # Quintet (2S = 4, S=2)
mol.symmetry = True
mol.build()

# SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Define DMET active space: Fe double d-shell (10), Fe 4s (1), 4x N 2pz (4) = 15 orbitals
# Number of electrons: Fe(II) is d6, so 6 (d) + 2 (4s) + 4x2 (N 2pz) = 16 electrons
ncas = 15
nelecas = 16

# Localize orbitals to select active space
from pyscf.tools import molden
from pyscf import lo

# Pipek-Mezey localization for easier selection
pm = lo.PM(mol, mf.mo_coeff)
loc_orbs = pm.kernel()

# Project to atomic orbitals to select Fe 3d, 4s, and N 2pz
ao_labels = mol.ao_labels()
fe_d_idx = [i for i, s in enumerate(ao_labels) if 'Fe' in s and '3d' in s]
fe_4s_idx = [i for i, s in enumerate(ao_labels) if 'Fe' in s and '4s' in s]
n_2pz_idx = [i for i, s in enumerate(ao_labels) if 'N ' in s and '2pz' in s]

# Select corresponding MO indices for active space
# For simplicity, select based on AO contributions
def select_active_orbitals(mol, mo_coeff, fe_d_idx, fe_4s_idx, n_2pz_idx, ncas):
    ao_labels = mol.ao_labels()
    nmo = mo_coeff.shape[1]
    weights = np.zeros(nmo)
    for i in range(nmo):
        for idx in fe_d_idx + fe_4s_idx + n_2pz_idx:
            weights[i] += np.sum(mo_coeff[:,i][idx]**2)
    # Select ncas orbitals with largest weights
    idx = np.argsort(weights)[-ncas:]
    idx.sort()
    return idx

active_idx = select_active_orbitals(mol, loc_orbs, fe_d_idx, fe_4s_idx, n_2pz_idx, ncas)

# Save active space orbitals to molden file
active_orbs = loc_orbs[:, active_idx]
molden.from_mo(mol, 'iron_porphyrin_active.molden', active_orbs)

# DMRG-CASSCF for Quintet (S=2, spin=4)
mol.spin = 4
mf = scf.RHF(mol)
mf.kernel()
mc_q = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_q.fcisolver.maxM = 1000
mc_q.mo_coeff = loc_orbs
mc_q.natorb = False
mc_q.cas_list = active_idx.tolist()
mc_q.kernel()
energy_q_dmrg_casscf = mc_q.e_tot

# DMRG-NEVPT2 for Quintet
nevpt2_q = DMRG_NEVPT2(mc_q)
energy_q_dmrg_nevpt2 = nevpt2_q.kernel()

# DMRG-CASSCF for Triplet (S=1, spin=2)
mol.spin = 2
mf = scf.RHF(mol)
mf.kernel()
mc_t = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_t.fcisolver.maxM = 1000
mc_t.mo_coeff = loc_orbs
mc_t.natorb = False
mc_t.cas_list = active_idx.tolist()
mc_t.kernel()
energy_t_dmrg_casscf = mc_t.e_tot

# DMRG-NEVPT2 for Triplet
nevpt2_t = DMRG_NEVPT2(mc_t)
energy_t_dmrg_nevpt2 = nevpt2_t.kernel()

# Output energy gaps
print("DMRG-CASSCF Quintet Energy: ", energy_q_dmrg_casscf)
print("DMRG-CASSCF Triplet Energy: ", energy_t_dmrg_casscf)
print("DMRG-CASSCF Triplet-Quintet Gap (eV): ", (energy_t_dmrg_casscf - energy_q_dmrg_casscf)*27.2114)

print("DMRG-NEVPT2 Quintet Energy: ", energy_q_dmrg_nevpt2)
print("DMRG-NEVPT2 Triplet Energy: ", energy_t_dmrg_nevpt2)
print("DMRG-NEVPT2 Triplet-Quintet Gap (eV): ", (energy_t_dmrg_nevpt2 - energy_q_dmrg_nevpt2)*27.2114)
```