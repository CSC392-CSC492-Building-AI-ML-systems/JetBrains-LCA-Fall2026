```python
import numpy as np
from pyscf import gto, scf, mcscf, dmrgscf, tools, lo
from pyscf.nevpt2 import DMRG_NEVPT2

# Define Iron-Porphyrin molecule geometry (example geometry)
mol = gto.Mole()
mol.atom = '''
Fe      0.000000    0.000000    0.000000
N       1.934000    0.000000    0.000000
N      -1.934000    0.000000    0.000000
N       0.000000    1.934000    0.000000
N       0.000000   -1.934000    0.000000
C       2.800000    1.200000    0.000000
C       2.800000   -1.200000    0.000000
C      -2.800000    1.200000    0.000000
C      -2.800000   -1.200000    0.000000
C       1.200000    2.800000    0.000000
C      -1.200000    2.800000    0.000000
C       1.200000   -2.800000    0.000000
C      -1.200000   -2.800000    0.000000
'''
mol.basis = 'def2-svp'
mol.spin = 0  # will be set for each state
mol.charge = 0
mol.symmetry = False
mol.build()

# SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Localize orbitals to help select active space
pm = lo.PM(mol, mf.mo_coeff)
loc_orbs = pm.kernel()

# Define DMET active space: Fe double d-shell (10), Fe 4s (2), 4 N 2pz (4)
# Find orbital indices for Fe 3d, 4d, 4s, and N 2pz
# For simplicity, select by AO labels (in practice, use population analysis or visualization)
ao_labels = mol.ao_labels()
fe_d_idx = [i for i, s in enumerate(ao_labels) if 'Fe' in s and ('3d' in s or '4d' in s)]
fe_4s_idx = [i for i, s in enumerate(ao_labels) if 'Fe' in s and '4s' in s]
n_2pz_idx = []
for i, s in enumerate(ao_labels):
    if 'N' in s and '2pz' in s:
        n_2pz_idx.append(i)
# If 2pz not labeled, use '2p' and z axis
if len(n_2pz_idx) < 4:
    n_2p_idx = [i for i, s in enumerate(ao_labels) if 'N' in s and '2p' in s]
    # Assume z is the third p orbital for each N
    n_2pz_idx = n_2p_idx[2::3]

active_ao_idx = fe_d_idx + fe_4s_idx + n_2pz_idx
# Project localized orbitals onto selected AO indices to get active orbitals
active_orbs = loc_orbs[:, active_ao_idx]
n_act_orb = active_orbs.shape[1]
n_act_ele_triplet = 6*2 + 2 + 4*2  # Fe d (10e), 4s (2e), N 2pz (8e) = 20e
n_act_ele_quintet = n_act_ele_triplet

# Save active orbitals to molden
tools.molden.from_mo(mol, 'active_space.molden', active_orbs)

def run_dmrg_casscf(mol, mf, n_elec, n_orb, spin, active_orbs):
    mol.spin = spin
    mf = scf.RHF(mol)
    mf.kernel()
    mc = dmrgscf.DMRGSCF(mf, n_orb, n_elec)
    mc.fcisolver.maxM = 1000
    mc.fcisolver.nproc = 4
    mc.mo_coeff = mcscf.sort_mo(mc, mf.mo_coeff, active_orbs)
    mc.kernel()
    return mc

def run_dmrg_nevpt2(mc):
    nevpt2 = DMRG_NEVPT2(mc)
    e_corr = nevpt2.kernel()
    return e_corr

# Quintet state (S=2, spin=4)
mc_quintet = run_dmrg_casscf(mol, mf, n_act_ele_quintet, n_act_orb, 4, active_orbs)
e_quintet_casscf = mc_quintet.e_tot
e_quintet_nevpt2 = e_quintet_casscf + run_dmrg_nevpt2(mc_quintet)

# Triplet state (S=1, spin=2)
mc_triplet = run_dmrg_casscf(mol, mf, n_act_ele_triplet, n_act_orb, 2, active_orbs)
e_triplet_casscf = mc_triplet.e_tot
e_triplet_nevpt2 = e_triplet_casscf + run_dmrg_nevpt2(mc_triplet)

# Output results
print(f"DMRG-CASSCF Quintet Energy: {e_quintet_casscf:.8f} Ha")
print(f"DMRG-CASSCF Triplet Energy: {e_triplet_casscf:.8f} Ha")
print(f"DMRG-CASSCF Triplet-Quintet Gap: {(e_triplet_casscf - e_quintet_casscf)*27.2114:.4f} eV")
print(f"DMRG-NEVPT2 Quintet Energy: {e_quintet_nevpt2:.8f} Ha")
print(f"DMRG-NEVPT2 Triplet Energy: {e_triplet_nevpt2:.8f} Ha")
print(f"DMRG-NEVPT2 Triplet-Quintet Gap: {(e_triplet_nevpt2 - e_quintet_nevpt2)*27.2114:.4f} eV")
```