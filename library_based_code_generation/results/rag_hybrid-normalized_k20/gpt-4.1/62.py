```python
import numpy as np
from pyscf import gto, scf, mcscf, dmrgscf, ao2mo, tools
from pyscf.nevpt2 import DMRG_NEVPT2

# Define Iron-Porphyrin molecule geometry (example geometry, replace with actual)
mol = gto.Mole()
mol.atom = '''
Fe    0.0000    0.0000    0.0000
N     1.9340    0.0000    0.0000
N    -1.9340    0.0000    0.0000
N     0.0000    1.9340    0.0000
N     0.0000   -1.9340    0.0000
C     2.8000    1.2000    0.0000
C     2.8000   -1.2000    0.0000
C    -2.8000    1.2000    0.0000
C    -2.8000   -1.2000    0.0000
... (rest of porphyrin ring)
'''
mol.basis = 'def2-svp'
mol.symmetry = True
mol.charge = 0
mol.spin = 4  # Quintet as default
mol.build()

# SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Define DMET active space: Fe double d-shell (10), Fe 4s (2), 4 N 2pz (4)
# Total: 10 (d) + 2 (4s) + 4 (N 2pz) = 16 orbitals
# Number of electrons: Fe (d6) + 2 (4s) + 4x2 (N 2pz) = 6 + 2 + 8 = 16
ncas = 16
nelecas = 16

# Localize orbitals to identify Fe d, 4s, and N 2pz
from pyscf import lo
pm = lo.PM(mol, mf.mo_coeff)
loc_orbs = pm.kernel()

# Project to atomic orbitals to select active space
from pyscf.mcscf.addons import project_init_guess
ao_labels = mol.ao_labels()
fe_d_idx = [i for i, l in enumerate(ao_labels) if l[0] == 'Fe' and l[1] == '3d']
fe_4s_idx = [i for i, l in enumerate(ao_labels) if l[0] == 'Fe' and l[1] == '4s']
n_2pz_idx = [i for i, l in enumerate(ao_labels) if l[0] == 'N' and l[2] == 'pz']

active_idx = fe_d_idx + fe_4s_idx + n_2pz_idx
active_idx = list(set(active_idx))
active_idx.sort()

# Project localized orbitals to get initial guess for active space
init_guess = project_init_guess(mol, loc_orbs, active_idx)

# DMRG settings
from pyscf.dmrgscf import DMRGCI
dmrg_kwargs = {'maxM': 1000, 'tol': 1e-6}

# --------- Quintet State Calculation ---------
mol.spin = 4  # 2S = 4 => S=2, quintet
mol.build(False, False)
mf = scf.RHF(mol)
mf.kernel()

mc_q = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_q.fcisolver = DMRGCI(mol, **dmrg_kwargs)
mc_q.mo_coeff = init_guess
mc_q.fix_spin_(ss=6.0)  # S(S+1) for S=2
mc_q.kernel()

# DMRG-NEVPT2 for quintet
nevpt2_q = DMRG_NEVPT2(mc_q)
e_nevpt2_q = nevpt2_q.kernel()

# Output active space orbitals to molden
tools.molden.from_mo(mol, 'iron_porphyrin_quintet_active.molden', mc_q.mo_coeff[:, mc_q.ncore:mc_q.ncore+mc_q.ncas])

# --------- Triplet State Calculation ---------
mol.spin = 2  # 2S = 2 => S=1, triplet
mol.build(False, False)
mf = scf.RHF(mol)
mf.kernel()

mc_t = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_t.fcisolver = DMRGCI(mol, **dmrg_kwargs)
mc_t.mo_coeff = init_guess
mc_t.fix_spin_(ss=2.0)  # S(S+1) for S=1
mc_t.kernel()

# DMRG-NEVPT2 for triplet
nevpt2_t = DMRG_NEVPT2(mc_t)
e_nevpt2_t = nevpt2_t.kernel()

# Output active space orbitals to molden
tools.molden.from_mo(mol, 'iron_porphyrin_triplet_active.molden', mc_t.mo_coeff[:, mc_t.ncore:mc_t.ncore+mc_t.ncas])

# --------- Energy Gap Calculation ---------
print("DMRG-CASSCF Quintet Energy: ", mc_q.e_tot)
print("DMRG-NEVPT2 Quintet Energy: ", e_nevpt2_q)
print("DMRG-CASSCF Triplet Energy: ", mc_t.e_tot)
print("DMRG-NEVPT2 Triplet Energy: ", e_nevpt2_t)
print("DMRG-CASSCF Triplet-Quintet Gap (Ha): ", mc_t.e_tot - mc_q.e_tot)
print("DMRG-NEVPT2 Triplet-Quintet Gap (Ha): ", e_nevpt2_t - e_nevpt2_q)
```