```python
import numpy as np
from pyscf import gto, scf, mcscf, dmrgscf, tools
from pyscf.nevpt2 import DMRG_NEVPT2

# Define Iron-Porphyrin molecule geometry (example coordinates)
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
mol.symmetry = False
mol.charge = 0
mol.spin = 4  # Quintet (2S = 4, S=2)
mol.verbose = 4
mol.build()

# SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Define DMET active space: Fe 3d (5), Fe 4d (5), Fe 4s (1), N 2pz (4)
# Total: 5+5+1+4 = 15 orbitals
# Number of electrons: Fe (3d6 4s2), N (2pz^1 each, 4 N), so 6+2+4=12
ncas = 15
nelecas = 12

# DMRG settings
dmrg_kwargs = {'maxM': 1000, 'tol': 1e-6}

# Helper function to select active space orbitals
def select_active_orbitals(mol, mf, ncas):
    # Use Pipek-Mezey localization to help select Fe d, s and N 2pz orbitals
    from pyscf import lo
    pm = lo.PM(mol, mf.mo_coeff)
    loc_orbs = pm.kernel()
    # For demonstration, select the first ncas orbitals
    # In practice, select based on population analysis or visual inspection
    cas_list = list(range(ncas))
    return cas_list

cas_list = select_active_orbitals(mol, mf, ncas)

# Quintet state calculation
mol.spin = 4  # Quintet
mf = scf.RHF(mol)
mf.kernel()
mc_quintet = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_quintet.fcisolver.maxM = dmrg_kwargs['maxM']
mc_quintet.fcisolver.tol = dmrg_kwargs['tol']
mc_quintet.cas_list = cas_list
mc_quintet.kernel()
tools.molden.from_mo(mol, 'iron_porphyrin_quintet_active.molden', mc_quintet.mo_coeff[:, mc_quintet.ncore:mc_quintet.ncore+mc_quintet.ncas])
e_quintet_casscf = mc_quintet.e_tot
nevpt2_quintet = DMRG_NEVPT2(mc_quintet)
e_quintet_nevpt2 = nevpt2_quintet.kernel()

# Triplet state calculation
mol.spin = 2  # Triplet
mf = scf.RHF(mol)
mf.kernel()
mc_triplet = dmrgscf.DMRGSCF(mf, ncas, nelecas)
mc_triplet.fcisolver.maxM = dmrg_kwargs['maxM']
mc_triplet.fcisolver.tol = dmrg_kwargs['tol']
mc_triplet.cas_list = cas_list
mc_triplet.kernel()
tools.molden.from_mo(mol, 'iron_porphyrin_triplet_active.molden', mc_triplet.mo_coeff[:, mc_triplet.ncore:mc_triplet.ncore+mc_triplet.ncas])
e_triplet_casscf = mc_triplet.e_tot
nevpt2_triplet = DMRG_NEVPT2(mc_triplet)
e_triplet_nevpt2 = nevpt2_triplet.kernel()

# Output energy gaps
print("DMRG-CASSCF Quintet Energy: ", e_quintet_casscf)
print("DMRG-CASSCF Triplet Energy: ", e_triplet_casscf)
print("DMRG-CASSCF Triplet-Quintet Gap (kcal/mol): ", (e_triplet_casscf - e_quintet_casscf) * 627.509)
print("DMRG-NEVPT2 Quintet Energy: ", e_quintet_nevpt2)
print("DMRG-NEVPT2 Triplet Energy: ", e_triplet_nevpt2)
print("DMRG-NEVPT2 Triplet-Quintet Gap (kcal/mol): ", (e_triplet_nevpt2 - e_quintet_nevpt2) * 627.509)
```