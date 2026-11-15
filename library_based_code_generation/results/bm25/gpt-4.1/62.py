```python
import numpy as np
from pyscf import gto, scf, mcscf, dmrgscf, tools

# Define Iron-Porphyrin molecule geometry (example geometry, replace with actual coordinates)
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
C     1.2000    2.8000    0.0000
C    -1.2000    2.8000    0.0000
C     1.2000   -2.8000    0.0000
C    -1.2000   -2.8000    0.0000
'''
mol.basis = 'def2-svp'
mol.symmetry = True
mol.charge = 0
mol.spin = 4  # Quintet: 2S = 4, S=2
mol.build()

# SCF calculation
mf = scf.RHF(mol)
mf.kernel()

# Define DMET active space: Fe double d-shell (10), Fe 4s (1), 4 N 2pz (4) = 15 orbitals
# Number of electrons: Fe(II) is d6, so 6 (d) + 2 (s) + 4*2 (N 2pz) = 16 electrons
ncas = 15
nelecas = 16

# Localize orbitals to help select active space
from pyscf import lo
loc_orb = lo.Boys(mol, mf.mo_coeff).kernel()

# Select active space indices (example indices, adjust as needed for your system)
# Here, we assume the first 15 localized orbitals correspond to the desired active space
cas_list = list(range(ncas))

# DMRG settings
from pyscf.dmrgscf import DMRGCI

def run_dmrg_casscf(mol, mf, spin, ncas, nelecas, cas_list, state_label):
    mc = dmrgscf.DMRGSCF(mf, ncas, nelecas)
    mc.fcisolver = DMRGCI(mol, maxM=1000)
    mc.fcisolver.spin = spin
    mc.mo_coeff = loc_orb
    mc.natorb = False
    mc.cas_list = cas_list
    mc.kernel()
    tools.molden.from_mo(mol, f'iron_porphyrin_{state_label}_active.molden', mc.mo_coeff[:, cas_list])
    return mc.e_tot, mc

# Quintet state (S=2, spin=4)
mol.spin = 4
mf_quintet = scf.RHF(mol)
mf_quintet.kernel()
e_quintet, mc_quintet = run_dmrg_casscf(mol, mf_quintet, spin=4, ncas=ncas, nelecas=nelecas, cas_list=cas_list, state_label='quintet')

# Triplet state (S=1, spin=2)
mol.spin = 2
mf_triplet = scf.RHF(mol)
mf_triplet.kernel()
e_triplet, mc_triplet = run_dmrg_casscf(mol, mf_triplet, spin=2, ncas=ncas, nelecas=nelecas, cas_list=cas_list, state_label='triplet')

# DMRG-NEVPT2 corrections
from pyscf import mrpt

nevpt2_quintet = mrpt.NEVPT(mc_quintet)
e_quintet_nevpt2 = nevpt2_quintet.kernel()

nevpt2_triplet = mrpt.NEVPT(mc_triplet)
e_triplet_nevpt2 = nevpt2_triplet.kernel()

# Output results
print(f"DMRG-CASSCF Quintet Energy: {e_quintet:.8f} Ha")
print(f"DMRG-CASSCF Triplet Energy: {e_triplet:.8f} Ha")
print(f"DMRG-NEVPT2 Quintet Energy: {e_quintet_nevpt2:.8f} Ha")
print(f"DMRG-NEVPT2 Triplet Energy: {e_triplet_nevpt2:.8f} Ha")
print(f"Triplet-Quintet Gap (CASSCF): {(e_triplet - e_quintet)*27.2114:.4f} eV")
print(f"Triplet-Quintet Gap (NEVPT2): {(e_triplet_nevpt2 - e_quintet_nevpt2)*27.2114:.4f} eV")
```