```python
import numpy as np
from pyscf import gto, scf, ao2mo, lib, tdscf

def get_molecule(mol_xyz, basis='sto-3g', charge=0, spin=0):
    mol = gto.Mole()
    mol.atom = mol_xyz
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol

def run_cis(mol):
    mf = scf.RHF(mol).run()
    cis = tdscf.TDHF(mf).run()
    return mf, cis

def get_mo_coeff(mf):
    return mf.mo_coeff

def get_cis_amplitudes(cis, state=0):
    # Returns (X, Y) amplitudes for the given excited state
    return cis.xy[state]

def get_2e_ao_integrals(molA, molB):
    # Compute 2e integrals between two molecules in AO basis
    # (μν|λσ) where μ,ν on molA, λ,σ on molB
    naoA = molA.nao_nr()
    naoB = molB.nao_nr()
    # Combine molecules into a supermolecule
    molAB = gto.Mole()
    molAB.atom = molA.atom + molB.atom
    molAB.basis = molA.basis
    molAB.charge = molA.charge + molB.charge
    molAB.spin = molA.spin + molB.spin
    molAB.build()
    eri = molAB.intor('int2e', aosym='s1')
    # Extract cross terms: (A,A|B,B)
    eri = eri[:naoA, :naoA, naoA:, naoA:]
    return eri

def transform_2e_ao_to_mo(eri_ao, moA, moB):
    # Transform (μν|λσ) to (ia|jb) where i,a on molA, j,b on molB
    # moA: (naoA, nmoA), moB: (naoB, nmoB)
    naoA, naoB = moA.shape[0], moB.shape[0]
    nmoA, nmoB = moA.shape[1], moB.shape[1]
    eri_mo = ao2mo.general(eri_ao, (moA, moA, moB, moB), compact=False)
    eri_mo = eri_mo.reshape(nmoA, nmoA, nmoB, nmoB)
    return eri_mo

def compute_J_coupling(eri_mo, XA, XB):
    # Coulomb term: J = sum_{ia,jb} XA_ia * XB_jb * (ia|jb)
    J = np.tensordot(XA, np.tensordot(XB, eri_mo, axes=([0,1],[2,3])), axes=([0,1],[0,1]))
    return J

def compute_K_coupling(eri_mo, XA, XB):
    # Exchange term: K = sum_{ia,jb} XA_ia * XB_jb * (ib|ja)
    eri_ex = eri_mo.transpose(0,3,2,1)  # (i,a,j,b) -> (i,b,j,a)
    K = np.tensordot(XA, np.tensordot(XB, eri_ex, axes=([0,1],[2,3])), axes=([0,1],[0,1]))
    return K

def compute_DFT_XC_coupling(*args, **kwargs):
    # Placeholder for DFT XC contribution, set to zero for HF/CIS
    return 0.0

def compute_coupling_matrix(molA_xyz, molB_xyz, basis='sto-3g'):
    # Build molecules
    molA = get_molecule(molA_xyz, basis)
    molB = get_molecule(molB_xyz, basis)
    # CIS calculations
    mfA, cisA = run_cis(molA)
    mfB, cisB = run_cis(molB)
    moA = get_mo_coeff(mfA)
    moB = get_mo_coeff(mfB)
    # Get CIS amplitudes for first excited state (singlet)
    XA, YA = get_cis_amplitudes(cisA, state=0)
    XB, YB = get_cis_amplitudes(cisB, state=0)
    # Use only X amplitudes for CIS
    # Get 2e integrals between molecules
    eri_ao = get_2e_ao_integrals(molA, molB)
    eri_mo = transform_2e_ao_to_mo(eri_ao, moA, moB)
    # Compute J and K
    J = compute_J_coupling(eri_mo, XA, XB)
    K = compute_K_coupling(eri_mo, XA, XB)
    # DFT XC contribution (zero for CIS)
    XC = compute_DFT_XC_coupling()
    # SET (singlet energy transfer) coupling: J + K + XC
    SET_coupling = J + K + XC
    # TET (triplet energy transfer) coupling: J - K + XC
    TET_coupling = J - K + XC
    # Return as a matrix
    coupling_matrix = np.array([[SET_coupling, 0],[0, TET_coupling]])
    return coupling_matrix

# Example usage:
molA_xyz = '''
O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587
'''
molB_xyz = '''
O 0.0 0.0 3.0
H 0.0 0.757 3.587
H 0.0 -0.757 3.587
'''

coupling_matrix = compute_coupling_matrix(molA_xyz, molB_xyz, basis='sto-3g')
print("Coupling matrix (SET, TET):")
print(coupling_matrix)
```