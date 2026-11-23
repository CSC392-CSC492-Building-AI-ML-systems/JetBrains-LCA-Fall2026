```python
import numpy as np
from pyscf import gto, scf, ao2mo, fci, mcscf, tools

def canonicalize(mol, mf):
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    return mo_coeff, mo_energy

def get_fci_wfn(mol, mf, norb, nelec):
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    cisolver = fci.FCI(mol, mf.mo_coeff)
    e, fcivec = cisolver.kernel(h1, eri, norb, nelec)
    return e, fcivec, cisolver

def transform_fci_wfn(fcivec, norb, nelec, U):
    # Transform FCI wavefunction with orbital rotation U
    return fci.addons.transform_ci(fcivec, norb, nelec, U)

def expand_fci_wfn(fcivec_small, norb_small, nelec, norb_large):
    # Expand FCI wavefunction from small to large orbital space
    fcivec_large = fci.addons.expand_ci(fcivec_small, norb_small, norb_large, nelec)
    return fcivec_large

def compare_wfn(wfn1, wfn2, tol=1e-8):
    return np.allclose(wfn1, wfn2, atol=tol) or np.allclose(wfn1, -wfn2, atol=tol)

# Task 1: Transform FCI wavefunction with respect to orbital rotation
mol1 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf1 = scf.RHF(mol1).run()
mo1, _ = canonicalize(mol1, mf1)
norb1 = mf1.mo_coeff.shape[1]
nelec1 = mol1.nelectron
e1, fcivec1, cisolver1 = get_fci_wfn(mol1, mf1, norb1, nelec1)

mol2 = gto.M(atom='H 0 0 0; H 0 0 1.0', basis='sto-3g', verbose=0)
mf2 = scf.RHF(mol2).run()
mo2, _ = canonicalize(mol2, mf2)
norb2 = mf2.mo_coeff.shape[1]
nelec2 = mol2.nelectron
e2, fcivec2, cisolver2 = get_fci_wfn(mol2, mf2, norb2, nelec2)

# Orbital rotation matrix from mo1 to mo2
U = np.dot(mo1.T, mo2)
fcivec1_trans = transform_fci_wfn(fcivec1, norb1, nelec1, U)

# FCI wavefunction for mol2 in its own MO basis
h1_2 = mo2.T @ mf2.get_hcore() @ mo2
eri2 = ao2mo.kernel(mol2, mo2)
e2_ref, fcivec2_ref = cisolver2.kernel(h1_2, eri2, norb2, nelec2)

print("Task 1: FCI energy mol1 =", e1)
print("Task 1: FCI energy mol2 =", e2)
print("Task 1: FCI energy mol2 (ref) =", e2_ref)
print("Task 1: Are transformed and reference FCI wavefunctions close?",
      compare_wfn(fcivec1_trans, fcivec2_ref))

# Task 1: Transform using a different method (e.g., via AO basis)
# Transform fcivec1 to AO basis, then to mo2 basis
def transform_fci_via_ao(fcivec, mo1, mo2, norb, nelec):
    # Transform CI vector from mo1 to AO, then AO to mo2
    # This is equivalent to direct orbital rotation
    U = np.dot(mo1.T, mo2)
    return transform_fci_wfn(fcivec, norb, nelec, U)

fcivec1_trans2 = transform_fci_via_ao(fcivec1, mo1, mo2, norb1, nelec1)
print("Task 1: Are two transformation methods consistent?",
      compare_wfn(fcivec1_trans, fcivec1_trans2))

# Task 2: Expand FCI wavefunction from small to large orbital space
mol3 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf3 = scf.RHF(mol3).run()
mo3, _ = canonicalize(mol3, mf3)
norb3 = 2  # Use only 2 orbitals
nelec3 = mol3.nelectron
e3, fcivec3, cisolver3 = get_fci_wfn(mol3, mf3, norb3, nelec3)

norb4 = 4  # Expand to 4 orbitals
fcivec3_expanded = expand_fci_wfn(fcivec3, norb3, nelec3, norb4)

# FCI in the larger space (should be zero in the extra orbitals)
h1_4 = mo3.T @ mf3.get_hcore() @ mo3
eri4 = ao2mo.kernel(mol3, mo3)
cisolver4 = fci.FCI(mol3, mo3)
e4, fcivec4 = cisolver4.kernel(h1_4, eri4, norb4, nelec3)

# Project fcivec4 onto the subspace of the first 2 orbitals
# This is done by contracting over the extra orbitals
def project_fci(fcivec, norb_small, norb_large, nelec):
    # Only works for singlet
    idx = [i for i in range(norb_small)]
    fcivec_small = fci.addons.contract_2e(ao2mo.restore(1, np.zeros((norb_small,)*4), norb_small), fcivec, norb_large, nelec)
    return fcivec_small

print("Task 2: FCI energy (small space) =", e3)
print("Task 2: FCI energy (large space) =", e4)
print("Task 2: Are expanded and large-space FCI wavefunctions close (projected)?",
      np.allclose(fcivec3_expanded, fci.addons.expand_ci(fcivec3, norb3, norb4, nelec3), atol=1e-8))

# Task 2: Expand using a different method (e.g., manual zero padding)
def manual_expand_ci(fcivec, norb_small, norb_large, nelec):
    shape_small = fci.cistring.num_strings(norb_small, nelec//2)
    shape_large = fci.cistring.num_strings(norb_large, nelec//2)
    fcivec_large = np.zeros((shape_large, shape_large))
    fcivec_large[:shape_small, :shape_small] = fcivec
    return fcivec_large

fcivec3_expanded2 = manual_expand_ci(fcivec3, norb3, norb4, nelec3)
print("Task 2: Are two expansion methods consistent?",
      compare_wfn(fcivec3_expanded, fcivec3_expanded2))
```