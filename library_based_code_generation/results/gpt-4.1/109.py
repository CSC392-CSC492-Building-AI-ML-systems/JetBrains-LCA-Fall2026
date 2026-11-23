```python
import numpy as np
from pyscf import gto, scf, cc, pbc
from pyscf.pbc import gto as pbcgto, scf as pbcscf, cc as pbccc, tools
from pyscf.pbc.tools import super_cell
from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc.cc import kintermediates_rhf
from pyscf.pbc.cc import krestricted_uhf
from pyscf.pbc import df
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import lattice
from pyscf.pbc.tools import band_structure

# Build a simple cell
cell = pbcgto.Cell()
cell.atom = '''
H 0 0 0
H 0 0 0.74
'''
cell.a = np.eye(3) * 3.0
cell.unit = 'B'
cell.basis = 'sto-3g'
cell.verbose = 0
cell.build()

# Create a 2x1x1 supercell
scell = super_cell(cell, [2,1,1])

# Gamma-point mean-field calculation
mf_gamma = pbcscf.RHF(scell, exxdiv=None)
mf_gamma.kernel()

# Gamma-point CCSD
cc_gamma = pbccc.CCSD(mf_gamma)
cc_gamma.kernel()
ccsd_gamma_energy = cc_gamma.e_tot

# Gamma-point IP/EA-EOMCCSD
ip_eomccsd_gamma = cc_gamma.ipccsd(nroots=1)[0][0]
ea_eomccsd_gamma = cc_gamma.eaccsd(nroots=1)[0][0]

# K-point sampling
nk = [2,1,1]
kpts = scell.make_kpts(nk, wrap_around=True)

# K-point mean-field calculation
kmf = pbcscf.KRHF(scell, kpts, exxdiv=None)
kmf.kernel()

# K-point CCSD
kcc = pbccc.KCCSD(kmf)
kcc.kernel()
ccsd_k_energy = kcc.e_tot

# K-point IP/EA-EOMCCSD
ip_eomccsd_k = kcc.ipccsd(nroots=1)[0][0]
ea_eomccsd_k = kcc.eaccsd(nroots=1)[0][0]

# Calculate differences
print("Mean-field energy difference (gamma - k-point):", mf_gamma.e_tot - kmf.e_tot)
print("CCSD energy difference (gamma - k-point):", ccsd_gamma_energy - ccsd_k_energy)
print("IP-EOMCCSD difference (gamma - k-point):", ip_eomccsd_gamma - ip_eomccsd_k)
print("EA-EOMCCSD difference (gamma - k-point):", ea_eomccsd_gamma - ea_eomccsd_k)
```