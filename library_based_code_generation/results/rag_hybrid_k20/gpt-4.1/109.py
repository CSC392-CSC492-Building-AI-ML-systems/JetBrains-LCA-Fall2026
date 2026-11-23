```python
import numpy as np
from pyscf import gto, scf, cc, pbc
from pyscf.pbc import gto as pbcgto, scf as pbcscf, cc as pbccc, tools
from pyscf.pbc.tools import super_cell

# Build a simple cell (e.g., diamond structure)
cell = pbcgto.Cell()
cell.atom = '''
C 0.000000 0.000000 0.000000
C 0.891700 0.891700 0.891700
'''
cell.a = '''
0.000000 1.783400 1.783400
1.783400 0.000000 1.783400
1.783400 1.783400 0.000000
'''
cell.unit = 'B'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 0
cell.build()

# Create a 2x1x1 supercell
scell = super_cell(cell, [2,1,1])

# Gamma-point mean-field calculation (molecular HF)
mf_gamma = pbcscf.RHF(scell, exxdiv=None)
mf_gamma.kernel()

# Gamma-point CCSD
cc_gamma = pbccc.CCSD(mf_gamma)
cc_gamma.kernel()
e_ip_gamma = cc_gamma.ipccsd(nroots=1)[0][0]
e_ea_gamma = cc_gamma.eaccsd(nroots=1)[0][0]

# K-point sampling
kmesh = [2,1,1]
kpts = scell.make_kpts(kmesh, space_group_symmetry=True)
kmf = pbcscf.KRHF(scell, kpts, exxdiv=None)
kmf.kernel()

# K-point CCSD
kcc = pbccc.KCCSD(kmf)
kcc.kernel()
e_ip_k = kcc.ipccsd(nroots=1)[0][0]
e_ea_k = kcc.eaccsd(nroots=1)[0][0]

# Calculate differences
print("Mean-field energy difference (gamma - k-point):", mf_gamma.e_tot - kmf.e_tot)
print("CCSD correlation energy difference (gamma - k-point):", cc_gamma.e_corr - kcc.e_corr)
print("IP-EOMCCSD energy difference (gamma - k-point):", e_ip_gamma - e_ip_k)
print("EA-EOMCCSD energy difference (gamma - k-point):", e_ea_gamma - e_ea_k)
```