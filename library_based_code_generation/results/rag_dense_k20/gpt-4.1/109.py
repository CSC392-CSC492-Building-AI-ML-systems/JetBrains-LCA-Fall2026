```python
import numpy as np
from pyscf import gto, scf, cc, pbc
from pyscf.pbc import gto as pbcgto, scf as pbcscf, cc as pbccc, tools
from pyscf.pbc.tools import super_cell
from pyscf.pbc.scf import krhf
from pyscf.pbc.cc import kccsd
from pyscf.pbc.cc import eom_kccsd

# Build a simple cell (e.g., diamond)
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

# Create a 2x2x2 supercell
scell = super_cell.get_supercell(cell, np.diag([2,2,2]))

# Gamma-point mean-field calculation
mf_gamma = pbcscf.RHF(scell, exxdiv=None)
mf_gamma.kernel()

# Gamma-point CCSD
cc_gamma = pbccc.CCSD(mf_gamma)
cc_gamma.kernel()
eom_ip_gamma = cc_gamma.ipccsd(nroots=1)[0]
eom_ea_gamma = cc_gamma.eaccsd(nroots=1)[0]

# K-point sampling
nk = [2,2,2]
kpts = scell.make_kpts(nk, wrap_around=True)
kmf = krhf.KRHF(scell, kpts, exxdiv=None)
kmf.kernel()

# K-point CCSD
kcc = kccsd.KCCSD(kmf)
kcc.kernel()
eom_ip_k = eom_kccsd.KCISD_IP(kcc).kernel(nroots=1)[0]
eom_ea_k = eom_kccsd.KCISD_EA(kcc).kernel(nroots=1)[0]

# Calculate differences
mf_diff = mf_gamma.e_tot - kmf.e_tot
ccsd_diff = cc_gamma.e_tot - kcc.e_tot
ip_diff = eom_ip_gamma[0] - eom_ip_k[0]
ea_diff = eom_ea_gamma[0] - eom_ea_k[0]

print("Difference in mean-field (gamma - k-point):", mf_diff)
print("Difference in CCSD (gamma - k-point):", ccsd_diff)
print("Difference in IP-EOMCCSD (gamma - k-point):", ip_diff)
print("Difference in EA-EOMCCSD (gamma - k-point):", ea_diff)
```