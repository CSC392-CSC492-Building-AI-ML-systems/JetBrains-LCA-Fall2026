```python
import os
import numpy as np
from simsopt._core import load
from simsopt.field.coil import CoilSet, curves_to_coils
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.virtualcasing import VirtualCasing
from simsopt.field.coil_optimization import CoilObjectiveFunction
from simsopt.field.coil_optimization import TaylorTest
from simsopt.field.coil_optimization import coil_optimization
from simsopt.field.surface import SurfaceRZFourier
from simsopt.util.mpi import MPI
from simsopt.util.logger import logger

# Parameters
LENGTH_PENALTY = 1e-2
L0 = 10.0  # Reference length for penalty, adjust as needed
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load W7-X equilibrium (assume VMEC file is available)
vmec_filename = "w7x_high_beta.vmec"  # Path to VMEC file with <beta> ~ 4%
eq = load(vmec_filename)

# Create target surface (e.g., last closed flux surface)
nphi = 64
ntheta = 64
surface = SurfaceRZFourier.from_vmec_surfaces(vmec_filename, range(1, 2), nphi=nphi, ntheta=ntheta)[0]

# Compute target B_external·n using virtual casing
vc = VirtualCasing(eq, surface, nphi=nphi, ntheta=ntheta)
Bext_dot_n = vc.Bn()

# Initialize coils (e.g., Fourier curves, 5 coils)
ncoils = 5
order = 8
coils = []
for i in range(ncoils):
    curve = load("input/curve{}.json".format(i+1))  # Assume initial curves are provided
    coil = curves_to_coils([curve])[0]
    coils.append(coil)
coilset = CoilSet(coils)

# Biot-Savart field from coils
bs = BiotSavart(coilset)

# Objective function: normal field error + length penalty
def objective():
    B_biot_n = bs.Bn(surface)
    normal_error = 0.5 * np.sum((B_biot_n - Bext_dot_n)**2 * surface.area())
    length_penalty = 0.0
    for coil in coilset.coils:
        length_penalty += 0.5 * (coil.length() - L0)**2
    length_penalty *= LENGTH_PENALTY
    return normal_error + length_penalty

# Simsopt objective function wrapper
class StageIICoilObjective(CoilObjectiveFunction):
    def __init__(self, bs, surface, Bext_dot_n, coilset, length_penalty, L0):
        super().__init__(coilset)
        self.bs = bs
        self.surface = surface
        self.Bext_dot_n = Bext_dot_n
        self.length_penalty = length_penalty
        self.L0 = L0

    def J(self):
        B_biot_n = self.bs.Bn(self.surface)
        normal_error = 0.5 * np.sum((B_biot_n - self.Bext_dot_n)**2 * self.surface.area())
        length_penalty = 0.0
        for coil in self.coilset.coils:
            length_penalty += 0.5 * (coil.length() - self.L0)**2
        length_penalty *= self.length_penalty
        return normal_error + length_penalty

    def dJ(self):
        # Use automatic differentiation provided by simsopt
        return super().dJ()

# Instantiate objective
obj = StageIICoilObjective(bs, surface, Bext_dot_n, coilset, LENGTH_PENALTY, L0)

# Taylor test
taylor_test = TaylorTest(obj)
taylor_test.run()
with open(os.path.join(OUTPUT_DIR, "taylor_test.txt"), "w") as f:
    f.write(str(taylor_test.results))

# Run optimization
result = coil_optimization(
    obj,
    method="L-BFGS-B",
    options={"maxiter": 200, "disp": True},
    output_dir=OUTPUT_DIR
)

# Save results
coilset.save(os.path.join(OUTPUT_DIR, "optimized_coils.json"))
np.save(os.path.join(OUTPUT_DIR, "Bext_dot_n.npy"), Bext_dot_n)
np.save(os.path.join(OUTPUT_DIR, "B_biot_n.npy"), bs.Bn(surface))
```