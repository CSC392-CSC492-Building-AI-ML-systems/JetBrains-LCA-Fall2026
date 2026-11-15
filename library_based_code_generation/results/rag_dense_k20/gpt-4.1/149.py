```python
import os
import numpy as np
from simsopt._core import make_objective
from simsopt.field.coil import coils_via_symmetries, BiotSavart
from simsopt.field.coil import CurveLength
from simsopt.field.coil import CurveRZFourier
from simsopt.field.coil import Coil
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.field.vc import VirtualCasing
from simsopt.field.surface import SurfaceRZFourier
from simsopt.field import compute_surface_integral
from simsopt.objectives import SquaredFlux
from simsopt.optimize import minimize
from simsopt.util import TaylorTest

# Parameters
LENGTH_PENALTY = 1e-2
L0 = 10.0  # Reference length for penalty (arbitrary, adjust as needed)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load W7-X equilibrium surface (assume file exists)
eq_filename = "w7x_high_beta.nc"  # Replace with actual file path
surface = SurfaceRZFourier.from_wout(eq_filename, nphi=64, ntheta=64, quadpoints_phi=64, quadpoints_theta=64)

# Set average beta to 4% (if needed, adjust surface accordingly)
# (Assume surface is already at desired beta)

# Compute target B_external·n using virtual casing
vc = VirtualCasing(surface, eq_filename)
Bext_dot_n = vc.Bn()

# Load or create initial coils (e.g., circular coils)
ncoils = 5
coils = []
for i in range(ncoils):
    curve = CurveRZFourier(npoints=100, order=5, stellsym=True)
    curve.set("R0", 5.5 + 0.1 * i)
    curve.set("Z0", 0.0)
    coil = Coil(curve, current=1e5)
    coils.append(coil)

# Optionally, use symmetry
coils = coils_via_symmetries(coils, nfp=5, stellsym=True)

# Biot-Savart field from coils
bs = BiotSavart(coils)

# Objective: squared error in B·n plus length penalty
def objective():
    B_biot = bs.B(surface.gamma().reshape((-1, 3)))
    n = surface.normal()
    B_biot_dot_n = np.sum(B_biot * n, axis=1)
    diff = B_biot_dot_n - Bext_dot_n
    area = surface.area()
    J_Bn = 0.5 * np.sum(diff**2) * area / diff.size

    J_length = 0.0
    for coil in coils:
        length = CurveLength(coil.curve).J()
        J_length += 0.5 * (length - L0) ** 2
    J_length *= LENGTH_PENALTY

    return J_Bn + J_length

# Make objective for simsopt
def J_and_dJ(x):
    for i, coil in enumerate(coils):
        coil.curve.x = x[i*coil.curve.dof:(i+1)*coil.curve.dof]
    J = objective()
    # Use automatic differentiation for gradient
    dJ = np.zeros_like(x)
    eps = 1e-8
    for i in range(len(x)):
        x0 = x[i]
        x[i] = x0 + eps
        Jp = objective()
        x[i] = x0 - eps
        Jm = objective()
        x[i] = x0
        dJ[i] = (Jp - Jm) / (2 * eps)
    return J, dJ

# Flatten coil dofs
x0 = np.concatenate([coil.curve.x for coil in coils])

# Taylor test
def taylor_test():
    def fun(x):
        for i, coil in enumerate(coils):
            coil.curve.x = x[i*coil.curve.dof:(i+1)*coil.curve.dof]
        return objective()
    def grad(x):
        _, dJ = J_and_dJ(x)
        return dJ
    taylor = TaylorTest(fun, grad, x0)
    taylor.check()

taylor_test()

# Run optimization
res = minimize(J_and_dJ, x0, method="L-BFGS-B", jac=True, options={'maxiter': 200, 'disp': True})

# Save results
for i, coil in enumerate(coils):
    np.save(os.path.join(OUTPUT_DIR, f"coil_{i}_curve.npy"), coil.curve.x)
    np.save(os.path.join(OUTPUT_DIR, f"coil_{i}_current.npy"), np.array([coil.current]))
np.save(os.path.join(OUTPUT_DIR, "optimized_x.npy"), res.x)
with open(os.path.join(OUTPUT_DIR, "optimization_result.txt"), "w") as f:
    f.write(str(res))
```