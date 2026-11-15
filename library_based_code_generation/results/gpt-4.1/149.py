```python
import os
import numpy as np
from simsopt._core import load
from simsopt.field.coil import Coil, coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil_collection import CoilCollection
from simsopt.field.virtualcasing import VirtualCasing
from simsopt.geo.curve import curves_to_vjp, curves_to_dcurve_length
from simsopt.objectives import QuadraticPenalty
from simsopt.util import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.geo.surface import SurfaceRZFourier
from simsopt.optimize import minimize

# Parameters
LENGTH_PENALTY = 1e-2
L0 = 10.0  # Reference length for penalty (meters)
NCOILS = 5
NFP = 5
COILPOINTS = 100
SURFACE_NPHI = 64
SURFACE_NTHETA = 64
VMEC_INPUT = "w7x_high_beta.input"  # Path to W7-X VMEC input file
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load VMEC equilibrium (W7-X, <beta> ~ 4%)
vmec = Vmec(VMEC_INPUT)
vmec.run()
surf = vmec.boundary

# 2. Create target surface for B·n calculation
surface = SurfaceRZFourier(
    surf.mpol, surf.ntor, surf.rc.copy(), surf.zc.copy(),
    nphi=SURFACE_NPHI, ntheta=SURFACE_NTHETA
)

# 3. Compute target B_external·n using virtual casing
vc = VirtualCasing(vmec, surface)
Bext = vc.Bnormal()

# 4. Initialize coils (Fourier curves)
from simsopt.geo.curve import CurveRZFourier
curves = []
for i in range(NCOILS):
    curve = CurveRZFourier(
        nfp=NFP, stellsym=True, quadpoints=COILPOINTS, order=5
    )
    curve.set("rc(0)", 10.0 + 0.5 * np.cos(2 * np.pi * i / NCOILS))
    curve.set("zc(0)", 0.0 + 0.5 * np.sin(2 * np.pi * i / NCOILS))
    curves.append(curve)
coils = [Coil(curve) for curve in curves]
coils = coils_via_symmetries(coils, NFP, True)
coil_collection = CoilCollection(coils)

# 5. Biot-Savart field from coils
bs = BiotSavart(coil_collection)

# 6. Objective function
def objective(x):
    coil_collection.set_dof(x)
    Bn_biot = bs.Bnormal(surface)
    diff = Bn_biot - Bext
    area = surface.area()
    J_Bn = 0.5 * np.sum(diff**2) * area / diff.size
    # Length penalty
    lengths = np.array([c.length() for c in coil_collection.coils])
    J_len = 0.5 * LENGTH_PENALTY * np.sum((lengths - L0)**2)
    return J_Bn + J_len

def grad_objective(x):
    coil_collection.set_dof(x)
    Bn_biot = bs.Bnormal(surface)
    diff = Bn_biot - Bext
    area = surface.area()
    # dJ/dx for Bn term
    dBndx = bs.dBnormal_by_dcoil(surface)
    grad_Bn = area / diff.size * np.dot(diff, dBndx)
    # dJ/dx for length penalty
    lengths = np.array([c.length() for c in coil_collection.coils])
    dlengthdx = np.concatenate([c.dlength_by_dcoeff() for c in coil_collection.coils])
    grad_len = LENGTH_PENALTY * np.dot(lengths - L0, dlengthdx)
    return grad_Bn + grad_len

# 7. Taylor test
def taylor_test(x0, v, epsilons=[1e-6, 1e-7, 1e-8]):
    f0 = objective(x0)
    g0 = grad_objective(x0)
    for eps in epsilons:
        f1 = objective(x0 + eps * v)
        fdiff = f1 - f0
        lin = eps * np.dot(g0, v)
        print(f"eps={eps:.1e}, f1-f0={fdiff:.3e}, linear={lin:.3e}, ratio={(fdiff/lin):.3f}")

# 8. Run Taylor test
x0 = coil_collection.get_dof()
v = np.random.randn(x0.size)
taylor_test(x0, v)

# 9. Run optimization
res = minimize(
    objective, x0, jac=grad_objective, method='L-BFGS-B',
    options={'maxiter': 200, 'disp': True}
)

# 10. Save results
np.save(os.path.join(OUTPUT_DIR, "optimized_coil_dofs.npy"), res.x)
coil_collection.set_dof(res.x)
coil_collection.save(os.path.join(OUTPUT_DIR, "optimized_coils.json"))
bs.save(os.path.join(OUTPUT_DIR, "optimized_biotsavart.json"))
np.save(os.path.join(OUTPUT_DIR, "target_Bext_normal.npy"), Bext)
np.save(os.path.join(OUTPUT_DIR, "optimized_Bnormal.npy"), bs.Bnormal(surface))
```