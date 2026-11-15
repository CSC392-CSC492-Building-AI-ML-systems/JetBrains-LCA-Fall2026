```python
import numpy as np
from simsopt.geo import SurfaceRZFourier, CurveCWSFourier, curves_to_vtk, curves_via_symmetries
from simsopt.field import BiotSavart, SquaredFlux
from simsopt.objectives import L2CurveLength, MinimumDistance, QuadraticPenalty
from simsopt.optimize import Optimizer
from scipy.optimize import minimize

# Parameters
ncoils = 4
nfilaments = 5  # multifilament grid per coil
nphi_coil = 100
nphi_surface = 64
order = 8
nfp = 2
sym = True

# 1. Initialize boundary magnetic surface
surface = SurfaceRZFourier(nphi_surface, mpol=order, ntor=order, nfp=nfp, stellsym=sym)
surface.set("rc(0,0)", 10.0)
surface.set("zc(1,0)", 2.0)

# 2. Create equally spaced curves (coils)
base_curves = []
for i in range(ncoils):
    curve = CurveCWSFourier(nphi_coil, order, nfp=nfp, stellsym=sym)
    phi = 2 * np.pi * i / ncoils
    curve.set("x0", 10.0 * np.cos(phi))
    curve.set("y0", 10.0 * np.sin(phi))
    curve.set("z0", 0.0)
    curve.set("rc(0)", 1.0)
    curve.set("zs(1)", 0.5)
    base_curves.append(curve)

# 3. Apply stellarator and rotation symmetries
coils = curves_via_symmetries(base_curves, nfp=nfp, stellsym=sym)

# 4. Create multifilament grid for each coil
from simsopt.geo import MultifilamentCoil
multi_coils = []
for coil in coils:
    multi_coil = MultifilamentCoil(coil, nfilaments, spacing=0.02)
    multi_coils.append(multi_coil)

# 5. Define Biot-Savart law for multifilament pack
bs = BiotSavart(multi_coils)

# 6. Define objective function: squared flux + curve length penalty + coil-to-coil distance penalty
flux_obj = SquaredFlux(surface, bs)
length_obj = L2CurveLength(coils)
distance_obj = MinimumDistance(coils, d_min=0.5, p=2)
objective = flux_obj + 1e-3 * length_obj + 1e-2 * distance_obj

# 7. Taylor test for derivative check
from simsopt._core.taylor_test import taylor_test
taylor_test(objective, multi_coils[0].curve, 'coefficients', seed=1, eps=1e-6)

# 8. Run optimization using L-BFGS-B
x0 = []
bounds = []
for coil in coils:
    x0.extend(coil.coefficients)
    bounds.extend([(-2, 2)] * len(coil.coefficients))
x0 = np.array(x0)

def set_coeffs(x):
    idx = 0
    for coil in coils:
        n = len(coil.coefficients)
        coil.coefficients = x[idx:idx+n]
        idx += n

def fun(x):
    set_coeffs(x)
    return objective.J()

def grad(x):
    set_coeffs(x)
    objective.dJ()
    grad = []
    for coil in coils:
        grad.extend(coil.dcoefficients)
    return np.array(grad)

res = minimize(fun, x0, jac=grad, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})

# 9. Save output in VTK format
curves_to_vtk(coils, "optimized_coils.vtk")
surface.to_vtk("optimized_surface.vtk")
```