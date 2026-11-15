```python
import numpy as np
from simsopt.field.coil import CurveCoil, coils_via_symmetries, BiotSavart
from simsopt.field.coil import MultifilamentCoil
from simsopt.field.coil import create_multifilament_grid
from simsopt.field.coil import coil_optimization
from simsopt.field.coil import curve_length
from simsopt.field.coil import coil_to_coil_distance
from simsopt.field.coil import squared_flux
from simsopt.field.coil import save_coils_vtk
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surface import SurfaceRZFourier
from simsopt.util.symmetry import StellaratorSymmetry, RotationalSymmetry
from simsopt.optimize import minimize
from simsopt._core.derivative import taylor_test

# Parameters
ncoils = 4
nfilaments_major = 5
nfilaments_minor = 3
npts_per_curve = 100
nphi_surface = 64
ntheta_surface = 32
order = 8
sym = StellaratorSymmetry(nfp=2, stellsym=True)
rotsym = RotationalSymmetry(nfp=2)
coil_radius = 0.15
coil_distance = 1.0
surface_major_radius = 1.0
surface_minor_radius = 0.3

# 1. Initialize boundary magnetic surface
surface = SurfaceRZFourier(nphi_surface, ntheta_surface, order, stellsym=True)
surface.set("rc(0,0)", surface_major_radius)
surface.set("zc(0,0)", 0.0)
surface.set("rc(1,0)", surface_minor_radius)
surface.set("zc(1,0)", 0.0)

# 2. Create equally spaced curves for coils
curves = create_equally_spaced_curves(
    ncoils, npts_per_curve, coil_radius, coil_distance, sym=sym
)

# 3. Create multifilament grid for each coil
multifilament_coils = []
for curve in curves:
    multifilament_grid = create_multifilament_grid(
        curve, nfilaments_major, nfilaments_minor, pack_type='rectangular'
    )
    multifilament_coil = MultifilamentCoil(multifilament_grid)
    multifilament_coils.append(multifilament_coil)

# 4. Apply stellarator and rotation symmetries
coils = coils_via_symmetries(multifilament_coils, [sym, rotsym])

# 5. Define Biot-Savart law
bs = BiotSavart(coils)

# 6. Define objective function
def objective():
    flux = squared_flux(surface, bs)
    length_penalty = sum(curve_length(coil.curve) for coil in coils)
    distance_penalty = coil_to_coil_distance(coils)
    return flux + 1e-3 * length_penalty + 1e-2 * distance_penalty

# 7. Taylor test
x0 = np.concatenate([coil.curve.x for coil in coils])
def fun(x):
    idx = 0
    for coil in coils:
        n = coil.curve.x.size
        coil.curve.x = x[idx:idx+n]
        idx += n
    return objective()
def grad(x):
    idx = 0
    grad = np.zeros_like(x)
    for coil in coils:
        n = coil.curve.x.size
        coil.curve.x = x[idx:idx+n]
        grad[idx:idx+n] = coil.curve.dobjective_dx()
        idx += n
    return grad
taylor_test(fun, grad, x0)

# 8. Run optimization
res = minimize(fun, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 100})

# 9. Save output in VTK format
for i, coil in enumerate(coils):
    save_coils_vtk([coil], f"coil_{i}.vtk")
curves_to_vtk([coil.curve for coil in coils], "coils_curves.vtk")
surface.to_vtk("surface.vtk")
```