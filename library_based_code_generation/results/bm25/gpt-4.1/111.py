```python
import numpy as np
from simsopt.field.coil import Coil, BiotSavart
from simsopt.field.coil import create_multifilament_grid
from simsopt.field.coil import curves_to_coils
from simsopt.field.coil import CurveRZFourier
from simsopt.field.coil import CurveXYZFourier
from simsopt.field.coil import apply_stellarator_symmetry, apply_rotation_symmetry
from simsopt.field.coil import coil_optimization
from simsopt.field.coil import curve_length
from simsopt.field.coil import coil_to_coil_distance
from simsopt.field.coil import squared_flux
from simsopt.field.coil import relax_and_split
from simsopt.field.coil import test_curves_and_surface
from simsopt.field.coil import test_biotsavart_gradient_symmetric_and_divergence_free
from simsopt.field.coil import test_independent_of_scaling
from simsopt.field.coil import test_independent_of_quadpoints
from simsopt.field.coil import test_out_of_bounds
from simsopt.field.coil import test_independent_of_resolution
from simsopt.field.coil import A_and_dA_vjp
from simsopt.field.coil import B_and_dB_vjp
from simsopt.field.coil import A_vjp
from simsopt.field.coil import _A_impl
from simsopt.field.coil import OneOf, OneOfTest, OneOfTstComposite, OneOfIntsComposite
from simsopt.field.coil import test_get_and_write_nml
from simsopt.field.coil import subtest_biotsavart_gradient_symmetric_and_divergence_free
from simsopt._core.optimizable import Optimizable
from simsopt.geo.surface import SurfaceRZFourier
from simsopt.geo import Surface
from simsopt.util import proc0_print
from scipy.optimize import minimize

# Parameters
ncoils = 4
nfilaments_major = 3
nfilaments_minor = 2
ncurves = ncoils
npoints_curve = 100
npoints_surface = 1000
order = 5
symmetry = 2  # nfp
rotation_symmetry = 2
coil_radius = 1.2
surface_radius = 1.0
surface_zeta = 0.0
length_penalty = 1e-3
distance_penalty = 1e-2
flux_weight = 1.0

# 1. Initialize boundary magnetic surface
surface = SurfaceRZFourier(nphi=64, ntheta=32, mpol=order, ntor=order)
rc = np.zeros(order+1)
zc = np.zeros(order+1)
rc[0] = surface_radius
surface.set_dofs(np.concatenate([rc, zc]))

# 2. Create equally spaced curves (coils)
curves = []
for i in range(ncurves):
    phi = 2 * np.pi * i / ncurves
    x0 = coil_radius * np.cos(phi)
    y0 = coil_radius * np.sin(phi)
    z0 = 0.0
    curve = CurveXYZFourier(npoints_curve, order)
    dofs = curve.get_dofs()
    dofs[0] = x0
    dofs[1] = y0
    dofs[2] = z0
    curve.set_dofs(dofs)
    curves.append(curve)

# 3. Apply stellarator and rotation symmetries
curves = apply_stellarator_symmetry(curves, symmetry)
curves = apply_rotation_symmetry(curves, rotation_symmetry)

# 4. Create multifilament grid for each coil
coils = []
for curve in curves:
    multifilament_curves = create_multifilament_grid(curve, nfilaments_major, nfilaments_minor, major_radius=0.02, minor_radius=0.01)
    multifilament_coils = curves_to_coils(multifilament_curves, current=1.0)
    coils.extend(multifilament_coils)

# 5. Biot-Savart law
bs = BiotSavart(coils)

# 6. Define objective function
def objective():
    flux = squared_flux(surface, bs)
    length = sum(curve_length(c) for c in curves)
    distance = coil_to_coil_distance(curves)
    return flux_weight * flux + length_penalty * length + distance_penalty * distance

# 7. Collect all optimizable dofs
optimizable = Optimizable()
for curve in curves:
    optimizable.register(curve)

# 8. Taylor test
def taylor_test():
    def fun(x):
        for i, curve in enumerate(curves):
            curve.set_dofs(x[i*curve.dof_count:(i+1)*curve.dof_count])
        return objective()
    x0 = np.concatenate([curve.get_dofs() for curve in curves])
    eps = 1e-6
    grad = np.zeros_like(x0)
    f0 = fun(x0)
    for i in range(len(x0)):
        x1 = x0.copy()
        x1[i] += eps
        f1 = fun(x1)
        grad[i] = (f1 - f0) / eps
    print("Taylor test gradient norm:", np.linalg.norm(grad))

taylor_test()

# 9. Run optimization
def fun(x):
    for i, curve in enumerate(curves):
        curve.set_dofs(x[i*curve.dof_count:(i+1)*curve.dof_count])
    return objective()

x0 = np.concatenate([curve.get_dofs() for curve in curves])
res = minimize(fun, x0, method='L-BFGS-B', options={'maxiter': 100, 'disp': True})

# 10. Save output in VTK format
for i, curve in enumerate(curves):
    curve.save(f"curve_{i}.vtk")
surface.save("surface.vtk")
for i, coil in enumerate(coils):
    coil.save(f"coil_{i}.vtk")
bs.save("biot_savart.vtk")
```