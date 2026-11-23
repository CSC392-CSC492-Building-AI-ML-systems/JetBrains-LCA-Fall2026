```python
import numpy as np
from simsopt.geo import SurfaceRZFourier, CurveCWSFourier, curves_to_vtk, create_equally_spaced_curves
from simsopt.field.coil import Coil, BiotSavart
from simsopt.field.multipole import MultifilamentCoil
from simsopt.field import compute_fieldlines
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives import CurveLength, MinimumDistance
from simsopt.util.symmetry import apply_stellarator_symmetry, apply_rotational_symmetry
from simsopt.solve import TaylorTest
from scipy.optimize import minimize

# Parameters
nphi = 64
ntheta = 32
ncoils = 6
nfilaments = 7  # multifilament grid per coil
order = 8
nfp = 2
symmetry = True

# 1. Initialize boundary magnetic surface
surf = SurfaceRZFourier(nphi, ntheta, mpol=5, ntor=5, stellsym=symmetry, nfp=nfp)
surf.set('rc(0,0)', 1.0)
surf.set('zc(1,0)', 0.3)

# 2. Create equally spaced curves (coil centerlines)
curves = create_equally_spaced_curves(ncoils, order, nfp=nfp, stellsym=symmetry, coil_radius=1.2, coil_z=0.0)

# 3. Apply stellarator and rotation symmetries
if symmetry:
    curves = apply_stellarator_symmetry(curves, nfp)
    curves = apply_rotational_symmetry(curves, nfp)

# 4. Create multifilament grid for each coil
coils = []
for curve in curves:
    multifilament = MultifilamentCoil(curve, nfilaments, pack_type='hex', pack_radius=0.03)
    coils.append(multifilament)

# 5. Define Biot-Savart law
bs = BiotSavart(coils)

# 6. Define objective function components
flux_obj = SquaredFlux(surf, bs, nphi=nphi, ntheta=ntheta)
length_penalty = CurveLength(curves, weight=1e-3)
distance_penalty = MinimumDistance(curves, d_min=0.12, weight=1e-2)

def objective(x):
    for c in curves:
        c.x = x[c.offset:c.offset+c.dof]
    return flux_obj.J() + length_penalty.J() + distance_penalty.J()

def grad_objective(x):
    for c in curves:
        c.x = x[c.offset:c.offset+c.dof]
    grad = np.zeros_like(x)
    grad += flux_obj.dJ()
    grad += length_penalty.dJ()
    grad += distance_penalty.dJ()
    return grad

# 7. Flatten variables for optimization
x0 = np.concatenate([c.x for c in curves])
offset = 0
for c in curves:
    c.offset = offset
    offset += c.dof

# 8. Taylor test
taylor = TaylorTest(objective, grad_objective, x0)
taylor.check()

# 9. Run optimization
res = minimize(objective, x0, jac=grad_objective, method='L-BFGS-B', options={'maxiter': 200})

# 10. Update curves with optimized variables
xopt = res.x
for c in curves:
    c.x = xopt[c.offset:c.offset+c.dof]

# 11. Save output in VTK format
curves_to_vtk(curves, filename="optimized_coils.vtk")
surf.to_vtk("boundary_surface.vtk")
for i, coil in enumerate(coils):
    coil.to_vtk(f"multifilament_coil_{i}.vtk")
```