```python
import numpy as np
from simsopt.field.coil import Coil, coils_via_symmetries, BiotSavart
from simsopt.field.coil import create_curves, create_multifilament_grid
from simsopt.field.magneticfield import SurfaceRZFourier
from simsopt.field.coil import CurveLength, MinimumDistance
from simsopt.field.coil import ToroidalFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.util.vtk import write_curves_to_vtk, write_surface_to_vtk
from simsopt.solve import least_squares_serial_solve
from scipy.optimize import minimize

# Parameters
ncoils = 4
nfilaments = 7  # multifilament grid per coil
npts_per_curve = 100
nphi = 64
ntheta = 32
nfp = 2  # number of field periods
sym = True

# 1. Initialize boundary magnetic surface
surface = SurfaceRZFourier(nphi, ntheta, mpol=3, ntor=3, stellsym=sym, nfp=nfp)
surface.set("rc(0,0)", 10.0)
surface.set("zc(1,0)", 2.0)
surface.fix_all()
write_surface_to_vtk(surface, "surface_init.vtk")

# 2. Create equally spaced curves (coils)
curves = create_curves(ncoils, npts_per_curve, stellsym=sym, nfp=nfp, coil_radius=1.5, coil_major_radius=8.0)
coils = [Coil(curve) for curve in curves]

# 3. Create multifilament grid for each coil
multi_coils = []
for coil in coils:
    multi_coils.append(create_multifilament_grid(coil, nfilaments, spacing=0.05))

# Flatten multifilament coils list
all_filaments = [fil for multi in multi_coils for fil in multi]

# 4. Apply stellarator and rotation symmetries
all_coils = coils_via_symmetries(all_filaments, nfp=nfp, stellsym=sym)

# 5. Define Biot-Savart law
bs = BiotSavart(all_coils)

# 6. Define objective function components
flux_obj = ToroidalFlux(surface, bs)
length_obj = CurveLength(all_coils)
dist_obj = MinimumDistance(all_coils, d_min=0.3, p=2)

# 7. Combine objective function
def objective():
    flux = flux_obj.J()
    length = length_obj.J()
    dist = dist_obj.J()
    return flux**2 + 1e-3 * length + 1e-2 * dist

def objective_grad():
    flux = flux_obj.J()
    grad_flux = flux_obj.dJ()
    grad_length = length_obj.dJ()
    grad_dist = dist_obj.dJ()
    return 2*flux*grad_flux + 1e-3*grad_length + 1e-2*grad_dist

# 8. Taylor test
def taylor_test():
    x0 = np.concatenate([c.get_dofs() for c in all_coils])
    grad = objective_grad()
    delta = np.random.randn(*x0.shape)
    delta /= np.linalg.norm(delta)
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    f0 = objective()
    for eps in epsilons:
        for i, c in enumerate(all_coils):
            dofs = c.get_dofs()
            c.set_dofs(dofs + eps * delta[i*len(dofs):(i+1)*len(dofs)])
        f1 = objective()
        approx = f0 + eps * np.dot(grad, delta)
        print(f"eps={eps:.1e}, f1-f0={f1-f0:.4e}, approx={approx-f0:.4e}, error={(f1-f0)-(approx-f0):.4e}")
        for i, c in enumerate(all_coils):
            dofs = c.get_dofs()
            c.set_dofs(dofs - eps * delta[i*len(dofs):(i+1)*len(dofs)])

taylor_test()

# 9. Optimization using L-BFGS-B
def pack_dofs():
    return np.concatenate([c.get_dofs() for c in all_coils])

def unpack_dofs(x):
    idx = 0
    for c in all_coils:
        n = len(c.get_dofs())
        c.set_dofs(x[idx:idx+n])
        idx += n

def fun(x):
    unpack_dofs(x)
    return objective()

def jac(x):
    unpack_dofs(x)
    return objective_grad()

x0 = pack_dofs()
res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': 100, 'disp': True})
unpack_dofs(res.x)

# 10. Save output in VTK format
write_curves_to_vtk([c.curve for c in all_coils], "optimized_coils.vtk")
write_surface_to_vtk(surface, "optimized_surface.vtk")
```