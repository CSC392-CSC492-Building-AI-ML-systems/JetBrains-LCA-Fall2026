```python
import numpy as np
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, curves_to_vtk, curves_to_csv, curves_to_json
from simsopt.geo import curves_to_wout, curves_to_makegrid
from simsopt.geo import CurveCWSFourier, create_equally_spaced_curves
from simsopt.objectives import (
    SquaredFlux,
    CoilLength,
    CoilCoilDistance,
    CoilSurfaceDistance,
    CurveCurvature,
    CurveMeanSquaredCurvature,
    LinearObjective,
)
from simsopt.solve import least_squares_serial_solve
from simsopt.util import TaylorTest

# Parameters
ncoils = 6
nfp = 2
order = 8
nphi = 64
ntheta = 32
surface_filename = "surface_input.txt"  # Replace with your surface file if needed

# 1. Initialize the boundary magnetic surface
surface = SurfaceRZFourier(nphi, ntheta, order, nfp=nfp)
# Optionally, load from file:
# surface.read(surface_filename)

# 2. Create initial coils
coils = create_equally_spaced_curves(
    CurveCWSFourier, ncoils, order=order, stellsym=True, nfp=nfp, R0=1.2, Z0=0.0, A=0.15
)
currents = [1e5] * ncoils  # Initial currents in Amperes

# 3. Biot-Savart field from coils
bs = BiotSavart(coils, currents)

# 4. Define objective function terms
# Target normal field: zero everywhere (vacuum field)
target_normal_field = np.zeros((nphi, ntheta))

# Magnetic field normal component error
flux_obj = SquaredFlux(surface, bs, target=target_normal_field)

# Coil length penalty
length_obj = CoilLength(coils)

# Coil-to-coil distance penalty
coilcoil_obj = CoilCoilDistance(coils, min_distance=0.12)

# Coil-to-surface distance penalty
coilsurf_obj = CoilSurfaceDistance(coils, surface, min_distance=0.12)

# Curvature penalty
curvature_obj = CurveCurvature(coils)

# Mean squared curvature penalty
meansqcurv_obj = CurveMeanSquaredCurvature(coils)

# 5. Form total objective function
w_flux = 1.0
w_length = 1e-3
w_coilcoil = 1e-2
w_coilsurf = 1e-2
w_curvature = 1e-3
w_meansqcurv = 1e-3

objective = LinearObjective(
    [
        flux_obj,
        length_obj,
        coilcoil_obj,
        coilsurf_obj,
        curvature_obj,
        meansqcurv_obj,
    ],
    [w_flux, w_length, w_coilcoil, w_coilsurf, w_curvature, w_meansqcurv],
)

# 6. Taylor test
taylor_test = TaylorTest(objective, coils[0].x)
taylor_test.check()

# 7. Run optimization
variables = []
for coil in coils:
    variables.append(coil.x)
variables = np.concatenate(variables)

def set_variables(x):
    idx = 0
    for coil in coils:
        n = coil.x.size
        coil.x = x[idx:idx+n]
        idx += n

def get_variables():
    return np.concatenate([coil.x for coil in coils])

def fun(x):
    set_variables(x)
    return objective.J()

def grad(x):
    set_variables(x)
    return objective.dJ()

result = least_squares_serial_solve(
    fun,
    grad,
    get_variables(),
    verbose=2,
    maxiter=100,
    tol=1e-6,
)

set_variables(result.x)

# 8. Reduce coil length penalty and re-optimize
w_length2 = 1e-4
objective2 = LinearObjective(
    [
        flux_obj,
        length_obj,
        coilcoil_obj,
        coilsurf_obj,
        curvature_obj,
        meansqcurv_obj,
    ],
    [w_flux, w_length2, w_coilcoil, w_coilsurf, w_curvature, w_meansqcurv],
)

def fun2(x):
    set_variables(x)
    return objective2.J()

def grad2(x):
    set_variables(x)
    return objective2.dJ()

result2 = least_squares_serial_solve(
    fun2,
    grad2,
    get_variables(),
    verbose=2,
    maxiter=100,
    tol=1e-6,
)

set_variables(result2.x)

# 9. Save optimized coil shapes and currents
curves_to_vtk(coils, "optimized_coils.vtk")
curves_to_csv(coils, "optimized_coils.csv")
curves_to_json(coils, "optimized_coils.json")
with open("optimized_currents.txt", "w") as f:
    for I in currents:
        f.write(f"{I}\n")
```