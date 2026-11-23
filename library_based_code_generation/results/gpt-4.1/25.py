```python
import numpy as np
from simsopt.field.coil import CurveRZFourier, Coil
from simsopt.field.coils import Coils
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfield import SurfaceRZFourier
from simsopt.objectives import (
    SquaredFlux,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveCurvature,
    CurveMeanSquaredCurvature,
    LeastSquaresObjective,
)
from simsopt.solve import least_squares_serial_solve, TaylorTest
import pickle

# Parameters
ncoils = 6
npts_coil = 100
nphi = 64
ntheta = 32
order = 5
coil_length_weight = 1e-2
coil_coil_dist_weight = 1e-2
coil_surface_dist_weight = 1e-2
curvature_weight = 1e-3
mean_squared_curvature_weight = 1e-3
target_Bn_weight = 1.0

# 1. Initialize the boundary magnetic surface
surface = SurfaceRZFourier.from_symmetry(
    nphi=nphi, ntheta=ntheta, mpol=order, ntor=order, stellsym=True
)
# Set surface coefficients (example: simple torus)
surface.set("rc(0,0)", 1.0)
surface.set("zc(1,0)", 0.3)

# 2. Create initial coils
coils = []
for i in range(ncoils):
    angle = 2 * np.pi * i / ncoils
    center = [1.0 * np.cos(angle), 1.0 * np.sin(angle), 0.0]
    curve = CurveRZFourier(npoints=npts_coil, order=order)
    curve.set("rc(0)", 1.0)
    curve.set("zc(1)", 0.3)
    # Rotate coil in toroidal direction
    curve.rotate(angle)
    coil = Coil(curve, current=1.0)
    coils.append(coil)
coils = Coils(coils)

# 3. Define Biot-Savart field
bs = BiotSavart(coils)

# 4. Define objective function terms
# Target normal field: zero everywhere (vacuum field)
target_Bn = np.zeros((nphi * ntheta,))

squared_flux = SquaredFlux(surface, bs, target_Bn, weight=target_Bn_weight)
coil_length = CurveLength(coils, weight=coil_length_weight)
coil_coil_dist = CurveCurveDistance(coils, dmin=0.1, weight=coil_coil_dist_weight)
coil_surface_dist = CurveSurfaceDistance(coils, surface, dmin=0.1, weight=coil_surface_dist_weight)
curvature = CurveCurvature(coils, weight=curvature_weight)
mean_squared_curvature = CurveMeanSquaredCurvature(coils, weight=mean_squared_curvature_weight)

# 5. Form total objective function
objective = LeastSquaresObjective(
    [squared_flux, coil_length, coil_coil_dist, coil_surface_dist, curvature, mean_squared_curvature]
)

# 6. Taylor test
taylor = TaylorTest(objective, coils.x)
taylor.check(order=2, show=True)

# 7. Run optimization
least_squares_serial_solve(objective, coils, verbose=2, maxiter=100)

# 8. Reduce coil length penalty and re-optimize
coil_length.weight = coil_length_weight * 0.1
least_squares_serial_solve(objective, coils, verbose=2, maxiter=100)

# 9. Save optimized coil shapes and currents
with open("optimized_coils.pkl", "wb") as f:
    pickle.dump(coils, f)
currents = [coil.current for coil in coils]
np.save("optimized_currents.npy", currents)
```