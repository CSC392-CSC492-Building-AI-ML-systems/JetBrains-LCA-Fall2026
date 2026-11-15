```python
import numpy as np
from simsopt.field.coil import Coil, coils_via_symmetries, BiotSavart
from simsopt.field.coil_collection import CoilCollection
from simsopt.field.magneticfield import MagneticField
from simsopt.field.surface import SurfaceRZFourier
from simsopt.objectives import (
    SquaredFlux,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveCurvature,
    CurveMeanSquaredCurvature,
    LinearObjective,
)
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import TaylorTest
import pickle

# 1. Initialize the boundary magnetic surface
nphi = 64
ntheta = 32
R0 = 10.0
r_minor = 2.0
surface = SurfaceRZFourier(nphi, ntheta, mpol=3, ntor=3)
surface.set("rc(0,0)", R0)
surface.set("rc(1,0)", r_minor)
surface.set("zc(0,0)", 0.0)
surface.set("zc(0,1)", 0.3 * r_minor)

# 2. Create initial coils
ncoils = 6
npts_coil = 100
coils = []
for i in range(ncoils):
    phi = 2 * np.pi * i / ncoils
    center = np.array([R0 + 1.5 * r_minor * np.cos(phi), 1.5 * r_minor * np.sin(phi), 0.0])
    coil = Coil.circular(npts_coil, center, r_minor * 0.7, axis='z')
    coil.current = 1e5
    coils.append(coil)
coils = coils_via_symmetries(coils, ncoils, True)
coil_collection = CoilCollection(coils)
bs = BiotSavart(coil_collection)

# 3. Define the target normal field on the surface
target_Bn = np.zeros(surface.gamma().shape[1])

# 4. Define individual terms of the objective function
flux_term = SquaredFlux(surface, bs, target_Bn, normalization=1.0)
length_term = CurveLength(coil_collection, normalization=1.0)
coilcoil_term = CurveCurveDistance(coil_collection, dmin=0.3, normalization=1.0)
coilsurf_term = CurveSurfaceDistance(coil_collection, surface, dmin=0.2, normalization=1.0)
curvature_term = CurveCurvature(coil_collection, normalization=1.0)
meansq_curvature_term = CurveMeanSquaredCurvature(coil_collection, normalization=1.0)

# 5. Form the total objective function
weights = {
    "flux": 1.0,
    "length": 0.01,
    "coilcoil": 0.1,
    "coilsurf": 0.1,
    "curvature": 0.01,
    "meansq_curvature": 0.01,
}
objective = LinearObjective(
    [
        flux_term,
        length_term,
        coilcoil_term,
        coilsurf_term,
        curvature_term,
        meansq_curvature_term,
    ],
    [weights["flux"], weights["length"], weights["coilcoil"], weights["coilsurf"], weights["curvature"], weights["meansq_curvature"]],
)

# 6. Perform a Taylor test
taylor_test = TaylorTest(objective, coil_collection.x)
taylor_test.check()

# 7. Run the optimization
least_squares_mpi_solve(objective, coil_collection, tol=1e-4, max_nfev=100)

# 8. Use result as initial guess for subsequent optimization with reduced coil length penalty
weights["length"] *= 0.1
objective2 = LinearObjective(
    [
        flux_term,
        length_term,
        coilcoil_term,
        coilsurf_term,
        curvature_term,
        meansq_curvature_term,
    ],
    [weights["flux"], weights["length"], weights["coilcoil"], weights["coilsurf"], weights["curvature"], weights["meansq_curvature"]],
)
least_squares_mpi_solve(objective2, coil_collection, tol=1e-4, max_nfev=100)

# 9. Save the optimized coil shapes and currents
with open("optimized_coils.pkl", "wb") as f:
    pickle.dump(coil_collection, f)
currents = [coil.current for coil in coil_collection.coils]
np.save("optimized_currents.npy", currents)
```