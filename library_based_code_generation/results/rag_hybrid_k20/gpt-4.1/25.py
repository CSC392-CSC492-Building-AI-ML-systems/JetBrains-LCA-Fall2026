```python
import numpy as np
from simsopt.field.coil import CircularCoil, coils_via_symmetries
from simsopt.field.coil import CoilSet
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfield import SurfaceRZFourier
from simsopt.objectives import (
    SquaredFlux,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveCurvature,
    CurveMeanSquaredCurvature,
    LinearObjective
)
from simsopt.solve import least_squares_serial_solve
from simsopt.util.taylor_test import taylor_test

# 1. Initialize the boundary magnetic surface
nphi = 64
ntheta = 32
R0 = 10.0
r0 = 1.0
surface = SurfaceRZFourier(nphi, ntheta, mpol=3, ntor=3, stellsym=True)
surface.set("rc(0,0)", R0)
surface.set("rc(1,0)", r0)
surface.set("zc(0,1)", 0.3)

# 2. Create initial coils
ncoils = 6
coils = []
coil_radius = 1.2
coil_current = 1e6
for i in range(ncoils):
    phi = 2 * np.pi * i / ncoils
    center = [R0 + 2.0 * np.cos(phi), 0.0, 2.0 * np.sin(phi)]
    normal = [0, 1, 0]
    coil = CircularCoil(center=center, normal=normal, radius=coil_radius, current=coil_current)
    coils.append(coil)
coils = coils_via_symmetries(coils, nfp=1, stellsym=True)
coilset = CoilSet(coils)
bs = BiotSavart(coilset)

# 3. Define the individual terms of the objective function
# Target normal field: zero normal field on the surface
target_normal_field = np.zeros(surface.normal().shape[0])

# Magnetic field normal component objective
flux_obj = SquaredFlux(surface, bs, target=target_normal_field, weight=1.0)

# Coil length penalty
length_obj = CurveLength(coilset, weight=1e-3)

# Coil-to-coil distance penalty
coilcoil_obj = CurveCurveDistance(coilset, min_dist=0.3, weight=1e-2)

# Coil-to-surface distance penalty
coilsurf_obj = CurveSurfaceDistance(coilset, surface, min_dist=0.3, weight=1e-2)

# Curvature penalty
curvature_obj = CurveCurvature(coilset, weight=1e-4)

# Mean squared curvature penalty
meansqcurv_obj = CurveMeanSquaredCurvature(coilset, weight=1e-4)

# 4. Form the total objective function
objective = LinearObjective([
    flux_obj,
    length_obj,
    coilcoil_obj,
    coilsurf_obj,
    curvature_obj,
    meansqcurv_obj
])

# 5. Perform a Taylor test
params = coilset.get_dofs()
taylor_test(objective, params)

# 6. Run the optimization
least_squares_serial_solve(objective, coilset, grad=True, tol=1e-4, max_nfev=100)

# 7. Reduce penalty for coil length and re-optimize
length_obj.weight = 1e-4  # Reduce penalty
objective = LinearObjective([
    flux_obj,
    length_obj,
    coilcoil_obj,
    coilsurf_obj,
    curvature_obj,
    meansqcurv_obj
])
least_squares_serial_solve(objective, coilset, grad=True, tol=1e-4, max_nfev=100)

# 8. Save the optimized coil shapes and currents
coilset.save("optimized_coils.json")
np.savez("optimized_coil_currents.npz", currents=[c.current for c in coilset.coils])
```