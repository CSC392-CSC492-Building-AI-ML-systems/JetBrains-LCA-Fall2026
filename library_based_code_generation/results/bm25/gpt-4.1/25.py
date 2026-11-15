```python
import numpy as np
from simsopt.field.coil import Coil, curves_to_coils
from simsopt.field.curve import CurveCWSFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfield import MagneticField
from simsopt.field.surface import SurfaceRZFourier
from simsopt.objectives import (
    SquaredFlux,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    Curvature,
    MeanSquaredCurvature,
    LinearObjective,
)
from simsopt.solve import least_squares_serial_solve
from simsopt.util import TaylorTest

# 1. Initialize the boundary magnetic surface
nphi = 64
ntheta = 32
R0 = 10.0
r0 = 1.0
surface = SurfaceRZFourier(nphi, ntheta, mpol=3, ntor=3)
surface.set("rc(0,0)", R0)
surface.set("rc(1,0)", r0)
surface.set("zc(1,0)", 0.3 * r0)

# 2. Create initial coils
ncoils = 6
order = 8
coils = []
for i in range(ncoils):
    center = [R0 + 1.5 * r0 * np.cos(2 * np.pi * i / ncoils), 1.5 * r0 * np.sin(2 * np.pi * i / ncoils), 0.0]
    curve = CurveCWSFourier(npoints=100, order=order)
    curve.set("x0", center[0])
    curve.set("y0", center[1])
    curve.set("z0", center[2])
    coil = Coil(curve, current=1e5)
    coils.append(coil)

# 3. Define the Biot-Savart field
bs = BiotSavart(coils)

# 4. Define the target normal field (zero normal field)
target_Bn = np.zeros((surface.nphi, surface.ntheta))

# 5. Define objective function terms
# Magnetic field normal component
J_Bn = SquaredFlux(surface, bs, target_Bn)

# Coil length penalty
length_penalty = 1e-3
J_length = LinearObjective([CurveLength(coil.curve) for coil in coils], length_penalty)

# Coil-to-coil distance penalty
distance_penalty = 1e-2
J_coilcoil = CurveCurveDistance([coil.curve for coil in coils], min_distance=0.3*r0, weight=distance_penalty)

# Coil-to-surface distance penalty
surf_distance_penalty = 1e-2
J_coilsurf = CurveSurfaceDistance([coil.curve for coil in coils], surface, min_distance=0.2*r0, weight=surf_distance_penalty)

# Curvature penalty
curvature_penalty = 1e-4
J_curvature = LinearObjective([Curvature(coil.curve) for coil in coils], curvature_penalty)

# Mean squared curvature penalty
mscurv_penalty = 1e-4
J_mscurv = LinearObjective([MeanSquaredCurvature(coil.curve) for coil in coils], mscurv_penalty)

# 6. Form the total objective function
objective = J_Bn + J_length + J_coilcoil + J_coilsurf + J_curvature + J_mscurv

# 7. Taylor test
taylor_test = TaylorTest(objective)
taylor_test.check()

# 8. Run the optimization
variables = []
for coil in coils:
    variables += coil.curve.x
    variables.append(coil.current)
least_squares_serial_solve(objective, coils, tol=1e-6, maxiter=100)

# 9. Reduce coil length penalty and re-optimize
J_length.weight = 1e-4
objective2 = J_Bn + J_length + J_coilcoil + J_coilsurf + J_curvature + J_mscurv
least_squares_serial_solve(objective2, coils, tol=1e-6, maxiter=100)

# 10. Save optimized coil shapes and currents
for i, coil in enumerate(coils):
    coil.save(f"optimized_coil_{i:02d}.json")
    np.savetxt(f"optimized_coil_{i:02d}_current.txt", [coil.current])
```