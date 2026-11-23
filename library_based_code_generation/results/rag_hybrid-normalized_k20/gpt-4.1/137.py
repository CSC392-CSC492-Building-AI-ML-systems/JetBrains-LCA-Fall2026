```python
import numpy as np
import scipy
import os
import sys

import porepy as pp
from porepy.grids import TensorGrid
from porepy.utils.grid_bucket import map_subgrid_to_grid
from porepy.numerics.fv import DualVEMMixDim
from porepy.numerics.fv import Advection
from porepy.numerics.fv import MassMatrix
from porepy.numerics.fv import Upwind
from porepy.io import GmshGridBucketWriter

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        aperture = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        # Dirichlet on left, Neumann elsewhere
        left = np.where(np.abs(g.face_centers[0] - domain[0][0]) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain[0][1]) < tol)[0]
        bc_type[left] = pp.BC_TYPE_DIRICHLET
        bc_value[left] = 1.0
        bc_type[right] = pp.BC_TYPE_DIRICHLET
        bc_value[right] = 0.0
        bc = pp.BoundaryCondition(g, faces=np.hstack([left, right]), cond=bc_type)
        d["permeability"] = k
        d["aperture"] = aperture
        d["source"] = source
        d["bc"] = bc
        d["bc_values"] = bc_value
    for e, d in gb.edges():
        d["aperture"] = 1.0

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.1 * np.ones(g.num_cells)
        discharge = d.get("discharge", np.zeros((g.dim, g.num_cells)))
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        left = np.where(np.abs(g.face_centers[0] - domain[0][0]) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain[0][1]) < tol)[0]
        bc_type[left] = pp.BC_TYPE_DIRICHLET
        bc_value[left] = 1.0
        bc_type[right] = pp.BC_TYPE_DIRICHLET
        bc_value[right] = 0.0
        bc = pp.BoundaryCondition(g, faces=np.hstack([left, right]), cond=bc_type)
        d["source"] = source
        d["porosity"] = porosity
        d["discharge"] = discharge
        d["bc"] = bc
        d["bc_values"] = bc_value

sys.path.append(os.path.abspath("soultz_grid_path"))
import soultz_grid

domain = [[0, 1], [0, 1]]
mesh_size = 0.05
tol = 1e-6
coarsen = False

gb = soultz_grid.create_grid(domain=domain, mesh_size=mesh_size)
gb.compute_geometry()
if coarsen:
    gb = pp.coarsen(gb)
pp.fvutils.assign_node_ordering(gb)

# Darcy problem
darcy_solver = DualVEMMixDim(gb, solver_type="direct")
add_data_darcy(gb, domain, tol)
A, b = darcy_solver.matrix_rhs()
solution = scipy.sparse.linalg.spsolve(A, b)
darcy_solver.split(solution)
for g, d in gb:
    d["discharge"] = darcy_solver.extract_discharge(g)
    d["pressure"] = darcy_solver.extract_pressure(g)
    d["projected_discharge"] = pp.fvutils.project_discharge(g, d["discharge"])
total_flow = 0.0
for g, d in gb:
    right = np.where(np.abs(g.face_centers[0] - domain[0][1]) < tol)[0]
    total_flow += np.sum(d["projected_discharge"][right])

# Transport problem
add_data_advection(gb, domain, tol)
advection_solver = Advection(gb)
mass_matrix_solver = MassMatrix(gb)
M, _ = mass_matrix_solver.matrix_rhs()
A_adv, b_adv = advection_solver.matrix_rhs()
lu = scipy.sparse.linalg.splu(M + 1.0 * A_adv)
c = np.zeros(M.shape[0])
dt = 0.01
T = 1.0
num_steps = int(T / dt)
export_times = np.linspace(0, T, 11)
export_idx = 0
production = []
for step in range(num_steps):
    t = step * dt
    rhs = M @ c + b_adv
    c = lu.solve(rhs)
    if export_idx < len(export_times) and t >= export_times[export_idx]:
        for g, d in gb:
            d["concentration"] = c[:g.num_cells]
            c = c[g.num_cells:]
        GmshGridBucketWriter(gb, "output", "transport_{:03d}".format(export_idx)).write()
        export_idx += 1
    # Compute production (outflow concentration)
    for g, d in gb:
        right = np.where(np.abs(g.face_centers[0] - domain[0][1]) < tol)[0]
        prod = np.sum(d["concentration"][right])
        production.append([t, prod])

# Export final solution
for g, d in gb:
    d["concentration"] = c[:g.num_cells]
    c = c[g.num_cells:]
GmshGridBucketWriter(gb, "output", "transport_final").write()

np.savetxt("production.txt", np.array(production))
```