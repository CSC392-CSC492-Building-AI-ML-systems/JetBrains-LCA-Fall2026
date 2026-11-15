```python
import numpy as np
import scipy.sparse as sps
import os
import sys

import porepy as pp
from porepy.grids.grid_bucket import GridBucket
from porepy.numerics.fv import tpfa
from porepy.numerics.mixed_dim import DualVEMMixDim
from porepy.numerics.advection import Upwind
from porepy.numerics.advection import MassMatrix
from porepy.io.grid_writer import GmshGridBucketWriter

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        d["permeability"] = pp.SecondOrderTensor(np.ones(g.num_cells))
        d["source"] = np.zeros(g.num_cells)
        d["aperture"] = np.ones(g.num_cells)
        bc = pp.BoundaryCondition(g, faces=np.array([], dtype=int), cond=np.array([], dtype=int))
        d["bc"] = bc
        d["bc_values"] = np.zeros(g.num_faces)
    for e, d in gb.edges():
        d["aperture"] = np.ones(d["mortar_grid"].num_cells)
        d["permeability"] = pp.SecondOrderTensor(np.ones(d["mortar_grid"].num_cells))
        d["normal_diffusivity"] = np.ones(d["mortar_grid"].num_cells)
        d["bc"] = pp.BoundaryCondition(d["mortar_grid"], faces=np.array([], dtype=int), cond=np.array([], dtype=int))
        d["bc_values"] = np.zeros(d["mortar_grid"].num_faces)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        d["source"] = np.zeros(g.num_cells)
        d["porosity"] = np.ones(g.num_cells)
        d["discharge"] = np.zeros((g.num_cells, g.dim))
        bc = pp.BoundaryCondition(g, faces=np.array([], dtype=int), cond=np.array([], dtype=int))
        d["bc"] = bc
        d["bc_values"] = np.zeros(g.num_faces)
    for e, d in gb.edges():
        d["porosity"] = np.ones(d["mortar_grid"].num_cells)
        d["discharge"] = np.zeros((d["mortar_grid"].num_cells, d["mortar_grid"].dim))
        d["bc"] = pp.BoundaryCondition(d["mortar_grid"], faces=np.array([], dtype=int), cond=np.array([], dtype=int))
        d["bc_values"] = np.zeros(d["mortar_grid"].num_faces)

# Append path and import soultz_grid
sys.path.append(os.path.abspath("."))
import soultz_grid

# Set up parameters for grid creation
domain = {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000, "zmin": 0, "zmax": 100}
mesh_size = {"fracture": 10, "matrix": 50}
tol = 1e-4
coarsen = False

# Create grid using soultz_grid module
gb = soultz_grid.create_grid(domain, mesh_size)
gb.compute_geometry()

if coarsen:
    gb = pp.coarsen(gb, refinement_level=1)
gb.assign_node_ordering()

# Darcy problem
darcy_solver = DualVEMMixDim(gb, solver_type="direct")
add_data_darcy(gb, domain, tol)
A, b = darcy_solver.matrix_rhs()
solution = sps.linalg.spsolve(A, b)
darcy_solver.split(solution)
discharge = {}
pressure = {}
for g, d in gb:
    pressure[g] = d["pressure"]
    discharge[g] = d["darcy_flux"]
for e, d in gb.edges():
    discharge[e] = d["mortar_flux"]
projected_discharge = {}
for g, d in gb:
    projected_discharge[g] = tpfa.project_discharge(g, d, discharge[g])
total_flow_rate = sum(np.sum(discharge[g]) for g, d in gb)

# Transport problem
add_data_advection(gb, domain, tol)
for g, d in gb:
    d["discharge"] = projected_discharge[g]
advection_solver = Upwind(gb)
mass_matrix_solver = MassMatrix(gb)
M, _ = mass_matrix_solver.matrix_rhs()
A_adv, b_adv = advection_solver.matrix_rhs()
lu = sps.linalg.splu(M + A_adv)
c = {}
for g, d in gb:
    c[g] = np.zeros(g.num_cells)
time_steps = 100
dt = 1.0
export_interval = 10
for step in range(time_steps):
    rhs = {}
    for g, d in gb:
        rhs[g] = M @ c[g] + b_adv
    for g, d in gb:
        c[g] = lu.solve(rhs[g])
    if step % export_interval == 0:
        for g, d in gb:
            d["concentration"] = c[g]
        GmshGridBucketWriter(gb, f"transport_step_{step}.pvd").write()
# Export final solution
for g, d in gb:
    d["concentration"] = c[g]
GmshGridBucketWriter(gb, "final_transport_solution.pvd").write()

# Save production data
production_data = np.array([total_flow_rate])
np.savetxt("production_data.txt", production_data)
```