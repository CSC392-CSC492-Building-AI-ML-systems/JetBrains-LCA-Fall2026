```python
import numpy as np
import scipy
import os
import sys

import porepy as pp
from porepy.grids.tensor_grid import TensorGrid
from porepy.utils.grid_bucket import map_subgrid_to_grid
from porepy.numerics.fv import DualVEMMixDim
from porepy.numerics.fv import mass_matrix as pp_mass_matrix
from porepy.numerics.fv import advection as pp_advection
from porepy.io.grid_writer import GmshGridBucketWriter

# Append path and import soultz_grid
sys.path.append(os.path.join(os.path.dirname(__file__), "soultz"))
import soultz_grid

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = "dir"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), bc_type=bc_type)
        bc_values = np.zeros(g.num_faces)
        d.update({
            "permeability": k,
            "source": source,
            "aperture": aperture,
            "bc": bc,
            "bc_values": bc_values,
        })
    for e, d in gb.edges():
        d["aperture"] = 1.0

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.1 * np.ones(g.num_cells)
        discharge = np.zeros((g.dim, g.num_cells))
        bc_type = "dir"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), bc_type=bc_type)
        bc_values = np.zeros(g.num_faces)
        d.update({
            "source": source,
            "porosity": porosity,
            "discharge": discharge,
            "bc": bc,
            "bc_values": bc_values,
        })

# Parameters for grid creation
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
mesh_size = 0.1
tol = 1e-6
coarsen = False

# Create grid using soultz_grid
gb = soultz_grid.create_grid(domain, mesh_size)
gb.compute_geometry()

if coarsen:
    gb = pp.coarsen(gb, refinement_level=1)
gb.assign_node_ordering()

# Darcy problem
darcy_solver = DualVEMMixDim(gb, solver_type="direct")
add_data_darcy(gb, domain, tol)
A, b = darcy_solver.matrix_rhs()
solution = pp.linalg.spsolve(A, b)
darcy_solver.split(solution)
discharge = {}
pressure = {}
for g, d in gb:
    pressure[g] = d["pressure"]
    discharge[g] = d["discharge"]
projected_discharge = {}
for g, d in gb:
    projected_discharge[g] = pp.project.discharge(g, d["discharge"])
total_flow_rate = sum(np.sum(d["discharge"]) for g, d in gb)

# Transport problem
add_data_advection(gb, domain, tol)
adv_solver = pp_advection.Upwind(gb)
mass_solver = pp_mass_matrix.MassMatrix(gb)
M, _ = mass_solver.matrix_rhs()
A_adv, _ = adv_solver.matrix_rhs()
from scipy.sparse.linalg import splu
lu = splu(M + A_adv)
c = {}
for g, d in gb:
    c[g] = np.zeros(g.num_cells)

# Time-stepping
dt = 1e-2
T = 1.0
num_steps = int(T / dt)
export_steps = 10
export_dir = "results"
os.makedirs(export_dir, exist_ok=True)
production_data = []

for step in range(num_steps):
    rhs = {}
    for g, d in gb:
        rhs[g] = M @ c[g] + dt * d["source"]
    for g, d in gb:
        c[g] = lu.solve(rhs[g])
    if step % export_steps == 0:
        for g, d in gb:
            pp.io._export_vtk_grid(g, c[g], os.path.join(export_dir, f"transport_{step:04d}.vtu"))
    production_data.append([step * dt, sum(np.sum(c[g]) for g, d in gb)])

# Export solution in PVD format
writer = GmshGridBucketWriter(gb, os.path.join(export_dir, "final_solution"))
writer.write()

# Save production data
np.savetxt(os.path.join(export_dir, "production_data.txt"), np.array(production_data))
```