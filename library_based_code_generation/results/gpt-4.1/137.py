```python
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import os
import sys

import porepy as pp
from porepy.numerics.fv import tpfa
from porepy.numerics.fv import upwind
from porepy.numerics.mixed_dim import DualVEMMixDim
from porepy.numerics.fv import mass_matrix

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1e-14 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        # Example: Dirichlet on left, Neumann elsewhere
        left = np.where(np.abs(g.face_centers[0] - domain[0]) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain[1]) < tol)[0]
        bc_type[left] = pp.BC_DIRICHLET
        bc_value[left] = 1.0
        bc_type[right] = pp.BC_DIRICHLET
        bc_value[right] = 0.0
        d['permeability'] = k
        d['source'] = source
        d['aperture'] = aperture
        d['bc'] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc_values'] = bc_value
    for e, d in gb.edges():
        d['aperture'] = np.ones(e.num_cells)
        d['permeability'] = 1e-14 * np.ones(e.num_cells)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.1 * np.ones(g.num_cells)
        discharge = d.get('darcy_flux', np.zeros(g.num_faces))
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        left = np.where(np.abs(g.face_centers[0] - domain[0]) < tol)[0]
        bc_type[left] = pp.BC_DIRICHLET
        bc_value[left] = 1.0
        d['source'] = source
        d['porosity'] = porosity
        d['discharge'] = discharge
        d['bc'] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc_values'] = bc_value
    for e, d in gb.edges():
        d['discharge'] = np.zeros(e.num_faces)
        d['porosity'] = 0.1 * np.ones(e.num_cells)

# Append soultz_grid path and import
soultz_grid_path = os.path.abspath('./soultz_grid')
if soultz_grid_path not in sys.path:
    sys.path.append(soultz_grid_path)
import soultz_grid

# Grid parameters
domain = [0, 1000]
mesh_size = 50
tol = 1e-6
coarsen = False

# Create grid
gb = soultz_grid.create_grid(domain=domain, mesh_size=mesh_size)
gb.compute_geometry()
if coarsen:
    gb = pp.coarsen(gb, refinement=2)
pp.set_node_ordering(gb)

# Darcy problem
add_data_darcy(gb, domain, tol)
solver = DualVEMMixDim(gb)
A, b = solver.matrix_rhs()
solution = spla.spsolve(A, b)
split_sol = solver.split(solution)
for g, d in gb:
    d['pressure'] = split_sol[g]['pressure']
    d['darcy_flux'] = solver.project_discharge(g, d, split_sol[g])
    d['darcy_flux_faces'] = d['darcy_flux']
    # Compute total flow rate at right boundary
    right = np.where(np.abs(g.face_centers[0] - domain[1]) < tol)[0]
    total_flow = np.sum(d['darcy_flux'][right])
    d['total_flow'] = total_flow

# Transport problem
add_data_advection(gb, domain, tol)
advection_solver = upwind.Upwind(gb)
mass_solver = mass_matrix.MassMatrix(gb)

A_adv, _ = advection_solver.matrix_rhs()
M, _ = mass_solver.matrix_rhs()
A = M + 0.0 * A_adv  # For explicit Euler, A = M

lu = spla.splu(A.tocsc())
c = np.zeros(A.shape[0])

# Time stepping
dt = 1.0
T = 100.0
num_steps = int(T // dt)
export_steps = 10
production = []

for step in range(num_steps):
    t = step * dt
    rhs = M @ c + dt * advection_solver.rhs(c)
    c = lu.solve(rhs)
    if step % export_steps == 0 or step == num_steps - 1:
        for g, d in gb:
            d['concentration'] = c[:g.num_cells]
            c = c[g.num_cells:]
        pp.export_pvd(gb, 'transport', time=t)
        # Save production data
        for g, d in gb:
            right = np.where(np.abs(g.face_centers[0] - domain[1]) < tol)[0]
            prod = np.sum(d['concentration'][right] * d['darcy_flux'][right])
            production.append([t, prod])

# Export production data
np.savetxt('production.txt', np.array(production), header='time, production')
```