```python
import os
import numpy as np
import porepy as pp
from porepy.io import dfm_3d_from_csv
from porepy.utils.exporter import _export_vtk_gb
from porepy.grids.grid_bucket import assign_node_ordering
from porepy.coarsening.coarsening import generate_coarse_grid_gb
from porepy.numerics.fv import fv
from porepy.numerics.advection import Advection
from porepy.numerics.mass_matrix import MassMatrix
from porepy.numerics.darcy import Darcy
from scipy.sparse.linalg import splu

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_values = np.zeros(g.num_faces)
        # Example: Dirichlet on left, Neumann elsewhere
        left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
        bc_type[left] = pp.BC.dirichlet
        bc_values[left] = 1.0
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        bc_type[right] = pp.BC.dirichlet
        bc_values[right] = 0.0
        d['permeability'] = k
        d['source'] = source
        d['aperture'] = aperture
        d['bc'] = pp.BC(g, bc_type, bc_values)
    for e, d in gb.edges():
        d['aperture'] = np.ones(e.num_cells)
        d['permeability'] = np.ones(e.num_cells)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.2 * np.ones(g.num_cells)
        discharge = np.zeros((g.dim, g.num_cells))
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_values = np.zeros(g.num_faces)
        left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
        bc_type[left] = pp.BC.dirichlet
        bc_values[left] = 1.0
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        bc_type[right] = pp.BC.dirichlet
        bc_values[right] = 0.0
        d['source'] = source
        d['porosity'] = porosity
        d['discharge'] = discharge
        d['bc'] = pp.BC(g, bc_type, bc_values)
    for e, d in gb.edges():
        d['discharge'] = np.zeros((e.dim, e.num_cells))

tol = 1e-6
export_folder = "results"
time = 0.0
num_time_steps = 100
dt = 1e-2
export_frequency = 10
coarsening = False

mesh_size = {'fracture': 0.1, 'matrix': 0.2}
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 1}

if not os.path.exists(export_folder):
    os.makedirs(export_folder)

csv_file = "grid.csv"
gb = dfm_3d_from_csv(csv_file, mesh_size=mesh_size)
gb.compute_geometry()
if coarsening:
    gb = generate_coarse_grid_gb(gb)
assign_node_ordering(gb)

darcy_solver = Darcy(gb)
add_data_darcy(gb, domain, tol)
darcy_solver.discretize()
A, b = darcy_solver.assemble_matrix_rhs()
solution = pp.solve(A, b)
darcy_solver.extract_and_project(solution)
darcy_solver.extract_pressure(solution)
darcy_solver.extract_discharge(solution)

total_flow = 0.0
for g, d in gb:
    bc_faces = np.where(d['bc'].is_neumann)[0]
    if bc_faces.size > 0:
        total_flow += np.sum(d['bc'].values[bc_faces])
_export_vtk_gb(gb, os.path.join(export_folder, "darcy_result"))

physics = {'advection': True, 'darcy': True}
advection_solver = Advection(gb)
mass_matrix_solver = MassMatrix(gb)
add_data_advection(gb, domain, tol)
gb.add_node_prop('time_step')
for g, d in gb:
    d['time_step'] = dt

advection_solver.discretize()
mass_matrix_solver.discretize()
A_adv, b_adv = advection_solver.assemble_matrix_rhs()
M, _ = mass_matrix_solver.assemble_matrix_rhs()
lu = splu(M + dt * A_adv)

solution_vec = np.zeros(M.shape[0])
times = np.zeros(num_time_steps + 1)
production = np.zeros(num_time_steps + 1)

for step in range(1, num_time_steps + 1):
    t = step * dt
    rhs = M @ solution_vec + dt * b_adv
    solution_vec = lu.solve(rhs)
    times[step] = t
    production[step] = np.sum(solution_vec)
    if step % export_frequency == 0:
        for g, d in gb:
            d['advection_solution'] = solution_vec[:g.num_cells]
            solution_vec = solution_vec[g.num_cells:]
        _export_vtk_gb(gb, os.path.join(export_folder, f"advection_{step:04d}"))

pvd_file = os.path.join(export_folder, "advection.pvd")
with open(pvd_file, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
    for step in range(0, num_time_steps + 1, export_frequency):
        f.write(f'<DataSet timestep="{times[step]}" group="" part="0" file="advection_{step:04d}.vtu"/>\n')
    f.write('</Collection>\n</VTKFile>\n')

np.savetxt(os.path.join(export_folder, "production.txt"), np.vstack((times, np.abs(production))).T)
```