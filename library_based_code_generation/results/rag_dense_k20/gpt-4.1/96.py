```python
import os
import numpy as np
import porepy as pp
from porepy.io.grid_import import dfm_2d_from_csv
from porepy.utils.exporter import _export_vtk_grid
from porepy.numerics.fv import tpfa
from porepy.numerics.fv import upwind
from porepy.numerics.fv import mass_matrix
from porepy.numerics.fv import assembler
from porepy.numerics.fv import solvers
from porepy.numerics.fv import utils as fv_utils
from porepy.numerics.fv import physics as fv_physics
from porepy.numerics.fv import boundary_conditions as bc
from scipy.sparse.linalg import splu

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = 'dir'
        bc_faces = g.tags['boundary_faces'].copy()
        bc_val = np.zeros(g.num_faces)
        if g.dim == 2:
            left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
            right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
            bc_faces[left] = 1
            bc_faces[right] = 1
            bc_val[left] = 1.0
            bc_val[right] = 0.0
        elif g.dim == 1:
            bc_faces[:] = 1
            bc_val[:] = 0.0
        bc_dict = {'type': bc.BoundaryCondition(g, faces=bc_faces, cond=bc_type), 'value': bc_val}
        d['permeability'] = k
        d['source'] = source
        d['aperture'] = aperture
        d['bc'] = bc_dict

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.2 * np.ones(g.num_cells)
        discharge = d.get('darcy_flux', np.zeros(g.num_faces))
        bc_type = 'dir'
        bc_faces = g.tags['boundary_faces'].copy()
        bc_val = np.zeros(g.num_faces)
        if g.dim == 2:
            left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
            right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
            bc_faces[left] = 1
            bc_faces[right] = 1
            bc_val[left] = 1.0
            bc_val[right] = 0.0
        elif g.dim == 1:
            bc_faces[:] = 1
            bc_val[:] = 0.0
        bc_dict = {'type': bc.BoundaryCondition(g, faces=bc_faces, cond=bc_type), 'value': bc_val}
        d['source_adv'] = source
        d['porosity'] = porosity
        d['darcy_flux'] = discharge
        d['bc_adv'] = bc_dict

tol = 1e-6
export_folder = "results"
time = 1.0
num_time_steps = 20
dt = time / num_time_steps
export_frequency = 5
coarsen = False

mesh_size = {'fracture': 0.1, 'matrix': 0.2}
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

csv_file = "grid.csv"
gb = dfm_2d_from_csv(csv_file, mesh_size, domain, tol)
gb.compute_geometry()
if coarsen:
    pp.coarsening.coarsen(gb, coarsening_level=1)
for g, d in gb:
    d['node_ordering'] = np.arange(g.num_cells)

darcy_solver = tpfa.Tpfa(gb)
add_data_darcy(gb, domain, tol)
darcy_solver.assemble_matrix_rhs()
darcy_solver.solve()
for g, d in gb:
    d['pressure'] = darcy_solver.extract_pressure(g)
    d['darcy_flux'] = darcy_solver.extract_flux(g)
    d['projected_darcy_flux'] = fv_utils.project_flux(g, d['darcy_flux'])

total_flow = 0.0
for g, d in gb:
    if g.dim == 2:
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        total_flow += np.sum(d['darcy_flux'][right] * g.face_areas[right])
_export_vtk_grid(gb, os.path.join(export_folder, "darcy_results"))

physics = fv_physics.Physics(gb)
advection_solver = upwind.Upwind(gb)
mass_solver = mass_matrix.MassMatrix(gb)
add_data_advection(gb, domain, tol)
for g, d in gb:
    d['dt'] = dt

advection_solver.assemble_matrix_rhs()
mass_solver.assemble_matrix_rhs()
A_adv = advection_solver.A
A_mass = mass_solver.A
A_sys = A_mass + dt * A_adv
lu = splu(A_sys.tocsc())

solution = np.zeros(A_sys.shape[0])
times = np.zeros(num_time_steps + 1)
production = np.zeros(num_time_steps + 1)

for step in range(num_time_steps + 1):
    t = step * dt
    times[step] = t
    rhs = mass_solver.b + dt * advection_solver.b
    solution = lu.solve(rhs)
    for g, d in gb:
        cell_slice = advection_solver.cell_dof(g)
        d['concentration'] = solution[cell_slice]
    if step % export_frequency == 0:
        _export_vtk_grid(gb, os.path.join(export_folder, f"advection_{step:03d}"))
    prod = 0.0
    for g, d in gb:
        if g.dim == 2:
            right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
            prod += np.sum(d['concentration'][right] * d['darcy_flux'][right] * g.face_areas[right])
    production[step] = np.abs(prod)

pvd_file = os.path.join(export_folder, "advection.pvd")
with open(pvd_file, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
    for step in range(0, num_time_steps + 1, export_frequency):
        f.write(f'<DataSet timestep="{times[step]}" group="" part="0" file="advection_{step:03d}.vtu"/>\n')
    f.write('</Collection>\n</VTKFile>\n')

np.savetxt(os.path.join(export_folder, "production.txt"), np.vstack((times, production)).T, header="time production")
```