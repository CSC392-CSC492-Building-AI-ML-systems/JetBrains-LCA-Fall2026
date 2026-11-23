```python
import os
import numpy as np
import porepy as pp
from porepy.io import GridBucketReader
from porepy.utils.exporter import export_vtk, export_pvd

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = 'dir'
        bc_faces = g.get_boundary_faces()
        bc_val = np.zeros(g.num_faces)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        d['permeability'] = k
        d['source'] = source
        d['aperture'] = aperture
        d['bc'] = bc
        d['bc_values'] = bc_val
    for e, d in gb.edges():
        d['aperture'] = np.ones(e.num_cells)
        d['permeability'] = np.ones(e.num_cells)
        d['normal_diffusivity'] = np.ones(e.num_cells)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.2 * np.ones(g.num_cells)
        discharge = np.zeros((g.num_cells, g.dim))
        bc_type = 'dir'
        bc_faces = g.get_boundary_faces()
        bc_val = np.zeros(g.num_faces)
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        d['source'] = source
        d['porosity'] = porosity
        d['discharge'] = discharge
        d['bc'] = bc
        d['bc_values'] = bc_val
    for e, d in gb.edges():
        d['discharge'] = np.zeros((e.num_cells, e.dim))

tol = 1e-5
export_folder = "results"
time = 10.0
num_time_steps = 100
dt = time / num_time_steps
export_frequency = 10
coarsening = False

mesh_size = {'fracture': 0.1, 'matrix': 0.2}
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

if not os.path.exists(export_folder):
    os.makedirs(export_folder)

csv_file = "grid.csv"
gb = GridBucketReader(csv_file).read()
for g, d in gb:
    g.compute_geometry()
if coarsening:
    gb = pp.coarsen(gb, tol)
for g, d in gb:
    g.assign_node_ordering()

darcy_solver = pp.DualVEMMixDim(gb, solver_type="direct")
add_data_darcy(gb, domain, tol)
darcy_solver.discretize()
darcy_solver.assemble_matrix_rhs()
darcy_solver.solve()
for g, d in gb:
    d['darcy_flux'] = darcy_solver.extract_discharge(g)
    d['pressure'] = darcy_solver.extract_pressure(g)
    d['projected_discharge'] = pp.project.discharge(g, d['darcy_flux'])
total_flow_rate = 0.0
for g, d in gb:
    bc_faces = g.get_boundary_faces()
    total_flow_rate += np.sum(d['darcy_flux'][bc_faces])
export_vtk.export_vtk(gb, os.path.join(export_folder, "darcy_results"), {'pressure': 'cell', 'darcy_flux': 'cell'})

physics = {'advection': True, 'mass_matrix': True}
advection_solver = pp.Upwind(gb)
mass_matrix_solver = pp.MassMatrix(gb)
add_data_advection(gb, domain, tol)
for g, d in gb:
    d['time_step'] = dt
advection_solver.discretize()
mass_matrix_solver.discretize()
A_adv, b_adv = advection_solver.assemble_matrix_rhs()
A_mass, b_mass = mass_matrix_solver.assemble_matrix_rhs()
from scipy.sparse.linalg import splu
lu_mass = splu(A_mass.tocsc())

solution = np.zeros(A_mass.shape[0])
times = np.zeros(num_time_steps + 1)
production = np.zeros(num_time_steps + 1)

for step in range(num_time_steps):
    t = (step + 1) * dt
    rhs = b_adv + A_adv @ solution
    solution = lu_mass.solve(rhs)
    times[step + 1] = t
    production[step + 1] = np.sum(solution)
    if step % export_frequency == 0 or step == num_time_steps - 1:
        for g, d in gb:
            d['concentration'] = solution[:g.num_cells]
        export_vtk.export_vtk(gb, os.path.join(export_folder, f"advection_{step:04d}"), {'concentration': 'cell'})
export_pvd(export_folder, "advection", num_time_steps, export_frequency)
np.savetxt(os.path.join(export_folder, "times.txt"), times)
np.savetxt(os.path.join(export_folder, "production.txt"), np.abs(production))
```