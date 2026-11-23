```python
import numpy as np
import porepy as pp
from porepy.models.darcy_and_transport import DarcyAndTransport
from porepy.io import to_vtk
from porepy.utils.exporter import export_pvd
from scipy.sparse.linalg import splu

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        perm = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = 1.0 if g.dim == gb.dim_max() else 1e-2
        bc_type = 'dir' * np.ones(g.num_faces, dtype='<U3')
        bc_value = np.zeros(g.num_faces)
        # Example: left boundary Dirichlet, right boundary Neumann
        left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        bc_type[left] = 'dir'
        bc_value[left] = 1.0
        bc_type[right] = 'neu'
        bc_value[right] = 0.0
        d['permeability'] = pp.SecondOrderTensor(perm)
        d['source'] = source
        d['aperture'] = aperture
        d['bc'] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc_values'] = bc_value
    for e, ed in gb.edges():
        ed['aperture'] = 1e-2

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.2 * np.ones(g.num_cells)
        discharge = d.get('darcy_flux', np.zeros(g.num_faces))
        bc_type = 'dir' * np.ones(g.num_faces, dtype='<U3')
        bc_value = np.zeros(g.num_faces)
        left = np.where(np.abs(g.face_centers[0] - domain['xmin']) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        bc_type[left] = 'dir'
        bc_value[left] = 1.0
        bc_type[right] = 'neu'
        bc_value[right] = 0.0
        d['source'] = source
        d['porosity'] = porosity
        d['discharge'] = discharge
        d['bc'] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc_values'] = bc_value

tol = 1e-6
export_folder = "results"
time = 1.0
num_time_steps = 50
dt = time / num_time_steps
export_frequency = 10
coarsening = False

mesh_size = {'fracture': 0.1, 'matrix': 0.2}
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

csv_file = "grid.csv"
gb = pp.io.grid_bucket_from_csv(csv_file)
gb.compute_geometry()
if coarsening:
    gb = pp.coarsen.coarsen(gb, coarsening_level=1)
pp.utils.assign_node_ordering(gb)

model = DarcyAndTransport(gb)
add_data_darcy(gb, domain, tol)
model.discretize()
A, b = model.assemble_matrix_rhs()
solution = pp.solve(A, b)
model.extract_variables(solution)
model.project_discharge(gb)
model.project_pressure(gb)

total_flow = 0.0
for g, d in gb:
    if g.dim == gb.dim_max():
        right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
        total_flow += np.sum(d['darcy_flux'][right] * g.face_areas[right])
to_vtk.export_vtk(gb, f"{export_folder}/darcy", {'pressure': 'pressure', 'darcy_flux': 'darcy_flux'})

physics = {'advection': True, 'diffusion': False}
advection_discretization = pp.advection.Upwind()
mass_matrix_discretization = pp.MassMatrix()
add_data_advection(gb, domain, tol)
for g, d in gb:
    d['time_step'] = dt

advection_matrices = {}
mass_matrices = {}
rhs_advection = {}
rhs_mass = {}
for g, d in gb:
    advection_matrices[g] = advection_discretization.matrix(g, d)
    mass_matrices[g] = mass_matrix_discretization.matrix(g, d)
    rhs_advection[g] = advection_discretization.rhs(g, d)
    rhs_mass[g] = mass_matrix_discretization.rhs(g, d)

lu_factors = {}
for g in gb.grids_of_dimension(gb.dim_max()):
    A_adv = mass_matrices[g] / dt + advection_matrices[g]
    lu_factors[g] = splu(A_adv.tocsc())

solution_vec = {}
for g, d in gb:
    solution_vec[g] = np.zeros(g.num_cells)
times = np.linspace(0, time, num_time_steps + 1)
production = np.zeros(num_time_steps + 1)

for step in range(1, num_time_steps + 1):
    t = step * dt
    for g, d in gb:
        rhs = rhs_mass[g] / dt + rhs_advection[g]
        solution_vec[g] = lu_factors[g].solve(rhs)
        d['concentration'] = solution_vec[g]
    for g, d in gb:
        if g.dim == gb.dim_max():
            right = np.where(np.abs(g.face_centers[0] - domain['xmax']) < tol)[0]
            production[step] += np.sum(d['concentration'][right] * d['darcy_flux'][right] * g.face_areas[right])
    if step % export_frequency == 0 or step == num_time_steps:
        to_vtk.export_vtk(gb, f"{export_folder}/advection_{step}", {'concentration': 'concentration'})

export_pvd(f"{export_folder}/advection.pvd", [f"{export_folder}/advection_{step}" for step in range(0, num_time_steps + 1, export_frequency)])
np.savetxt(f"{export_folder}/production.txt", np.vstack((times, np.abs(production))).T, header="time production")
```