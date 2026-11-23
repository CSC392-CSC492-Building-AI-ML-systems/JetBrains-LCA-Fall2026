```python
import os
import numpy as np
import porepy as pp
from porepy.io.grid_import import dfm_2d_from_csv
from porepy.utils.exporter import _export_vtk_gb
from porepy.solvers.darcy import Darcy
from porepy.solvers.transport import Transport
from porepy.solvers.mass_matrix import MassMatrix
from porepy.utils.data_assigner import ParabolicDataAssigner
from scipy.sparse.linalg import splu

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1.0 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = 1.0 * np.ones(g.num_cells)
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        # Example: left boundary Dirichlet, right boundary Neumann
        left = np.where(np.abs(g.face_centers[0] - domain["xmin"]) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain["xmax"]) < tol)[0]
        bc_type[left] = pp.BC_DIRICHLET
        bc_value[left] = 1.0
        bc_type[right] = pp.BC_NEUMANN
        bc_value[right] = 0.0
        d.update({
            "permeability": k,
            "source": source,
            "aperture": aperture,
            "bc": pp.BoundaryCondition(g, faces=np.hstack([left, right]), cond=bc_type[np.hstack([left, right])]),
            "bc_values": bc_value,
        })
    for e, d in gb.edges():
        d["aperture"] = 1.0

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.2 * np.ones(g.num_cells)
        discharge = np.zeros((g.dim, g.num_cells))
        bc_type = np.zeros(g.num_faces, dtype=np.int32)
        bc_value = np.zeros(g.num_faces)
        # Example: left boundary Dirichlet, right boundary Neumann
        left = np.where(np.abs(g.face_centers[0] - domain["xmin"]) < tol)[0]
        right = np.where(np.abs(g.face_centers[0] - domain["xmax"]) < tol)[0]
        bc_type[left] = pp.BC_DIRICHLET
        bc_value[left] = 1.0
        bc_type[right] = pp.BC_NEUMANN
        bc_value[right] = 0.0
        d.update({
            "source": source,
            "porosity": porosity,
            "discharge": discharge,
            "bc": pp.BoundaryCondition(g, faces=np.hstack([left, right]), cond=bc_type[np.hstack([left, right])]),
            "bc_values": bc_value,
        })
    for e, d in gb.edges():
        d["aperture"] = 1.0

tol = 1e-6
export_folder = "results"
time = 1.0
num_time_steps = 20
dt = time / num_time_steps
export_frequency = 5
coarsening = False

mesh_size = {"fracture": 0.05, "matrix": 0.1}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

if not os.path.exists(export_folder):
    os.makedirs(export_folder)

gb = dfm_2d_from_csv("input_grid.csv", domain=domain, mesh_size=mesh_size)
gb.compute_geometry()
if coarsening:
    gb = pp.coarsen(gb)
pp.set_node_ordering(gb)

darcy_solver = Darcy(gb)
add_data_darcy(gb, domain, tol)
darcy_solver.discretize()
darcy_solver.assemble_matrix_rhs()
darcy_solver.solve()

pressure = {}
discharge = {}
for g, d in gb:
    pressure[g] = d[pp.STATE][pp.PRIMARY_VARIABLES]["pressure"]
    discharge[g] = darcy_solver.extract_discharge(g, d)

total_flow = 0.0
for g, d in gb:
    right = np.where(np.abs(g.face_centers[0] - domain["xmax"]) < tol)[0]
    total_flow += np.sum(discharge[g][0, right] * g.face_areas[right])

_export_vtk_gb(gb, os.path.join(export_folder, "darcy_result"))

physics = {"advection": True, "diffusion": False}
advection_solver = Transport(gb)
mass_matrix_solver = MassMatrix(gb)
add_data_advection(gb, domain, tol)
ParabolicDataAssigner(gb).assign_time_step(dt)

advection_solver.discretize()
advection_solver.assemble_matrix_rhs()
A_adv = advection_solver.A
b_adv = advection_solver.b

mass_matrix_solver.discretize()
mass_matrix_solver.assemble_matrix_rhs()
M = mass_matrix_solver.A

A_sys = M + dt * A_adv
lu = splu(A_sys.tocsc())

solution = {}
for g, d in gb:
    solution[g] = np.zeros(g.num_cells)

times = np.zeros(num_time_steps + 1)
production = np.zeros(num_time_steps + 1)

for step in range(num_time_steps + 1):
    t = step * dt
    times[step] = t
    rhs = M @ np.concatenate([solution[g] for g, d in gb]) + dt * b_adv
    sol_vec = lu.solve(rhs)
    offset = 0
    for g, d in gb:
        nc = g.num_cells
        solution[g] = sol_vec[offset:offset + nc]
        offset += nc
    # Compute production (e.g., outflow at right boundary)
    prod = 0.0
    for g, d in gb:
        right = np.where(np.abs(g.face_centers[0] - domain["xmax"]) < tol)[0]
        prod += np.sum(solution[g][right])
    production[step] = np.abs(prod)
    if step % export_frequency == 0:
        for g, d in gb:
            d[pp.STATE]["transport"] = solution[g]
        _export_vtk_gb(gb, os.path.join(export_folder, f"advection_{step:03d}"))

# Export PVD file
with open(os.path.join(export_folder, "advection.pvd"), "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for step in range(0, num_time_steps + 1, export_frequency):
        f.write(f'    <DataSet timestep="{times[step]}" group="" part="0" file="advection_{step:03d}.vtu"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

np.savetxt(os.path.join(export_folder, "production.txt"), np.vstack((times, production)).T, header="time production")
```