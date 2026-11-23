```python
import numpy as np
import scipy
import os
import sys

import porepy as pp
from porepy.models import DarcyAndTransport
from porepy.utils.parameters import Parameters

def add_data_darcy(gb, domain, tol):
    for g, d in gb:
        k = 1e-12 * np.ones(g.num_cells)
        source = np.zeros(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = 'dir'
        bc = pp.BoundaryCondition(g, faces=np.array([], dtype=int), bc_type=bc_type)
        bc_values = np.zeros(g.num_faces)
        d.update({
            "permeability": k,
            "source": source,
            "aperture": aperture,
            "bc": bc,
            "bc_values": bc_values,
        })
    for e, ed in gb.edges():
        ed["aperture"] = np.ones(e.num_cells)
        ed["permeability"] = 1e-12 * np.ones(e.num_cells)
        ed["bc"] = pp.BoundaryCondition(e, faces=np.array([], dtype=int), bc_type='neu')
        ed["bc_values"] = np.zeros(e.num_faces)

def add_data_advection(gb, domain, tol):
    for g, d in gb:
        source = np.zeros(g.num_cells)
        porosity = 0.1 * np.ones(g.num_cells)
        discharge = np.zeros((g.dim, g.num_cells))
        bc_type = 'dir'
        bc = pp.BoundaryCondition(g, faces=np.array([], dtype=int), bc_type=bc_type)
        bc_values = np.zeros(g.num_faces)
        d.update({
            "source": source,
            "porosity": porosity,
            "discharge": discharge,
            "bc": bc,
            "bc_values": bc_values,
        })
    for e, ed in gb.edges():
        ed["porosity"] = 0.1 * np.ones(e.num_cells)
        ed["discharge"] = np.zeros((e.dim, e.num_cells))
        ed["bc"] = pp.BoundaryCondition(e, faces=np.array([], dtype=int), bc_type='neu')
        ed["bc_values"] = np.zeros(e.num_faces)

sys.path.append(os.path.abspath("path_to_soultz_grid"))
import soultz_grid

domain = {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000, "zmin": 0, "zmax": 1000}
tol = 1e-5
coarsen = False

gb = soultz_grid.create_grid(domain, tol)
pp.compute_geometry(gb)
if coarsen:
    gb = pp.coarsen(gb, coarsening_factor=2)
pp.assign_node_ordering(gb)

darcy_model = DarcyAndTransport(gb, domain, tol)
add_data_darcy(gb, domain, tol)
darcy_model.discretize()
A, b = darcy_model.assemble_matrix_rhs()
solution = scipy.sparse.linalg.spsolve(A, b)
split_sol = darcy_model.split(solution)
pressure = split_sol["pressure"]
discharge = darcy_model.project_discharge(solution)
total_flow_rate = np.sum(discharge)

add_data_advection(gb, domain, tol)
transport_params = Parameters()
transport_params.set("time_step", 10.0)
transport_params.set("final_time", 1000.0)
transport_params.set("export_interval", 100.0)

advection_solver = pp.advection.Upwind(gb)
mass_matrix_solver = pp.MassMatrix(gb)
advection_solver.discretize()
mass_matrix_solver.discretize()
A_adv, b_adv = advection_solver.assemble_matrix_rhs()
M, _ = mass_matrix_solver.assemble_matrix_rhs()
lu = scipy.sparse.linalg.splu(M + transport_params.get("time_step") * A_adv)
c = np.zeros(M.shape[0])
time = 0.0
export_times = np.arange(0, transport_params.get("final_time") + transport_params.get("export_interval"),
                        transport_params.get("export_interval"))
export_idx = 0

while time < transport_params.get("final_time"):
    rhs = M @ c + transport_params.get("time_step") * b_adv
    c = lu.solve(rhs)
    time += transport_params.get("time_step")
    if export_idx < len(export_times) and time >= export_times[export_idx]:
        pp.to_vtk(gb, "transport_solution_t{:.0f}".format(time), data={"concentration": c})
        export_idx += 1

pp.to_vtk(gb, "final_transport_solution", data={"concentration": c})
np.savetxt("production_data.txt", np.array([total_flow_rate]))
```