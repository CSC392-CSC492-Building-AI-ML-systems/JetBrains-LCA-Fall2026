```python
import numpy as np
from scipy import sparse
import porepy as pp

def add_data(gb, domain, perm_factor):
    for g, d in gb:
        k = perm_factor * np.ones(g.num_cells)
        d[pp.PARAMETERS] = pp.Parameters(g)
        d[pp.PARAMETERS].set_value("permeability", k)
        d[pp.PARAMETERS].set_value("source", np.zeros(g.num_cells))
        if g.dim == 2:
            d[pp.PARAMETERS].set_value("aperture", np.ones(g.num_cells))
        bc_type = np.array(["dir"] * g.num_faces)
        bc_value = np.zeros(g.num_faces)
        tol = 1e-7
        for face, center in enumerate(g.face_centers.T):
            if abs(center[0] - domain[0][0]) < tol:
                bc_value[face] = 1.0
            elif abs(center[0] - domain[0][1]) < tol:
                bc_value[face] = 0.0
        d[pp.PARAMETERS].set_value("bc_type", bc_type)
        d[pp.PARAMETERS].set_value("bc_values", bc_value)

def write_network(file_name):
    network_str = """# Predefined fracture network
2
0.2 0.2 0.8 0.2
0.2 0.8 0.8 0.8
"""
    with open(file_name, "w") as f:
        f.write(network_str)

def main(perm_factor, description, coarsen=False, export=False):
    mesh_size = {"h": 0.05}
    domain = [[0, 1], [0, 1]]
    file_name = "network.csv"
    write_network(file_name)
    network = pp.fracs.network_2d_from_csv(file_name)
    gb = pp.fracs.grid_bucket_2d(network, domain, mesh_size)
    gb.compute_geometry()
    if coarsen:
        gb = pp.coarsening.generate_coarse_grid(gb)
    pp.fracs.assign_node_ordering(gb)
    add_data(gb, domain, perm_factor)
    flow = pp.ad.Flow(gb)
    source = pp.ad.Source(gb)
    flow.discretize()
    source.discretize()
    A, b = flow.assemble_matrix_rhs()
    sol = sparse.linalg.spsolve(A, b)
    sol_split = flow.split(sol)
    discharge = flow.extract_discharge(sol_split)
    pressure = flow.extract_pressure(sol_split)
    projected_discharge = flow.project_discharge(discharge)
    if export:
        pp.io._export_vtk_grid(gb, f"results_{description}.vtk", {"pressure": pressure, "discharge": projected_discharge})

def test_vem_blocking():
    main(perm_factor=1e-6, description="blocking", coarsen=False, export=False)

def test_vem_permeable():
    main(perm_factor=1e2, description="permeable", coarsen=False, export=False)
```