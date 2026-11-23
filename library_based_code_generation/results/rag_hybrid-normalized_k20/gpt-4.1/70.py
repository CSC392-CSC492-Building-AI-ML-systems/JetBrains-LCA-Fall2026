```python
import numpy as np
from scipy import sparse
import porepy as pp

def add_data(gb, domain, perm_factor):
    for g, d in gb:
        k = perm_factor * np.ones(g.num_cells)
        d[pp.PARAMETERS] = pp.Parameters(g)
        d[pp.PARAMETERS].set_value("permeability", k)
        d[pp.PARAMETERS].set_value("aperture", np.ones(g.num_cells))
        d[pp.PARAMETERS].set_value("source", np.zeros(g.num_cells))
        bc_type = "dirichlet"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        tol = 1e-7
        centers = g.face_centers
        left = np.abs(centers[0] - domain["xmin"]) < tol
        right = np.abs(centers[0] - domain["xmax"]) < tol
        bc_values[left] = 1.0
        bc_values[right] = 0.0
        d[pp.PARAMETERS].set_value("bc", bc)
        d[pp.PARAMETERS].set_value("bc_values", bc_values)
    for e, d in gb.edges():
        mg = gb.mdg.nodes[e]
        d[pp.PARAMETERS] = pp.Parameters(mg)
        d[pp.PARAMETERS].set_value("aperture", np.ones(mg.num_cells))
        d[pp.PARAMETERS].set_value("permeability", np.ones(mg.num_cells))
        d[pp.PARAMETERS].set_value("source", np.zeros(mg.num_cells))

def write_network(file_name):
    network_str = """# Predefined fracture network
2
0.2 0.2 0.8 0.2
0.5 0.0 0.5 1.0
"""
    with open(file_name, "w") as f:
        f.write(network_str)

def main(perm_factor, description, coarsen=False, export=False):
    mesh_size = 0.05
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    file_name = f"network_{description}.txt"
    write_network(file_name)
    network = pp.fracs.fracture_importer.FractureNetwork2d.from_file(file_name, domain)
    gb = network.mesh(mesh_size)
    gb.compute_geometry()
    if coarsen:
        gb = pp.coarsening.generate_coarse_grid(gb)
    pp.utils.grid_bucket.assign_node_ordering(gb)
    add_data(gb, domain, perm_factor)
    flow = pp.ad.Flow(gb)
    flow.discretize()
    A, b = flow.assemble_matrix_rhs()
    sol = sparse.linalg.spsolve(A, b)
    flow.split(sol)
    discharge = flow.extract_discharge()
    pressure = flow.extract_pressure()
    projected_discharge = flow.project_discharge()
    if export:
        pp.io._export_vtk_grid.export_vtk(gb, f"results_{description}.vtk", {"pressure": pressure, "discharge": projected_discharge})

def test_vem_blocking():
    main(perm_factor=1e-6, description="blocking", coarsen=False, export=False)

def test_vem_permeable():
    main(perm_factor=1e2, description="permeable", coarsen=False, export=False)
```