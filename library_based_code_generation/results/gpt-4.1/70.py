```python
import numpy as np
from scipy import sparse
import porepy as pp

def add_data(gb, domain, perm_factor):
    for g, d in gb:
        k = perm_factor * np.ones(g.num_cells)
        d["permeability"] = pp.SecondOrderTensor(k)
        d["source"] = np.zeros(g.num_cells)
        if g.dim == 2:
            d["aperture"] = np.ones(g.num_cells)
        elif g.dim == 1:
            d["aperture"] = np.ones(g.num_cells) * 1e-3
        bc_type = "dir"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        tol = 1e-7
        x = g.face_centers[0]
        y = g.face_centers[1]
        left = np.abs(x - domain[0][0]) < tol
        right = np.abs(x - domain[0][1]) < tol
        bc.is_dir[left] = True
        bc.is_dir[right] = True
        bc_values[left] = 1.0
        bc_values[right] = 0.0
        d["bc"] = bc
        d["bc_values"] = bc_values

def write_network(file_name):
    network_str = """2
0.2 0.2 0.8 0.2
0.5 0.0 0.5 1.0
"""
    with open(file_name, "w") as f:
        f.write(network_str)

def main(perm_factor, description, coarsen=False, export=False):
    mesh_size = {"h": 0.05}
    domain = [[0, 1], [0, 1]]
    file_name = f"network_{description}.txt"
    write_network(file_name)
    network = pp.fracs.ImportNetwork2d(domain, file_name)
    gb = network.mesh(mesh_size)
    gb.compute_geometry()
    if coarsen:
        pp.coarsening.coarsen(gb)
    pp.contact_conditions.assign_node_ordering(gb)
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
        exporter = pp.Exporter(gb, f"results_{description}")
        for g, d in gb:
            exporter.write_vtk([g], {"pressure": pressure[g], "discharge": projected_discharge[g]})

def test_vem_blocking():
    main(perm_factor=1e-6, description="blocking", coarsen=False, export=False)

def test_vem_permeable():
    main(perm_factor=1e2, description="permeable", coarsen=False, export=False)
```