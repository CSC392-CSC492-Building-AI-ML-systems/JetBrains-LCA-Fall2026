```python
import numpy as np
import scipy.sparse as sps
import porepy as pp

def add_data(gb, domain, perm_factor):
    for g, d in gb:
        k = perm_factor * np.ones(g.num_cells)
        d['permeability'] = pp.SecondOrderTensor(g.dim, k)
        d['source'] = np.zeros(g.num_cells)
        if g.dim == 2:
            d['aperture'] = np.ones(g.num_cells)
        elif g.dim == 1:
            d['aperture'] = np.ones(g.num_cells) * 1e-3
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=np.array(['dir']*g.num_faces))
        d['bc'] = bc
        bc_values = np.zeros(g.num_faces)
        d['bc_values'] = bc_values

def write_network(file_name):
    network_str = """
    2
    0.2 0.2 0.8 0.2
    0.5 0.0 0.5 1.0
    """
    with open(file_name, 'w') as f:
        f.write(network_str)

def main(perm_factor, description, coarsen=False, export=False):
    mesh_size = {"h": 0.05}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    file_name = f"network_{description}.txt"
    write_network(file_name)
    network = pp.fracs.ImportNetwork2d(file_name)
    gb = network.mesh(mesh_size, domain)
    pp.fracs.compute_geometry(gb)
    if coarsen:
        gb = pp.coarsen.coarsen(gb)
    pp.fracs.assign_node_ordering(gb)
    add_data(gb, domain, perm_factor)
    flow = pp.DarcyAndTransport(gb)
    flow.assemble_matrix_rhs()
    A, b = flow.A, flow.b
    sol = sps.linalg.spsolve(A, b)
    flow.split(sol)
    discharge = flow.extract_discharge()
    pressure = flow.extract_pressure()
    projected_discharge = flow.project_discharge()
    if export:
        exporter = pp.Exporter(gb, f"results_{description}")
        exporter.write_vtk({"pressure": pressure, "projected_discharge": projected_discharge})

def test_vem_blocking():
    main(perm_factor=1e-6, description="blocking", coarsen=False, export=False)

def test_vem_permeable():
    main(perm_factor=1e2, description="permeable", coarsen=False, export=False)
```