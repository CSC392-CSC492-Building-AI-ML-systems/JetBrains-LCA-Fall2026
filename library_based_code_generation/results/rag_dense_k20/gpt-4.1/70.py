```python
import numpy as np
import scipy.sparse as sps
import porepy as pp

def add_data(gb, domain, perm_factor):
    for g, d in gb:
        k = perm_factor * np.ones(g.num_cells)
        d['permeability'] = pp.SecondOrderTensor(k)
        d['aperture'] = np.ones(g.num_cells)
        d['source'] = np.zeros(g.num_cells)
        bc_type = 'dir' if g.dim == domain['dim'] else 'neu'
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc'] = bc
        d['bc_values'] = np.zeros(g.num_faces)
        if g.dim == domain['dim']:
            tol = 1e-7
            left = np.abs(g.face_centers[0] - domain['xmin']) < tol
            right = np.abs(g.face_centers[0] - domain['xmax']) < tol
            d['bc'].is_dir[left] = True
            d['bc'].is_dir[right] = True
            d['bc_values'][left] = 1.0
            d['bc_values'][right] = 0.0

def write_network(file_name):
    network_str = """
    2
    0.2 0.2 0.8 0.2
    0.5 0.0 0.5 1.0
    """
    with open(file_name, 'w') as f:
        f.write(network_str)

def main(perm_factor, description, coarsen=False, export=False):
    mesh_size = 0.05
    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'dim': 2}
    file_name = f"network_{description}.txt"
    write_network(file_name)
    network = pp.fracs.ImportNetwork2d.from_file(file_name)
    gb = network.mesh(mesh_size, domain=[domain['xmin'], domain['xmax'], domain['ymin'], domain['ymax']])
    gb.compute_geometry()
    if coarsen:
        pp.coarsening.coarsen(gb)
    pp.fvutils.assign_node_ordering(gb)
    add_data(gb, domain, perm_factor)
    flow_solver = pp.algorithms.Flow(gb)
    flow_solver.discretize()
    A, b = flow_solver.assemble_matrix_rhs()
    sol = sps.linalg.spsolve(A, b)
    flow_solver.split(sol)
    discharge = {}
    pressure = {}
    for g, d in gb:
        pressure[g] = d['pressure']
        discharge[g] = d['discharge']
    projected_discharge = {}
    for g, d in gb:
        projected_discharge[g] = pp.fvutils.project_discharge(g, d['discharge'])
    if export:
        exporter = pp.Exporter(gb, f"results_{description}")
        for g, d in gb:
            exporter.write_vtk([('pressure', d['pressure']), ('projected_discharge', projected_discharge[g])])

def test_vem_blocking():
    main(perm_factor=1e-6, description="blocking", coarsen=False, export=False)

def test_vem_permeable():
    main(perm_factor=1e2, description="permeable", coarsen=False, export=False)
```