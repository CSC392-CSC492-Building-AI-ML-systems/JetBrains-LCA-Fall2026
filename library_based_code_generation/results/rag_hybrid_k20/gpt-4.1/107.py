```python
import numpy as np
import porepy as pp
import os

def add_data_to_grid_bucket(gb, perm_factor=1.0):
    for g, d in gb:
        # Permeability tensor
        k = perm_factor * np.ones(g.dim)
        d['permeability'] = pp.SecondOrderTensor(k)
        # Aperture
        if g.dim < 3:
            d['aperture'] = 1e-3 if g.dim == 2 else 1e-4
        # Boundary conditions
        bc_type = np.array(['dir'] * g.num_faces)
        bc_value = np.zeros(g.num_faces)
        # Set Dirichlet on left, Neumann elsewhere
        left_faces = np.where(np.abs(g.face_centers[0] - g.bounds[0][0]) < 1e-7)[0]
        bc_type[left_faces] = 'dir'
        bc_value[left_faces] = 1.0
        right_faces = np.where(np.abs(g.face_centers[0] - g.bounds[0][1]) < 1e-7)[0]
        bc_type[right_faces] = 'dir'
        bc_value[right_faces] = 0.0
        d['bc'] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d['bc_values'] = bc_value
        d['source'] = np.zeros(g.num_cells)
    for e, ed in gb.edges():
        # Coupling permeability
        mg, sg = e
        mg_dim, sg_dim = mg.dim, sg.dim
        if mg_dim == 3 and sg_dim == 2:
            ed['permeability'] = 1e-2 * perm_factor
        elif mg_dim == 2 and sg_dim == 1:
            ed['permeability'] = 1e-3 * perm_factor
        else:
            ed['permeability'] = 1e-4 * perm_factor

def write_network_to_csv(network_str, filename):
    with open(filename, 'w') as f:
        f.write(network_str)

def main(perm_factor, description, mesh_size, coarse=False):
    # Define network as a string
    network_str = """\
name,x1,y1,z1,x2,y2,z2
frac1,0,0,0,1,0,0
frac2,0.5,-0.5,0,0.5,0.5,0
"""
    csv_filename = "network.csv"
    write_network_to_csv(network_str, csv_filename)
    # Create grid bucket from CSV
    network = pp.network_3d_from_csv(csv_filename)
    domain = {"xmin": 0, "xmax": 1, "ymin": -0.5, "ymax": 0.5, "zmin": 0, "zmax": 0}
    gb = pp.meshing.simplex.triangle_grid_from_gmsh(network, domain, mesh_size)
    gb.compute_geometry()
    if coarse:
        gb = pp.coarsening.generate_coarse_grid_gb(gb, coarsening_factor=2)
        gb.compute_geometry()
    add_data_to_grid_bucket(gb, perm_factor)
    # Set up and solve
    solver = pp.DualVEMMixDim(gb, solver_type="direct")
    solver.discretize()
    solver.assemble_matrix_rhs()
    solution = solver.solve()
    # Project solution
    for g, d in gb:
        d['pressure'] = solution[solver.dof_manager.dof_ind(g, 'pressure')]
    # Export to VTK
    vtk_folder = "vtk_results"
    os.makedirs(vtk_folder, exist_ok=True)
    vtk_file = os.path.join(vtk_folder, f"result_{description}.vtk")
    pp.io._export_vtk_grid.export_vtk(gb, "pressure", vtk_file)
    # Print info
    print(f"Scenario: {description}")
    print(f"Number of grids: {len(gb.grids())}")
    for g in gb.grids():
        print(f"Grid dim: {g.dim}, num cells: {g.num_cells}")

def blocking_scenario():
    main(perm_factor=1e-8, description="blocking", mesh_size=0.05, coarse=False)

def permeable_scenario():
    main(perm_factor=1.0, description="permeable", mesh_size=0.05, coarse=False)

blocking_scenario()
permeable_scenario()
```