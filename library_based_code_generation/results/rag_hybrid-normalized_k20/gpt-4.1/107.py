```python
import numpy as np
import porepy as pp
import os

def add_data_to_grid_bucket(gb, permeability_factor=1.0):
    for g, d in gb:
        # Permeability tensor
        perm = permeability_factor * np.ones(g.dim)
        d["permeability"] = pp.SecondOrderTensor(g, perm)
        # Aperture
        if g.dim < 3:
            d["aperture"] = np.ones(g.num_cells)
        # Boundary conditions
        bc_type = np.array(["dir"] * g.num_faces)
        bc_value = np.zeros(g.num_faces)
        # Set Dirichlet on left, Neumann elsewhere
        left = np.where(np.abs(g.face_centers[0] - g.bounds[0][0]) < 1e-7)[0]
        right = np.where(np.abs(g.face_centers[0] - g.bounds[0][1]) < 1e-7)[0]
        bc_type[left] = "dir"
        bc_value[left] = 1.0
        bc_type[right] = "dir"
        bc_value[right] = 0.0
        d["bc"] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d["bc_values"] = bc_value
        d["source"] = np.zeros(g.num_cells)
    for e, ed in gb.edges():
        # Coupling permeability
        ed["permeability"] = np.ones(ed["mortar_grid"].num_cells)

def write_network_to_csv(network_str, filename):
    with open(filename, "w") as f:
        f.write(network_str)

def main(permeability_factor, description, mesh_size, coarse=False):
    # Define network as a string
    network_str = """x,y,z,r
0.2,0.5,0,0.01
0.8,0.5,0,0.01
"""
    csv_file = "network.csv"
    write_network_to_csv(network_str, csv_file)

    # Create grid bucket from CSV
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    mesh_args = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}
    network = pp.fracs.network_from_csv(csv_file)
    gb = pp.meshing.simplex.grid_bucket_from_network(network, domain, mesh_args)

    # Compute geometry
    gb.compute_geometry()

    # Optionally generate coarse grid
    if coarse:
        gb = pp.coarsening.generate_coarse_grid_gb(gb)

    # Assign parameters
    add_data_to_grid_bucket(gb, permeability_factor)

    # Set up and solve system
    solver = pp.DualVEMMixDim("flow")
    solver.discretize(gb)
    solver.assemble_matrix_rhs(gb)
    solution = solver.solve(gb)

    # Project solution
    solver.extract_and_project(gb, solution)

    # Export to VTK
    vtk_folder = "vtk_results"
    os.makedirs(vtk_folder, exist_ok=True)
    vtk_file = os.path.join(vtk_folder, f"result_{description}.vtk")
    pp.io._export_vtk_grid.export_vtk(gb, "pressure", vtk_file)

    # Print grid bucket info
    print(f"Scenario: {description}")
    print(f"Number of grids: {len(gb.grids())}")
    for g, d in gb:
        print(f"Grid dim: {g.dim}, num cells: {g.num_cells}")

def blocking_scenario():
    main(permeability_factor=1e-8, description="blocking", mesh_size=0.05)

def permeable_scenario():
    main(permeability_factor=1.0, description="permeable", mesh_size=0.05)

blocking_scenario()
permeable_scenario()
```