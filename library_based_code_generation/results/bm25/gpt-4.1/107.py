```python
import numpy as np
import porepy as pp
import os

def add_data_to_grid_bucket(gb, permeability_factor=1.0):
    for g, d in gb:
        # Permeability
        perm = permeability_factor * np.ones(g.num_cells)
        if g.dim == 3:
            k = pp.SecondOrderTensor(perm)
        elif g.dim == 2:
            k = pp.SecondOrderTensor(perm)
            d["aperture"] = np.ones(g.num_cells) * 1e-3
        elif g.dim == 1:
            k = pp.SecondOrderTensor(perm)
            d["aperture"] = np.ones(g.num_cells) * 1e-4
        d["permeability"] = k

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

        # Source term
        d["source"] = np.zeros(g.num_cells)

    for e, ed in gb.edges():
        mg = gb.edge_grids(e)
        # Coupling permeability
        ed["permeability"] = np.ones(mg.num_cells) * permeability_factor
        ed["aperture"] = np.ones(mg.num_cells) * 1e-4

def write_network_to_csv(network_str, filename):
    with open(filename, "w") as f:
        f.write(network_str)

def main(permeability_factor, description, mesh_size, coarse=False):
    # Define network as a string
    network_str = """\
POINTS
0 0 0
1 0 0
1 1 0
0 1 0
SEGMENTS
0 1
1 2
2 3
3 0
"""
    csv_filename = "network.csv"
    write_network_to_csv(network_str, csv_filename)

    # Create grid bucket from CSV
    network = pp.fracs.network_from_csv(csv_filename)
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    mesh_args = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size, "mesh_size_bound": mesh_size}
    gb = pp.meshing.cartesian_grid_from_network(network, domain, mesh_args)

    # Compute geometry
    gb.compute_geometry()

    # Optionally generate coarse grid
    if coarse:
        pp.coarsening.coarse_grid(gb)

    # Assign parameters
    add_data_to_grid_bucket(gb, permeability_factor=permeability_factor)

    # Set up and solve
    solver = pp.DualVEMMixDim(gb, solver_type="direct")
    solver.discretize()
    solver.assemble_matrix_rhs()
    solution = solver.solve()
    solver.extract_and_project(solution)

    # Export to VTK
    vtk_folder = "vtk_results"
    os.makedirs(vtk_folder, exist_ok=True)
    vtk_file = os.path.join(vtk_folder, f"result_{description}.vtk")
    pp.io.to_vtk(gb, vtk_file, {"pressure": solver.extract_pressure(solution)})

    # Print grid bucket info
    print(f"Scenario: {description}")
    print(f"Number of grids: {len(gb.grids())}")
    for g, d in gb:
        print(f"Grid dim: {g.dim}, num cells: {g.num_cells}")

def blocking_scenario():
    main(permeability_factor=1e-10, description="blocking", mesh_size=0.1)

def permeable_scenario():
    main(permeability_factor=1.0, description="permeable", mesh_size=0.1)

blocking_scenario()
permeable_scenario()
```