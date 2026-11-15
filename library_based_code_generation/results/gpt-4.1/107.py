```python
import numpy as np
import porepy as pp
import os

def add_data_to_grid_bucket(gb, permeability_factor=1.0):
    for g, d in gb:
        if g.dim == 3:
            perm = permeability_factor * np.ones(g.num_cells)
        elif g.dim == 2:
            perm = permeability_factor * np.ones(g.num_cells)
        elif g.dim == 1:
            perm = permeability_factor * np.ones(g.num_cells)
        else:
            perm = permeability_factor * np.ones(g.num_cells)
        aperture = np.ones(g.num_cells)
        bc_type = "dir"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        bc_values = np.zeros(g.num_faces)
        d.update({
            "permeability": pp.SecondOrderTensor(perm),
            "aperture": aperture,
            "bc": bc,
            "bc_values": bc_values,
            "source": np.zeros(g.num_cells),
        })
    for e, ed in gb.edges():
        mg = gb.mdg.edges[e]
        g_l, g_h = mg
        # Assign coupling permeability
        edge_perm = permeability_factor * np.ones(gb.edge_props(e)["num_cells"])
        ed["permeability"] = edge_perm
        ed["aperture"] = np.ones_like(edge_perm)

def write_network_to_csv(network_str, filename):
    with open(filename, "w") as f:
        f.write(network_str)

def main(permeability_factor, description, mesh_size, coarse=False):
    # Define network as a string
    network_str = """x,y,z,r
0.2,0.2,0.5,0.01
0.8,0.8,0.5,0.01
"""
    csv_filename = "network.csv"
    write_network_to_csv(network_str, csv_filename)

    # Create grid bucket from CSV
    network = pp.fracs.network_from_csv(csv_filename)
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    mesh_args = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size, "mesh_size_bound": mesh_size}
    gb = pp.meshing.cartesian_grid_from_network(network, domain, mesh_args)

    # Compute geometry
    for g, d in gb:
        pp.fvutils.compute_geometry(g)

    # Optionally generate coarse grid
    if coarse:
        gb = pp.coarsening.coarse_grid(gb)

    # Assign parameters
    add_data_to_grid_bucket(gb, permeability_factor=permeability_factor)

    # Solve system
    solver = pp.DualVEMMixDim(gb)
    solver.discretize()
    solver.assemble_matrix_rhs()
    solution = solver.solve()
    solver.extract_and_project(solution)

    # Export to VTK
    vtk_folder = "vtk_results"
    os.makedirs(vtk_folder, exist_ok=True)
    pp.export_vtk.export_grid_bucket(gb, os.path.join(vtk_folder, f"result_{description}"))

    # Print grid bucket info
    print(f"Description: {description}")
    print(f"Number of grids: {gb.num_graph_nodes()}")
    for g, d in gb:
        print(f"Grid dim: {g.dim}, num cells: {g.num_cells}")

def blocking_scenario():
    main(permeability_factor=1e-8, description="blocking", mesh_size=0.05)

def permeable_scenario():
    main(permeability_factor=1.0, description="permeable", mesh_size=0.05)

blocking_scenario()
permeable_scenario()
```