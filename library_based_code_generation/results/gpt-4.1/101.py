```python
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
import pandas as pd

def add_data(gb, domain):
    for g, d in gb:
        if g.dim == 2:
            perm = 1.0
            aperture = 1.0
        elif g.dim == 1:
            perm = 1e-2
            aperture = 1e-3
        else:
            perm = 1.0
            aperture = 1.0
        d["permeability"] = pp.SecondOrderTensor(g, perm * np.ones(g.num_cells))
        d["aperture"] = aperture * np.ones(g.num_cells)
        bc_type = "dir"
        bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type * np.ones(g.num_faces, dtype=object))
        bc_values = np.zeros(g.num_faces)
        tol = 1e-7
        if g.dim == 2:
            for face, center in enumerate(g.face_centers.T):
                if np.isclose(center[0], domain[0][0], atol=tol):
                    bc_values[face] = 1.0
                elif np.isclose(center[0], domain[0][1], atol=tol):
                    bc_values[face] = 0.0
        d["bc"] = bc
        d["bc_values"] = bc_values
    for e, d in gb.edges():
        mg = gb.nodes_of_edge(e)
        d["aperture"] = np.ones(gb.edge_props(e)["num_cells"])
        d["normal_diffusivity"] = 1.0

def plot_over_line(gb, points, var_name="pressure", filename=None):
    values = []
    for pt in points:
        found = False
        for g, d in gb:
            if g.dim != 2:
                continue
            cell = g.locate_point(pt)
            if cell is not None:
                val = d[var_name][cell]
                values.append(val)
                found = True
                break
        if not found:
            values.append(np.nan)
    plt.figure()
    dist = np.linalg.norm(points - points[0], axis=1)
    plt.plot(dist, values, marker='o')
    plt.xlabel("Distance along line")
    plt.ylabel(var_name.capitalize())
    if filename:
        plt.savefig(filename.replace('.csv', '.png'))
        df = pd.DataFrame({"distance": dist, var_name: values})
        df.to_csv(filename, index=False)
    plt.close()

tol = 1e-7
mesh_size = 0.1
domain = [[0, 1], [0, 1]]

gb = pp.io.grid_bucket.load_grid_bucket_csv("grid_bucket.csv")
gb.compute_geometry()
coarsen_factor = 2
pp.coarsening.coarsen(gb, coarsen_factor)
pp.utils.assign_node_ordering(gb)

add_data(gb, domain)

solver = pp.DualVEMMixDim(gb)
A, b = solver.matrix_rhs()
solution = pp.linalg.spsolve(A, b)

solution_split = solver.split(solution)
discharge = solution_split["discharge"]
pressure = solution_split["pressure"]
solver.project_discharge(discharge)

for g, d in gb:
    d["pressure"] = pressure[g]
    d["discharge"] = discharge[g] if g.dim == 2 else None

pp.io.export_vtk(gb, "results", data=["pressure", "discharge"])

bbox = np.array([[0, 0], [1, 1]])
n_points = 100
x_line = np.linspace(bbox[0, 0], bbox[1, 0], n_points)
y_line = np.linspace(bbox[0, 1], bbox[1, 1], n_points)
points_x = np.vstack([x_line, np.full_like(x_line, 0.5)]).T
points_y = np.vstack([np.full_like(y_line, 0.5), y_line]).T

plot_over_line(gb, points_x, var_name="pressure", filename="pressure_along_x.csv")
plot_over_line(gb, points_y, var_name="pressure", filename="pressure_along_y.csv")

diameter = max(np.linalg.norm(np.array(domain)[:, 1] - np.array(domain)[:, 0]))
num_cells_2d = sum(g.num_cells for g, _ in gb if g.dim == 2)
num_cells_1d = sum(g.num_cells for g, _ in gb if g.dim == 1)
print(f"Diameter: {diameter}")
print(f"Number of 2D cells: {num_cells_2d}")
print(f"Number of 1D cells: {num_cells_1d}")
```