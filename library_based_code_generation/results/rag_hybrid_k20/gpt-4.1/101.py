```python
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

def add_data(gb, domain):
    for g, d in gb:
        if g.dim == 2:
            perm = 1e-12 * np.ones(g.num_cells)
            aperture = 1.0
        elif g.dim == 1:
            perm = 1e-8 * np.ones(g.num_cells)
            aperture = 1e-3
        else:
            perm = 1.0 * np.ones(g.num_cells)
            aperture = 1.0
        d["permeability"] = pp.SecondOrderTensor(g.dim, perm)
        d["aperture"] = aperture
        bc_type = np.array(["dir"] * g.num_faces)
        bc_value = np.zeros(g.num_faces)
        tol = 1e-7
        for face, center in enumerate(g.face_centers.T):
            if np.abs(center[0] - domain["xmin"]) < tol:
                bc_type[face] = "dir"
                bc_value[face] = 1.0
            elif np.abs(center[0] - domain["xmax"]) < tol:
                bc_type[face] = "dir"
                bc_value[face] = 0.0
        d["bc"] = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), cond=bc_type)
        d["bc_values"] = bc_value

def plot_over_line(gb, points, variable, filename):
    values = []
    for pt in points:
        for g, d in gb:
            if g.dim == 2:
                cell = g.locate_point(pt)
                if cell is not None:
                    val = d[variable][cell]
                    values.append(val)
                    break
            elif g.dim == 1:
                cell = g.locate_point(pt)
                if cell is not None:
                    val = d[variable][cell]
                    values.append(val)
                    break
    arr = np.array(values)
    np.savetxt(filename, arr, delimiter=",")
    plt.plot(np.linspace(0, 1, len(arr)), arr)
    plt.xlabel("Position along line")
    plt.ylabel(variable)
    plt.title(f"{variable} over line")
    plt.savefig(filename.replace(".csv", ".png"))
    plt.close()

tol = 1e-7
mesh_size = {"2D": 0.1, "1D": 0.05}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

gb = pp.io.grid_bucket.load_grid_bucket_csv("grid_bucket.csv")
gb.compute_geometry()
coarse_gb = pp.coarsening.generate_coarse_grid(gb, coarsening_level=1)
coarse_gb.assign_node_ordering()

add_data(coarse_gb, domain)

flow = pp.DualVEMMixDim("flow")
flow.discretize(coarse_gb)
A, b = flow.assemble_matrix_rhs(coarse_gb)
solution = pp.linalg.spsolve(A, b)

flow.split(coarse_gb, solution)
for g, d in coarse_gb:
    if g.dim > 0:
        d["discharge"] = flow.project_discharge(g, d)
    d["pressure"] = d["pressure"]

pp.io.export_vtk(coarse_gb, "output", data=["pressure", "discharge"])

bbox = np.array([[domain["xmin"], domain["xmax"]], [domain["ymin"], domain["ymax"]]])
n_points = 100
x_line = np.linspace(domain["xmin"], domain["xmax"], n_points)
y_line = np.linspace(domain["ymin"], domain["ymax"], n_points)
points_x = np.vstack([x_line, 0.5 * np.ones_like(x_line)]).T
points_y = np.vstack([0.5 * np.ones_like(y_line), y_line]).T

plot_over_line(coarse_gb, points_x, "pressure", "pressure_over_x.csv")
plot_over_line(coarse_gb, points_y, "pressure", "pressure_over_y.csv")

diameter = max([g.diameter for g, _ in coarse_gb])
num_cells_2d = sum([g.num_cells for g, _ in coarse_gb if g.dim == 2])
num_cells_1d = sum([g.num_cells for g, _ in coarse_gb if g.dim == 1])

print(f"Diameter of grid bucket: {diameter}")
print(f"Number of 2D cells: {num_cells_2d}")
print(f"Number of 1D cells: {num_cells_1d}")
```