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

        bc = pp.BoundaryCondition(g, faces=np.array([], dtype=int), cond=np.array([], dtype=int))
        bc_faces = g.get_boundary_faces()
        bc_types = np.array([pp.BoundaryConditionType.dirichlet] * bc_faces.size)
        bc_values = np.zeros(bc_faces.size)
        tol = 1e-7
        centers = g.face_centers[:, bc_faces]
        if g.dim == 2:
            for i, fc in enumerate(bc_faces):
                x, y = centers[:, i]
                if abs(x - domain["xmin"]) < tol:
                    bc_types[i] = pp.BoundaryConditionType.dirichlet
                    bc_values[i] = 1.0
                elif abs(x - domain["xmax"]) < tol:
                    bc_types[i] = pp.BoundaryConditionType.dirichlet
                    bc_values[i] = 0.0
                else:
                    bc_types[i] = pp.BoundaryConditionType.neumann
                    bc_values[i] = 0.0
        elif g.dim == 1:
            for i, fc in enumerate(bc_faces):
                x, y = centers[:, i]
                if abs(x - domain["xmin"]) < tol:
                    bc_types[i] = pp.BoundaryConditionType.dirichlet
                    bc_values[i] = 1.0
                elif abs(x - domain["xmax"]) < tol:
                    bc_types[i] = pp.BoundaryConditionType.dirichlet
                    bc_values[i] = 0.0
                else:
                    bc_types[i] = pp.BoundaryConditionType.neumann
                    bc_values[i] = 0.0
        d["bc"] = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_types)
        d["bc_values"] = bc_values

def plot_over_line(gb, variable, points, filename):
    values = []
    for pt in points:
        found = False
        for g, d in gb:
            if g.dim != 2:
                continue
            cell = g.locate_point(pt)
            if cell is not None:
                val = d[variable][cell]
                values.append(val)
                found = True
                break
        if not found:
            values.append(np.nan)
    np.savetxt(filename, np.column_stack((points[:, 0], points[:, 1], values)), delimiter=",", header="x,y,value", comments="")
    plt.figure()
    plt.plot(np.linalg.norm(points - points[0], axis=1), values, marker='o')
    plt.xlabel("Distance along line")
    plt.ylabel(variable)
    plt.title(f"{variable} along line")
    plt.savefig(filename.replace(".csv", ".png"))
    plt.close()

tol = 1e-7
mesh_size = {"2D": 0.1, "1D": 0.05}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

gb = pp.io.grid_bucket.load_csv("grid_bucket.csv")
gb.compute_geometry()
coarsen_factor = 2
gb = pp.coarsening.coarsen(gb, coarsen_factor)
pp.node_ordering.assign_node_ordering(gb)

add_data(gb, domain)

flow = pp.DualVEMMixDim("flow")
flow.discretize(gb)
A, b = flow.assemble_matrix_rhs(gb)
solution = pp.linalg.solve(A, b)

solution_split = flow.split(gb, solution)
pressure = {}
discharge = {}
for g, d in gb:
    pressure[g] = solution_split[g]["pressure"]
    discharge[g] = flow.project_discharge(g, d, solution_split[g])

for g, d in gb:
    d["pressure"] = pressure[g]
    d["discharge"] = discharge[g]

pp.io.vtk_export.export_grid_bucket(gb, "output", data=["pressure", "discharge"])

bbox = np.array([[domain["xmin"], domain["ymin"]], [domain["xmax"], domain["ymax"]]])
n_points = 100
x_line = np.linspace(domain["xmin"], domain["xmax"], n_points)
y_line = np.linspace(domain["ymin"], domain["ymax"], n_points)
points_x = np.column_stack((x_line, np.full_like(x_line, 0.5 * (domain["ymin"] + domain["ymax"]))))
points_y = np.column_stack((np.full_like(y_line, 0.5 * (domain["xmin"] + domain["xmax"])), y_line))

plot_over_line(gb, "pressure", points_x, "pressure_over_x.csv")
plot_over_line(gb, "pressure", points_y, "pressure_over_y.csv")

diameter = max(np.linalg.norm(np.array([domain["xmax"], domain["ymax"]]) - np.array([domain["xmin"], domain["ymin"]])))
num_cells_2d = sum(g.num_cells for g, _ in gb if g.dim == 2)
num_cells_1d = sum(g.num_cells for g, _ in gb if g.dim == 1)

print(f"Diameter of grid bucket: {diameter}")
print(f"Number of 2D cells: {num_cells_2d}")
print(f"Number of 1D cells: {num_cells_1d}")
```