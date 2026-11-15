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
        bc = pp.BoundaryCondition(g)
        bc_faces = np.zeros(g.num_faces, dtype=np.int32)
        bc_values = np.zeros(g.num_faces)
        tol = 1e-8
        if g.dim == 2:
            for face, center in enumerate(g.face_centers.T):
                if np.abs(center[0] - domain["xmin"]) < tol:
                    bc_faces[face] = pp.BoundaryCondition.Dirichlet
                    bc_values[face] = 1.0
                elif np.abs(center[0] - domain["xmax"]) < tol:
                    bc_faces[face] = pp.BoundaryCondition.Dirichlet
                    bc_values[face] = 0.0
                else:
                    bc_faces[face] = pp.BoundaryCondition.Neumann
                    bc_values[face] = 0.0
        else:
            bc_faces[:] = pp.BoundaryCondition.Neumann
            bc_values[:] = 0.0
        d["bc"] = pp.BoundaryCondition(g, bc_faces)
        d["bc_values"] = bc_values
        d["source"] = np.zeros(g.num_cells)
    for e, d in gb.edges():
        mg = gb.mdg.nodes[e]
        d["aperture"] = 1.0
        d["normal_diffusivity"] = 1.0

def plot_over_line(gb, points, var_name="pressure"):
    vals = []
    for pt in points:
        found = False
        for g, d in gb:
            if g.dim != 2:
                continue
            cell = g.locate_point(pt)
            if cell is not None:
                val = d[var_name][cell]
                vals.append(val)
                found = True
                break
        if not found:
            vals.append(np.nan)
    return np.array(vals)

tol = 1e-8
mesh_size = 0.1
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

gb = pp.io.grid_bucket.load_grid_bucket_csv("grid_bucket.csv")
gb.compute_geometry()
coarse_gb = pp.coarsening.generate_coarse_grid(gb, coarsening_level=1)
coarse_gb.assign_node_ordering()

add_data(coarse_gb, domain)

flow_solver = pp.DualVEMMixDim(coarse_gb, "flow")
A, b = flow_solver.assemble_matrix_rhs()
solution = pp.linalg.spsolve(A, b)
flow_solver.split(solution)
flow_solver.extract_discharge()
flow_solver.project_discharge()

for g, d in coarse_gb:
    d["pressure"] = d["solution"]
    if "discharge" in d:
        d["discharge_proj"] = d["discharge"]

pp.io.export_vtk.export_vtk_grid(coarse_gb, "output", {"pressure": "pressure", "discharge": "discharge_proj"})

bbox = [domain["xmin"], domain["xmax"], domain["ymin"], domain["ymax"]]
n_points = 100
x_line = np.linspace(domain["xmin"], domain["xmax"], n_points)
y_line = np.linspace(domain["ymin"], domain["ymax"], n_points)
points_x = np.vstack([x_line, 0.5 * np.ones_like(x_line)]).T
points_y = np.vstack([0.5 * np.ones_like(y_line), y_line]).T

pressure_x = plot_over_line(coarse_gb, points_x, var_name="pressure")
pressure_y = plot_over_line(coarse_gb, points_y, var_name="pressure")

df_x = pd.DataFrame({"x": x_line, "pressure": pressure_x})
df_y = pd.DataFrame({"y": y_line, "pressure": pressure_y})
df_x.to_csv("pressure_over_x.csv", index=False)
df_y.to_csv("pressure_over_y.csv", index=False)

diameter = max(domain["xmax"] - domain["xmin"], domain["ymax"] - domain["ymin"])
num_cells_2d = sum(g.num_cells for g, _ in coarse_gb if g.dim == 2)
num_cells_1d = sum(g.num_cells for g, _ in coarse_gb if g.dim == 1)
print(f"Diameter: {diameter}")
print(f"Number of 2D cells: {num_cells_2d}")
print(f"Number of 1D cells: {num_cells_1d}")
```