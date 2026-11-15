```python
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

def add_data(gb, domain):
    for g, d in gb:
        if g.dim == 2:
            k = 1e-12 * np.ones(g.num_cells)
            d["permeability"] = pp.SecondOrderTensor(k)
            d["aperture"] = np.ones(g.num_cells)
            bc = pp.BoundaryCondition(g, faces=np.arange(g.num_faces), 
                                      cond=np.array(["dir"]*g.num_faces))
            bc_val = np.zeros(g.num_faces)
            tol = 1e-7
            x = g.face_centers[0]
            y = g.face_centers[1]
            bc_val[np.abs(x - domain["xmin"]) < tol] = 1
            bc_val[np.abs(x - domain["xmax"]) < tol] = 0
            d["bc"] = bc
            d["bc_values"] = bc_val
            d["source"] = np.zeros(g.num_cells)
        elif g.dim == 1:
            k = 1e-10 * np.ones(g.num_cells)
            d["permeability"] = pp.SecondOrderTensor(k)
            d["aperture"] = 1e-3 * np.ones(g.num_cells)
            d["bc"] = pp.BoundaryCondition(g, faces=np.array([], dtype=int), cond=np.array([]))
            d["bc_values"] = np.array([])
            d["source"] = np.zeros(g.num_cells)
        elif g.dim == 0:
            d["permeability"] = pp.SecondOrderTensor(np.array([1e-8]))
            d["aperture"] = np.array([1e-3])
            d["bc"] = pp.BoundaryCondition(g, faces=np.array([], dtype=int), cond=np.array([]))
            d["bc_values"] = np.array([])
            d["source"] = np.zeros(g.num_cells)
    for e, d in gb.edges():
        mg = gb.edge_grids(e)
        if mg.dim == 1:
            d["aperture"] = 1e-3 * np.ones(mg.num_cells)
        elif mg.dim == 0:
            d["aperture"] = np.array([1e-3])

def plot_over_line(gb, points, var_name, filename):
    vals = []
    for pt in points:
        for g, d in gb:
            if g.dim == 2:
                cell = g.locate_point(pt)
                if cell is not None:
                    vals.append(d[var_name][cell])
                    break
    vals = np.array(vals)
    np.savetxt(filename, np.column_stack((points[:,0], points[:,1], vals)), delimiter=',', header='x,y,value', comments='')
    plt.figure()
    plt.plot(np.linalg.norm(points - points[0], axis=1), vals, marker='o')
    plt.xlabel('Distance along line')
    plt.ylabel(var_name)
    plt.title(f'{var_name} along line')
    plt.savefig(filename.replace('.csv', '.png'))
    plt.close()

tol = 1e-7
mesh_size = {"fracture": 0.1, "matrix": 0.1}
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

gb = pp.io.grid_bucket.load_csv("grid_bucket.csv")
gb.compute_geometry()
pp.coarsening.generate_coarse_grid(gb, coarsening_level=1)
pp.utils.assign_node_ordering(gb)

add_data(gb, domain)

flow = pp.DualVEMMixDim("flow")
flow.discretize(gb)
A, b = flow.assemble_matrix_rhs(gb)
solution = pp.linalg.spsolve(A, b)

flow.split(gb, solution)
flow.extract_discharge(gb)
flow.project_discharge(gb)

pp.io.export_vtk.export_vtk(gb, "output", 
    data={"pressure": {g: d["pressure"] for g, d in gb},
          "discharge": {g: d["discharge"] for g, d in gb}})

bbox = np.array([[domain["xmin"], domain["ymin"]], [domain["xmax"], domain["ymax"]]])
n_points = 100
x_line = np.linspace(domain["xmin"], domain["xmax"], n_points)
y_line = np.linspace(domain["ymin"], domain["ymax"], n_points)
points_x = np.column_stack((x_line, np.full_like(x_line, 0.5)))
points_y = np.column_stack((np.full_like(y_line, 0.5), y_line))

plot_over_line(gb, points_x, "pressure", "pressure_over_x.csv")
plot_over_line(gb, points_y, "pressure", "pressure_over_y.csv")

diameter = max(np.linalg.norm(g.cell_centers.T - gb.bounding_box()[0], axis=1).max() for g, _ in gb)
num_cells_2d = sum(g.num_cells for g, _ in gb if g.dim == 2)
num_cells_1d = sum(g.num_cells for g, _ in gb if g.dim == 1)

print(f"Diameter of grid bucket: {diameter}")
print(f"Number of 2D cells: {num_cells_2d}")
print(f"Number of 1D cells: {num_cells_1d}")
```