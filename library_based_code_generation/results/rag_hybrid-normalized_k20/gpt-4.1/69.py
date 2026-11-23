```python
import argparse
import numpy as np
import matplotlib.pyplot as plt

from vem import (
    PoissonInterfaceVEMModel,
    PoissonCVEMEstimator,
    ConformingScalarVEMLaplaceIntegrator2d,
    assembly_for_sspace_and_vspace_with_vector_basis,
)
from mesh import PolygonalMesh2d, refine_mesh_adaptively
from utils import plot_error, plot_mesh

# Parse command line arguments
parser = argparse.ArgumentParser(description="Solve Poisson equation using VEM on polygonal mesh.")
parser.add_argument('--degree', type=int, default=1, help='Degree of the VEM space')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of mesh refinement iterations')
parser.add_argument('--adaptive_param', type=float, default=0.5, help='Adaptive parameter for refinement')
args = parser.parse_args()

degree = args.degree
max_iter = args.max_iter
adaptive_param = args.adaptive_param

# Problem setup: Poisson equation -Δu = f on unit square with Dirichlet BC
def f(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

domain = [[0, 0], [1, 0], [1, 1], [0, 1]]
mesh = PolygonalMesh2d(domain, initial_refinement=0)

error_matrix = []
dof_list = []

dof_threshold = 100000

for it in range(max_iter):
    # Set up VEM space and function
    vem_model = PoissonInterfaceVEMModel(mesh, degree)
    integrator = ConformingScalarVEMLaplaceIntegrator2d(mesh, degree)
    sspace, vspace = vem_model.sspace, vem_model.vspace

    # Assemble stiffness matrix and RHS
    A, rhs = assembly_for_sspace_and_vspace_with_vector_basis(sspace, vspace, integrator, f)

    # Apply Dirichlet boundary conditions
    boundary_dofs = sspace.find_boundary_dofs()
    u_D = np.array([u_exact(*sspace.dof_coordinates[dof]) for dof in boundary_dofs])
    A, rhs = sspace.apply_dirichlet_bc(A, rhs, boundary_dofs, u_D)

    # Solve linear system
    u = np.linalg.solve(A, rhs)

    # Compute error
    error = sspace.compute_L2_error(u, u_exact)
    error_matrix.append(error)
    dof_list.append(sspace.number_of_dofs())

    print(f"Iteration {it+1}: DOFs = {sspace.number_of_dofs()}, L2 error = {error:.3e}")

    # Mark cells for refinement based on residual estimate
    estimator = PoissonCVEMEstimator(mesh, degree, u, f)
    cell_errors = estimator.compute_cell_residuals()
    marked_cells = estimator.mark_cells(cell_errors, adaptive_param)

    # Adaptive mesh refinement
    mesh = refine_mesh_adaptively(mesh, marked_cells)

    # Terminate if DOF threshold exceeded
    if sspace.number_of_dofs() > dof_threshold:
        print("DOF threshold exceeded. Terminating refinement loop.")
        break

# Display error rates
error_matrix = np.array(error_matrix)
dof_list = np.array(dof_list)
rates = np.log(error_matrix[1:] / error_matrix[:-1]) / np.log(dof_list[1:] / dof_list[:-1])
print("Convergence rates:", rates)

# Save DOFs and error matrix to text files
np.savetxt("vem_dofs.txt", dof_list)
np.savetxt("vem_errors.txt", error_matrix)

# Plot error
plt.figure()
plt.loglog(dof_list, error_matrix, marker='o')
plt.xlabel("Degrees of Freedom")
plt.ylabel("L2 Error")
plt.title("VEM Poisson Error Convergence")
plt.grid(True)
plt.savefig("vem_error_plot.png")
plt.show()

# Plot final mesh
plot_mesh(mesh, show=True, save_path="vem_final_mesh.png")
```