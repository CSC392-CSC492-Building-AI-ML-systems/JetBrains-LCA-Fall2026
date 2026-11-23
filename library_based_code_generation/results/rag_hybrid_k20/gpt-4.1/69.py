```python
import argparse
import numpy as np
import matplotlib.pyplot as plt

from vem import (
    PoissonInterfaceVEMModel,
    ConformingScalarVEMLaplaceIntegrator2d,
    PoissonCVEMEstimator,
    assembly_for_sspace_and_vspace_with_vector_basis,
)
from mesh import PolygonalMesh2d, refine_mesh_adaptively
from utils import plot_mesh, plot_error

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

def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

domain = [[0, 0], [1, 0], [1, 1], [0, 1]]
mesh = PolygonalMesh2d(domain, initial_refinement=0)

error_matrix = []
dof_list = []

max_dof = 100000

for it in range(max_iter):
    # Set up VEM space and function
    vem_model = PoissonInterfaceVEMModel(mesh, degree)
    integrator = ConformingScalarVEMLaplaceIntegrator2d(mesh, degree)
    
    # Assemble stiffness matrix and RHS
    A, rhs = assembly_for_sspace_and_vspace_with_vector_basis(vem_model, integrator, f)
    
    # Apply Dirichlet boundary conditions
    boundary_dofs = vem_model.get_dirichlet_dofs()
    u_dirichlet = np.array([exact_solution(*vem_model.dof_coordinates[dof]) for dof in boundary_dofs])
    A, rhs = vem_model.apply_dirichlet_bc(A, rhs, boundary_dofs, u_dirichlet)
    
    # Solve linear system
    u = np.linalg.solve(A, rhs)
    
    # Compute error
    error = vem_model.compute_L2_error(u, exact_solution)
    error_matrix.append(error)
    dof_list.append(len(u))
    
    print(f"Iteration {it+1}: DOF = {len(u)}, L2 error = {error:.3e}")
    
    # Mark cells for refinement based on residual estimate
    estimator = PoissonCVEMEstimator(mesh, degree, u, f)
    cell_errors = estimator.compute_cell_errors()
    marked_cells = estimator.mark_cells(cell_errors, adaptive_param)
    
    # Adaptive mesh refinement
    mesh = refine_mesh_adaptively(mesh, marked_cells)
    
    if len(u) > max_dof:
        print("Maximum number of degrees of freedom exceeded. Stopping refinement.")
        break

# Display error rates
error_matrix = np.array(error_matrix)
dof_list = np.array(dof_list)
rates = np.log(error_matrix[1:] / error_matrix[:-1]) / np.log(dof_list[1:] / dof_list[:-1])
print("Convergence rates:", rates)

# Save DOF and error matrix to text files
np.savetxt("dof_list.txt", dof_list)
np.savetxt("error_matrix.txt", error_matrix)

# Plot error
plt.figure()
plt.loglog(dof_list, error_matrix, marker='o')
plt.xlabel('Degrees of Freedom')
plt.ylabel('L2 Error')
plt.title('Error vs DOF')
plt.grid(True)
plt.savefig("error_plot.png")
plt.show()

# Plot final mesh
plot_mesh(mesh)
plt.savefig("final_mesh.png")
plt.show()
```