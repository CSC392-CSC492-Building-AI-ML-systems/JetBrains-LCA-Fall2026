```python
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ParabolicVEMSolver2d import ParabolicVEMSolver2d
from ParabolicCVEMModel2d import ParabolicCVEMModel2d
from PoissonCVEMEstimator import PoissonCVEMEstimator
from ConformingScalarVEMLaplaceIntegrator2d import ConformingScalarVEMLaplaceIntegrator2d
from test_from_polygon_gmsh import test_from_polygon_gmsh
from ifem_amg_solver import ifem_amg_solver

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solve Poisson equation using VEM on polygonal mesh.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the VEM space')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of mesh refinement iterations')
parser.add_argument('--adaptive_param', type=float, default=0.5, help='Adaptive parameter for marking')
args = parser.parse_args()

degree = args.degree
max_iter = args.max_iter
adaptive_param = args.adaptive_param

# Problem setup: Poisson equation -Delta u = f on unit square with Dirichlet BC
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def rhs_function(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def boundary_function(x, y):
    return exact_solution(x, y)

# Initialize mesh and error matrix
mesh = test_from_polygon_gmsh(domain='unit_square', h=0.2)
error_matrix = []
dof_list = []

dof_threshold = 100000

for it in range(max_iter):
    # Set up VEM space and function
    vem_model = ParabolicCVEMModel2d(mesh, degree)
    vem_space = vem_model.space

    # Assemble stiffness matrix and RHS
    integrator = ConformingScalarVEMLaplaceIntegrator2d(vem_space)
    A = integrator.assemble_matrix()
    F = vem_space.assemble_rhs(rhs_function)

    # Apply Dirichlet boundary conditions
    boundary_dofs = vem_space.get_boundary_dofs()
    boundary_values = vem_space.interpolate_boundary(boundary_function)
    A, F = vem_space.apply_dirichlet_bc(A, F, boundary_dofs, boundary_values)

    # Solve linear system
    u = ifem_amg_solver(A, F)

    # Compute error
    error = vem_space.compute_L2_error(u, exact_solution)
    error_matrix.append(error)
    dof_list.append(vem_space.ndof)

    print(f"Iteration {it+1}: ndof = {vem_space.ndof}, L2 error = {error:.4e}")

    # Mark cells for refinement based on residual estimator
    estimator = PoissonCVEMEstimator(mesh, vem_space, u, rhs_function)
    cell_errors = estimator.compute_cell_errors()
    threshold = adaptive_param * np.max(cell_errors)
    marked_cells = [i for i, e in enumerate(cell_errors) if e > threshold]

    # Adaptively refine the mesh
    if len(marked_cells) == 0:
        print("No cells marked for refinement. Stopping.")
        break
    mesh.refine_cells(marked_cells)

    # Terminate if dof exceeds threshold
    if vem_space.ndof > dof_threshold:
        print("DOF threshold exceeded. Stopping.")
        break

# Display error rates
error_matrix = np.array(error_matrix)
dof_list = np.array(dof_list)
rates = np.log(error_matrix[1:] / error_matrix[:-1]) / np.log(dof_list[1:] / dof_list[:-1])
print("Convergence rates:", rates)

# Save ndof and error matrix to text files
np.savetxt('vem_ndof.txt', dof_list)
np.savetxt('vem_error.txt', error_matrix)

# Plot error
plt.figure()
plt.loglog(dof_list, error_matrix, marker='o')
plt.xlabel('Degrees of Freedom')
plt.ylabel('L2 Error')
plt.title('VEM Poisson Error Convergence')
plt.grid(True)
plt.savefig('vem_error_plot.png')
plt.show()

# Plot final mesh
mesh.plot()
plt.title('Final Polygonal Mesh')
plt.savefig('vem_final_mesh.png')
plt.show()
```