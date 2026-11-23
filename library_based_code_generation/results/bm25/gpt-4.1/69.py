```python
import argparse
import numpy as np
import matplotlib.pyplot as plt

from vem.mesh import PolygonalMesh
from vem.space import VEMSpace
from vem.assembly import assemble_stiffness_matrix, assemble_rhs
from vem.boundary import apply_dirichlet_bc
from vem.error import compute_error
from vem.refinement import mark_cells_for_refinement, refine_mesh_adaptively

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solve Poisson equation using VEM on polygonal mesh.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the VEM space')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of mesh refinement iterations')
parser.add_argument('--adaptive_param', type=float, default=0.5, help='Adaptive parameter for refinement')
args = parser.parse_args()

degree = args.degree
max_iter = args.max_iter
adaptive_param = args.adaptive_param

# Problem definition: Poisson equation -Δu = f on unit square with u=0 on boundary
def f(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_solution_grad(x, y):
    return np.array([np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                     np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)])

# Initialize mesh and error matrix
domain = [[0, 0], [1, 0], [1, 1], [0, 1]]
mesh = PolygonalMesh(domain, initial_divisions=2)
error_matrix = []
dof_list = []

DOF_THRESHOLD = 10000

for iteration in range(max_iter):
    # Set up VEM space
    vem_space = VEMSpace(mesh, degree)
    ndof = vem_space.num_dofs()
    dof_list.append(ndof)

    # Assemble stiffness matrix and right-hand side
    A = assemble_stiffness_matrix(vem_space, a=lambda x, y: 1.0)
    b = assemble_rhs(vem_space, f)

    # Apply Dirichlet boundary conditions
    boundary_nodes = mesh.get_boundary_nodes()
    u_D = lambda x, y: 0.0
    A_bc, b_bc = apply_dirichlet_bc(A, b, vem_space, boundary_nodes, u_D)

    # Solve linear system
    u = np.linalg.solve(A_bc, b_bc)

    # Compute error
    err = compute_error(vem_space, u, exact_solution)
    error_matrix.append(err)

    print(f"Iteration {iteration+1}: ndof={ndof}, error={err}")

    # Mark cells for refinement based on residual estimate
    residuals = vem_space.compute_residuals(u, f)
    marked_cells = mark_cells_for_refinement(residuals, adaptive_param)

    # Adaptively refine the mesh
    if len(marked_cells) > 0:
        mesh = refine_mesh_adaptively(mesh, marked_cells)
    else:
        break

    # Terminate if number of degrees of freedom exceeds threshold
    if ndof > DOF_THRESHOLD:
        print("DOF threshold reached. Stopping refinement.")
        break

# Save error rates and DOFs
np.savetxt('vem_dofs.txt', np.array(dof_list), fmt='%d')
np.savetxt('vem_errors.txt', np.array(error_matrix))

# Plot error convergence
plt.figure()
plt.loglog(dof_list, error_matrix, marker='o')
plt.xlabel('Degrees of Freedom')
plt.ylabel('Error')
plt.title('VEM Error Convergence')
plt.grid(True)
plt.savefig('vem_error_convergence.png')
plt.show()

# Plot final mesh
plt.figure()
mesh.plot()
plt.title('Final Polygonal Mesh')
plt.savefig('vem_final_mesh.png')
plt.show()
```