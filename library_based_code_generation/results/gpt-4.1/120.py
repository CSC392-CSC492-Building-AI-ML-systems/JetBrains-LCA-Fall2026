```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.solver import LinearFESolver

# Parse command line arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <max_iter> <theta> <k>")
    sys.exit(1)

max_iter = int(sys.argv[1])
theta = float(sys.argv[2])
k = int(sys.argv[3])

# Initialize mesh (unit square as example)
mesh = HalfEdgeMesh2d.unit_square_mesh(n=8)

# Initialize error data storage
error_data = []

for it in range(max_iter):
    # Define function space
    space = LagrangeFiniteElementSpace(mesh, p=k)
    
    # Define boundary condition (homogeneous Dirichlet as example)
    bc = DirichletBC(space, lambda x: np.zeros_like(x[0]))
    
    # Assemble system (simplified friction problem)
    A = space.stiff_matrix()
    F = space.source_vector(lambda x: np.ones_like(x[0]))
    
    # Apply boundary condition
    bc.apply(A, F)
    
    # Solve the linear system
    solver = LinearFESolver()
    uh = solver.solve(A, F)
    
    # Calculate residuals (simplified as L2 norm of the residual)
    residual = np.linalg.norm(A @ uh - F)
    
    # Calculate high order terms (placeholder, e.g., H1 seminorm)
    high_order = space.integralalg.L2_norm(uh)
    
    # Save results and error data
    np.savez(f'results_iter_{it}.npz', uh=uh, residual=residual, high_order=high_order)
    error_data.append([mesh.number_of_cells(), residual, high_order])
    
    # Plot mesh and save image
    fig = plt.figure()
    ax = fig.gca()
    mesh.add_plot(ax)
    plt.title(f'Iteration {it}')
    plt.savefig(f'mesh_iter_{it}.png')
    plt.close(fig)
    
    # Refine mesh based on residuals if not last iteration
    if it < max_iter - 1:
        # Mark cells for refinement (simplified: mark all if residual > theta)
        if residual > theta:
            marked_cells = np.arange(mesh.number_of_cells())
            mesh.refine_triangle_mesh(marked_cells)
        else:
            break

# Save final error data
error_data = np.array(error_data)
np.savetxt('error_data.txt', error_data, header='cells residual high_order')

# Multi-rate plot
plt.figure()
plt.loglog(error_data[:,0], error_data[:,1], '-o', label='Residual')
plt.loglog(error_data[:,0], error_data[:,2], '-s', label='High Order')
plt.xlabel('Number of cells')
plt.ylabel('Error')
plt.legend()
plt.title('Multi-rate plot')
plt.savefig('multi_rate_plot.png')
plt.show()
```