```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import test_polygon_mesh
from fealpy.fem import LagrangeFiniteElementSpaceOnEdgeMesh
from fealpy.solver import HighOrderLagrangeFEMFastSolver

# Parse command line arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <max_iter> <theta> <k>")
    sys.exit(1)

max_iter = int(sys.argv[1])
theta = float(sys.argv[2])
k = float(sys.argv[3])

# Initialize mesh (halfedge polygon mesh)
mesh = test_polygon_mesh()

# Initialize finite element space
space = LagrangeFiniteElementSpaceOnEdgeMesh(mesh, p=1)

# Initialize solver for simplified friction problem
solver = HighOrderLagrangeFEMFastSolver(space, theta=theta, k=k)

# Prepare to store error data
error_data = []

for it in range(max_iter):
    # Solve the problem
    uh = solver.solve()
    
    # Calculate residuals and high order terms
    residuals = solver.compute_residuals(uh)
    high_order_terms = solver.compute_high_order_terms(uh)
    
    # Save results and error data to file
    np.savez(f'results_iter_{it}.npz', uh=uh, residuals=residuals, high_order_terms=high_order_terms)
    error = np.linalg.norm(residuals)
    error_data.append(error)
    np.save(f'error_data_iter_{it}.npy', np.array(error_data))
    
    # Plot mesh and save as image
    fig, ax = plt.subplots()
    mesh.add_plot(ax)
    plt.title(f'Iteration {it}')
    plt.savefig(f'mesh_iter_{it}.png')
    plt.close(fig)
    
    # Refine mesh based on residuals if not last iteration
    if it < max_iter - 1:
        mesh = mesh.refine(residuals)
        space = LagrangeFiniteElementSpaceOnEdgeMesh(mesh, p=1)
        solver = HighOrderLagrangeFEMFastSolver(space, theta=theta, k=k)

# Save final error data
np.save('final_error_data.npy', np.array(error_data))

# Display multi-rate plot
plt.figure()
plt.plot(range(1, max_iter+1), error_data, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Multi-rate Error Plot')
plt.grid(True)
plt.savefig('multi_rate_error_plot.png')
plt.show()
```