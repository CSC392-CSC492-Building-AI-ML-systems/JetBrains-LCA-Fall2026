```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.mesh import ATriMesher
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.tools import save_data
from fealpy.plot import showmultirate

# Parse command line arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <max_iter> <theta> <k>")
    sys.exit(1)

max_iter = int(sys.argv[1])
theta = float(sys.argv[2])
k = float(sys.argv[3])

# Initialize mesh (unit square as example)
domain = [0, 1, 0, 1]
n = 10
mesher = ATriMesher(domain, n, n)
mesh = mesher.mesh()

# Plot and save initial mesh
fig = plt.figure()
ax = fig.gca()
mesh.add_plot(ax)
plt.title("Initial Mesh")
plt.savefig("mesh_0.png")
plt.close(fig)

# Initialize function space
space = LagrangeFiniteElementSpace(mesh, p=1)

# Placeholder for error data
error_data = {
    'iteration': [],
    'residual': [],
    'high_order_term': [],
}

# Simplified friction problem setup (placeholders)
def solve_friction_problem(space, theta, k):
    # Placeholder for actual solver
    u = np.zeros(space.number_of_dofs())
    return u

def calculate_residual(u, space):
    # Placeholder for residual calculation
    return np.random.rand()

def calculate_high_order_term(u, space):
    # Placeholder for high order term calculation
    return np.random.rand()

for it in range(max_iter):
    # Solve the problem
    u = solve_friction_problem(space, theta, k)
    
    # Calculate residuals and high order terms
    residual = calculate_residual(u, space)
    high_order_term = calculate_high_order_term(u, space)
    
    # Save results and error data
    np.save(f"solution_{it}.npy", u)
    error_data['iteration'].append(it)
    error_data['residual'].append(residual)
    error_data['high_order_term'].append(high_order_term)
    save_data(f"error_data_{it}.npz", iteration=it, residual=residual, high_order_term=high_order_term)
    
    # Plot and save mesh
    fig = plt.figure()
    ax = fig.gca()
    mesh.add_plot(ax)
    plt.title(f"Mesh at Iteration {it}")
    plt.savefig(f"mesh_{it}.png")
    plt.close(fig)
    
    # Refine mesh if not last iteration
    if it < max_iter - 1:
        # Mark cells for refinement based on residuals (dummy marking)
        cell_markers = np.zeros(mesh.number_of_cells(), dtype=bool)
        cell_markers[:mesh.number_of_cells()//4] = True  # Dummy marking
        mesh.refine(cell_markers)
        space = LagrangeFiniteElementSpace(mesh, p=1)

# Save final error data
save_data("final_error_data.npz", **error_data)

# Multi-rate plot
showmultirate(
    [np.array(error_data['residual']), np.array(error_data['high_order_term'])],
    labels=['Residual', 'High Order Term'],
    xlabel='Iteration',
    ylabel='Error',
    title='Multi-rate Plot'
)
plt.savefig("multirate_plot.png")
plt.show()
```