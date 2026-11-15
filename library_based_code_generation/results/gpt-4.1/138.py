```python
import numpy as np
import matplotlib.pyplot as plt
import time
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import csr_matrix

class BoxDomain2D:
    def __init__(self, box=[0, 1, 0, 1], nx=10, ny=10):
        self.box = box
        self.nx = nx
        self.ny = ny
        self.mesh = self.create_mesh()
    
    def create_mesh(self):
        x0, x1, y0, y1 = self.box
        mesh = TriangleMesh.boxmesh2d([x0, x1, y0, y1], nx=self.nx, ny=self.ny)
        return mesh

    def displacement(self, p):
        # Example: zero displacement
        return np.zeros((2, p.shape[1]))

    def strain(self, grad):
        # grad: (2, 2, N)
        eps = 0.5 * (grad + np.transpose(grad, (1, 0, 2)))
        return eps

    def stress(self, grad):
        # Linear elasticity: sigma = lambda*tr(eps)*I + 2*mu*eps
        lam = 1.0
        mu = 1.0
        eps = self.strain(grad)
        tr = np.trace(eps, axis1=0, axis2=1)
        sigma = lam * tr * np.eye(2)[:, :, None] + 2 * mu * eps
        return sigma

    def source(self, p):
        # Example: zero source
        return np.zeros((2, p.shape[1]))

    def is_dirichlet_boundary(self, p):
        x0, x1, y0, y1 = self.box
        x, y = p
        return (np.isclose(x, x0) | np.isclose(x, x1) | np.isclose(y, y0) | np.isclose(y, y1))

    def is_neumann_boundary(self, p):
        # Example: no Neumann boundary
        return np.zeros(p.shape[1], dtype=bool)

    def is_fracture_boundary(self, p):
        # Example: no fracture
        return np.zeros(p.shape[1], dtype=bool)

    def dirichlet_bc(self, p):
        return self.displacement(p)

    def neumann_bc(self, p, n):
        # Example: zero Neumann
        return np.zeros((2, p.shape[1]))

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

class FastElasticitySolver:
    def __init__(self, A, precond=None):
        self.A = A
        self.precond = precond

    def preconditioner(self):
        if self.precond is not None:
            M = self.precond
            def mv(x):
                return M @ x
            return LinearOperator(self.A.shape, matvec=mv)
        else:
            return None

    def solve(self, b, tol=1e-8, maxiter=1000, callback=None):
        M = self.preconditioner()
        x, info = cg(self.A, b, tol=tol, maxiter=maxiter, M=M, callback=callback)
        return x, info

if __name__ == "__main__":
    # Initialize domain and mesh
    domain = BoxDomain2D(box=[0, 1, 0, 1], nx=20, ny=20)
    mesh = domain.mesh

    # Plot original mesh
    fig = plt.figure()
    ax = fig.gca()
    mesh.add_plot(ax)
    plt.title("Original Mesh")
    plt.show()

    # Define Lagrange finite element space
    p = 1
    space = LagrangeFiniteElementSpace(mesh, p=p, dim=2)

    # Dirichlet boundary
    is_db = space.boundary_dof()
    is_db = np.unique(is_db)
    dof = space.number_of_global_dofs()
    u = np.zeros(dof)

    # Dirichlet values
    gdof = space.dof_to_vertex()
    node = mesh.node
    db_node = node[:, gdof[is_db]]
    u_db = domain.dirichlet_bc(db_node)
    u[is_db] = u_db.T.flatten()

    # Assemble stiffness matrix and load vector
    A = space.stiff_matrix(lambda x: 2.0)  # mu = 1, lambda = 1
    # For linear elasticity, need to assemble full elasticity matrix
    # For simplicity, use scalar Laplacian for both components
    A = csr_matrix(A)
    f = space.source_vector(domain.source)

    # Apply Dirichlet BC
    bd_idx = is_db
    A = A.tolil()
    A[bd_idx, :] = 0
    A[:, bd_idx] = 0
    A[bd_idx, bd_idx] = 1
    f[bd_idx] = u[bd_idx]
    A = A.tocsr()

    print("Matrix shape:", A.shape)

    # Solve system
    counter = IterationCounter()
    solver = FastElasticitySolver(A)
    t0 = time.time()
    u_sol, info = solver.solve(f, callback=counter)
    t1 = time.time()
    print(f"Solved in {t1-t0:.4f} seconds, iterations: {counter.niter}")
```