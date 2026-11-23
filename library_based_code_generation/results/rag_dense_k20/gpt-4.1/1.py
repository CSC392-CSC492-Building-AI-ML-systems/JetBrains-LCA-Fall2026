```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace2d
from fealpy.pde.heat_conduction_2d import HeatConductionPDEData
from fealpy.fem import ParabolicMFEMSolver
from fealpy.tools import showmultirate

def main(
    nx=10, ny=10,           # Number of spatial divisions in x and y
    nt=20,                  # Number of temporal divisions
    theta=0.8,              # Adaptive refinement stop threshold
    refine_mark=0.2,        # Fraction of elements to refine
    coarsen_mark=0.05,      # Fraction of elements to coarsen
    T=0.1,                  # Final time
    plot_steps=[0, 5, 10, 15, 20],  # Time steps to plot solution
    mesh_plot_prefix='mesh_step',
    solution_plot_prefix='solution_step'
):
    # PDE model
    pde = HeatConductionPDEData()
    # Initial mesh
    domain = np.array([[0, 1], [0, 1]])
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    # Time step
    dt = T / nt
    # Finite element space
    space = LagrangeFiniteElementSpace2d(mesh, p=1)
    # Initial condition
    u0 = space.interpolation(pde.init_solution)
    # Dirichlet boundary condition
    is_dirichlet = mesh.entity_barycenter('edge')[:, 0] == 0
    def dirichlet_bc(p):
        return pde.dirichlet(p, 0)
    # Time stepping
    u = u0.copy()
    t = 0
    for step in range(1, nt+1):
        t += dt
        # Assemble system
        A = space.stiff_matrix()
        M = space.mass_matrix()
        F = space.source_vector(lambda p: pde.source(p, t))
        # Backward Euler
        b = M @ u + dt * F
        lhs = M + dt * A
        # Apply Dirichlet BC
        bc = space.dirichlet_bc(dirichlet_bc, t)
        u_new = space.linear_solve(lhs, b, bc)
        # Error estimation (recovery-based)
        grad_u = space.grad_recovery(u_new)
        error = np.sqrt(np.sum(space.integral(lambda p: np.sum((grad_u(p) - pde.gradient(p, t))**2, axis=1))))
        print(f"Time step {step}, t={t:.3f}, error={error:.6e}")
        # Plot solution at specified steps
        if step in plot_steps:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            space.plot_surface(ax, u_new)
            plt.title(f"Numerical solution at t={t:.3f}")
            plt.savefig(f"{solution_plot_prefix}_{step:03d}.png")
            plt.close(fig)
        # Adaptive refinement
        eta = space.recovery_estimate(u_new)
        max_eta = np.max(eta)
        marked = eta > theta * max_eta
        if np.any(marked):
            mesh.refine_triangle(marked)
            space = LagrangeFiniteElementSpace2d(mesh, p=1)
            u_new = space.interpolation(lambda p: u_new(space.find_cell(p)))
            # Save mesh plot
            fig, ax = plt.subplots()
            mesh.add_plot(ax)
            plt.title(f"Mesh after refinement at step {step}")
            plt.savefig(f"{mesh_plot_prefix}_refine_{step:03d}.png")
            plt.close(fig)
        # Adaptive coarsening
        coarsen_marked = eta < coarsen_mark * max_eta
        if np.any(coarsen_marked):
            mesh.coarsen_triangle(coarsen_marked)
            space = LagrangeFiniteElementSpace2d(mesh, p=1)
            u_new = space.interpolation(lambda p: u_new(space.find_cell(p)))
            # Save mesh plot
            fig, ax = plt.subplots()
            mesh.add_plot(ax)
            plt.title(f"Mesh after coarsening at step {step}")
            plt.savefig(f"{mesh_plot_prefix}_coarsen_{step:03d}.png")
            plt.close(fig)
        # Check error threshold
        if error < theta:
            print(f"Error below threshold {theta}, stopping refinement.")
            break
        u = u_new.copy()

if __name__ == '__main__':
    main(
        nx=10,
        ny=10,
        nt=20,
        theta=0.05,
        refine_mark=0.2,
        coarsen_mark=0.05,
        T=0.1,
        plot_steps=[0, 5, 10, 15, 20]
    )
```