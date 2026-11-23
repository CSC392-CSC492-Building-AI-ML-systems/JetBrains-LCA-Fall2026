```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.heat_conduction_2d import HeatConductionPDEData
from fealpy.tools.show import showmultirate
from fealpy.mesh.mesh_tools import boundary_adaptive_refine, adaptive_coarsen
from fealpy.tools.plot import plot_scalar_on_mesh

def main(
    nx=10, ny=10,          # Number of spatial divisions in x and y
    nt=20,                 # Number of temporal divisions
    theta=0.2,             # Adaptive refinement stop threshold
    refine_mark=0.3,       # Fraction of elements to refine
    coarsen_mark=0.05,     # Fraction of elements to coarsen
    T=0.1,                 # Final time
    plot_steps=[0, 5, 10, 15, 20]  # Time steps to plot solution
):
    # PDE model
    pde = HeatConductionPDEData()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    space = LagrangeFiniteElementSpace(mesh, p=1)
    dt = T / nt
    u0 = space.interpolation(pde.init_solution)
    u = u0.copy()
    t = 0.0

    # Dirichlet boundary condition
    is_dirichlet = mesh.entity_barycenter('edge')
    is_dirichlet = pde.is_dirichlet_boundary(is_dirichlet)
    dirichlet_dofs = space.find_dof_on_boundary(is_dirichlet)

    # Time stepping
    for n in range(1, nt+1):
        t += dt

        # Assemble system
        A = space.stiff_matrix()
        M = space.mass_matrix()
        F = space.source_vector(lambda x: pde.source(x, t))
        b = M @ u + dt * F
        AA = M + dt * A

        # Apply Dirichlet BC
        bc = pde.dirichlet(lambda x: pde.dirichlet(x, t))
        space.set_dirichlet_bc(AA, b, dirichlet_dofs, bc)

        # Solve
        u = np.linalg.solve(AA, b)

        # Error estimation (recovery-based)
        uh = space.function(u)
        grad_uh = space.grad_recovery(u)
        error = space.integralalg.L2_error(pde.gradient, grad_uh, t)
        print(f"Time step {n}, t={t:.3f}, recovery error={error:.6e}")

        # Adaptive refinement
        while error > theta:
            eta = space.recovery_estimate(u)
            marked = mesh.refine_marker(eta, refine_mark)
            mesh, relation = boundary_adaptive_refine(mesh, marked)
            space = LagrangeFiniteElementSpace(mesh, p=1)
            u = space.interpolation(uh)
            uh = space.function(u)
            grad_uh = space.grad_recovery(u)
            error = space.integralalg.L2_error(pde.gradient, grad_uh, t)
            # Save mesh plot
            plt.figure()
            mesh.add_plot(plt)
            plt.title(f"Refined mesh at t={t:.3f}")
            plt.savefig(f"mesh_refine_t{n}_step{t:.3f}.png")
            plt.close()
            print(f"  Refined mesh, recovery error={error:.6e}")

        # Adaptive coarsening
        eta = space.recovery_estimate(u)
        marked = mesh.coarsen_marker(eta, coarsen_mark)
        mesh, relation = adaptive_coarsen(mesh, marked)
        space = LagrangeFiniteElementSpace(mesh, p=1)
        u = space.interpolation(uh)
        # Save mesh plot
        plt.figure()
        mesh.add_plot(plt)
        plt.title(f"Coarsened mesh at t={t:.3f}")
        plt.savefig(f"mesh_coarsen_t{n}_step{t:.3f}.png")
        plt.close()

        # Plot solution at specified steps
        if n in plot_steps:
            plt.figure()
            plot_scalar_on_mesh(mesh, u, shading='gouraud')
            plt.title(f"Numerical solution at t={t:.3f}")
            plt.colorbar()
            plt.savefig(f"solution_t{n}_step{t:.3f}.png")
            plt.close()

if __name__ == '__main__':
    main(
        nx=10,
        ny=10,
        nt=20,
        theta=0.01,
        refine_mark=0.3,
        coarsen_mark=0.05,
        T=0.1,
        plot_steps=[0, 5, 10, 15, 20]
    )
```