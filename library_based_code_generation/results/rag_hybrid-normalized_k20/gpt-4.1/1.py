```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.heat_conduction_2d import HeatConductionPDEData
from fealpy.tools.show import showmultirate
from fealpy.mesh.mesh_tools import boundary_adaptive_refine, adaptive_coarsen
from fealpy.timeintegratoralg import UniformTimeLine

def main(
    nx=10, ny=10,           # spatial divisions
    nt=20,                  # temporal divisions
    theta=0.2,              # adaptive refinement stop threshold
    refine_mark=0.3,        # fraction of elements to refine
    coarsen_mark=0.1,       # fraction of elements to coarsen
    plot_steps=[0, 5, 10, 15, 20]  # time steps to plot solution
):
    # PDE model
    pde = HeatConductionPDEData()
    domain = np.array([[0, 1], [0, 1]])
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    space = LagrangeFiniteElementSpace(mesh, p=1)
    timeline = UniformTimeLine(0, pde.T, nt)
    u0 = space.interpolation(pde.init_solution)
    uh = u0.copy()
    dt = timeline.dt

    fig_dir = "mesh_plots"
    import os
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for t_idx, t in enumerate(timeline):
        print(f"Time step {t_idx}/{nt}, t={t:.4f}")

        # Assemble system
        A = space.stiff_matrix()
        M = space.mass_matrix()
        F = space.source_vector(lambda x: pde.source(x, t+dt))
        bc = space.dirichlet_bc(pde.dirichlet, threshold=1e-14)
        uh_old = uh.copy()

        # Backward Euler: (M + dt*A) u^{n+1} = M u^n + dt*F
        b = M @ uh + dt * F
        A_sys = M + dt * A
        space.set_dirichlet_bc(A_sys, b, pde.dirichlet, threshold=1e-14)
        uh[:] = np.linalg.solve(A_sys, b)

        # Error estimation (recovery-based)
        grad_uh = space.grad_recovery(uh)
        eta = space.recovery_estimate(uh, grad_uh)
        error = np.sqrt(np.sum(eta**2))
        print(f"  Recovery error estimate: {error:.6e}")

        # Plot mesh
        plt.figure()
        mesh.add_plot(plt)
        plt.title(f"Mesh at t={t:.3f}, step={t_idx}")
        plt.savefig(f"{fig_dir}/mesh_t{t_idx:03d}.png")
        plt.close()

        # Adaptive refinement
        while error > theta:
            marked = eta > np.quantile(eta, 1-refine_mark)
            mesh = boundary_adaptive_refine(mesh, marked)
            space = LagrangeFiniteElementSpace(mesh, p=1)
            uh = space.interpolation(lambda x: uh_old[space.find_dof_with_point(x)])
            grad_uh = space.grad_recovery(uh)
            eta = space.recovery_estimate(uh, grad_uh)
            error = np.sqrt(np.sum(eta**2))
            print(f"    Refined mesh, error: {error:.6e}")
            plt.figure()
            mesh.add_plot(plt)
            plt.title(f"Refined mesh at t={t:.3f}, step={t_idx}")
            plt.savefig(f"{fig_dir}/refined_mesh_t{t_idx:03d}.png")
            plt.close()

        # Plot solution at specified steps
        if t_idx in plot_steps:
            plt.figure()
            space.plot(uh, shading='gouraud')
            plt.title(f"Numerical solution at t={t:.3f}")
            plt.colorbar()
            plt.savefig(f"{fig_dir}/solution_t{t_idx:03d}.png")
            plt.close()

        # Adaptive coarsening
        marked_coarsen = eta < np.quantile(eta, coarsen_mark)
        mesh = adaptive_coarsen(mesh, marked_coarsen)
        space = LagrangeFiniteElementSpace(mesh, p=1)
        uh = space.interpolation(lambda x: uh[space.find_dof_with_point(x)])
        plt.figure()
        mesh.add_plot(plt)
        plt.title(f"Coarsened mesh at t={t:.3f}, step={t_idx}")
        plt.savefig(f"{fig_dir}/coarsened_mesh_t{t_idx:03d}.png")
        plt.close()

if __name__ == "__main__":
    main(
        nx=10,
        ny=10,
        nt=20,
        theta=0.05,
        refine_mark=0.3,
        coarsen_mark=0.1,
        plot_steps=[0, 5, 10, 15, 20]
    )
```