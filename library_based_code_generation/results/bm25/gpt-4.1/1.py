```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.model.heat_conduction_model_2d import HeatConductionModel2D
from fealpy.tools.show import showmultirate

def main(
    nx=10, ny=10,  # spatial divisions
    nt=20,         # temporal divisions
    theta=0.2,     # adaptive refinement stop threshold
    refine_mark=0.3, # fraction of elements to refine
    coarsen_mark=0.05, # fraction of elements to coarsen
    T=0.1,         # final time
    plot_steps=[0, 5, 10, 15, 20] # time steps to plot solution
):
    # Define the model
    model = HeatConductionModel2D()
    domain = model.domain()
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    timeline = UniformTimeLine(0, T, nt)
    dt = timeline.current_time_step_length()
    space = LagrangeFiniteElementSpace(mesh, p=1)
    u0 = space.interpolation(model.u0)
    u = u0.copy()
    bc = model.dirichlet_boundary_condition()
    error_list = []

    for k in range(nt):
        t1 = timeline.next_time_level()
        # Assemble system
        A = space.stiff_matrix()
        M = space.mass_matrix()
        F = space.source_vector(lambda x: model.f(x, t1))
        b = M @ u + dt * F
        AA = M + dt * A
        # Apply Dirichlet BC
        bc_dof = space.find_dof_with_dirichlet_boundary(bc)
        uD = space.interpolation(lambda x: model.g_D(x, t1))
        AA, b = space.set_dirichlet_bc(AA, b, bc_dof, uD)
        # Solve
        u = np.linalg.solve(AA, b)
        # Error estimation (recovery-based)
        eta = space.recovery_estimate(u)
        error = np.sqrt(np.sum(eta**2))
        error_list.append(error)
        print(f"Time step {k+1}/{nt}, t={t1:.4f}, error={error:.6e}")
        # Plot solution at specified steps
        if k in plot_steps:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            space.plot(u, ax=ax, cmap='rainbow')
            plt.title(f"Numerical solution at t={t1:.4f}")
            plt.savefig(f"solution_t{k:03d}.png")
            plt.close()
        # Adaptive refinement
        while error > theta:
            marked = space.mark(eta, theta=refine_mark)
            mesh = mesh.adaptive_refine(marked)
            plt.figure()
            mesh.add_plot(plt)
            plt.title(f"Refined mesh at t={t1:.4f}")
            plt.savefig(f"mesh_refine_t{k:03d}.png")
            plt.close()
            space = LagrangeFiniteElementSpace(mesh, p=1)
            u = space.interpolation(lambda x: model.g_D(x, t1))
            A = space.stiff_matrix()
            M = space.mass_matrix()
            F = space.source_vector(lambda x: model.f(x, t1))
            b = M @ u + dt * F
            AA = M + dt * A
            bc_dof = space.find_dof_with_dirichlet_boundary(bc)
            uD = space.interpolation(lambda x: model.g_D(x, t1))
            AA, b = space.set_dirichlet_bc(AA, b, bc_dof, uD)
            u = np.linalg.solve(AA, b)
            eta = space.recovery_estimate(u)
            error = np.sqrt(np.sum(eta**2))
            print(f"  Refined: error={error:.6e}")
        # Adaptive coarsening
        marked_coarse = space.mark(eta, theta=coarsen_mark, method='coarsen')
        if np.any(marked_coarse):
            mesh = mesh.adaptive_coarsen(marked_coarse)
            plt.figure()
            mesh.add_plot(plt)
            plt.title(f"Coarsened mesh at t={t1:.4f}")
            plt.savefig(f"mesh_coarsen_t{k:03d}.png")
            plt.close()
            space = LagrangeFiniteElementSpace(mesh, p=1)
            u = space.interpolation(lambda x: model.g_D(x, t1))
    # Plot error history
    plt.figure()
    plt.semilogy(range(1, nt+1), error_list, marker='o')
    plt.xlabel('Time step')
    plt.ylabel('Recovery error')
    plt.title('Error vs Time step')
    plt.savefig('error_history.png')
    plt.close()

if __name__ == '__main__':
    main()
```