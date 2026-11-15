```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.tools import showmultirate
from fealpy.pde.heat_2d import HeatConductionData_2d
from fealpy.mesh.mesh_tools import find_node_with_value
import os

def main(
    nx=10, ny=10,  # spatial divisions
    nt=20,         # temporal divisions
    theta=0.2,     # adaptive refinement stop threshold
    refine_mark=0.3, # fraction of elements to refine
    coarsen_mark=0.05, # fraction of elements to coarsen
    T=0.1,         # final time
    plot_steps=[0, 5, 10, 15, 20], # time steps to plot solution
    mesh_dir='mesh_plots', # directory to save mesh plots
    sol_dir='solution_plots' # directory to save solution plots
):
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    if not os.path.exists(sol_dir):
        os.makedirs(sol_dir)

    pde = HeatConductionData_2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    timeline = UniformTimeLine(0, T, nt)
    dt = timeline.dt

    degree = 1
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    u0 = space.interpolation(pde.init_solution())

    bc = DirichletBC(space, pde.dirichlet, threshold=1e-14)

    A = space.stiff_matrix()
    M = space.mass_matrix()
    F = space.source_vector(pde.source, timeline[0])

    uh = u0.copy()
    t0 = timeline[0]

    for step, t1 in enumerate(timeline[1:], 1):
        F = space.source_vector(pde.source, t1)
        b = M @ uh + dt * F
        A_sys = M + dt * A
        bc.apply(A_sys, b, pde.dirichlet, t1)
        uh[:] = np.linalg.solve(A_sys, b)

        # Recovery-based error estimator
        grad_uh = space.grad_recovery(uh)
        eta = space.recovery_estimate(uh, grad_uh)
        error = np.sqrt(np.sum(eta**2))
        print(f"Time step {step}, t={t1:.4f}, error={error:.6e}")

        # Plot solution at specified steps
        if step in plot_steps:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            node = mesh.node
            tri = mesh.cell
            ax.plot_trisurf(node[:,0], node[:,1], uh, triangles=tri, cmap='viridis')
            ax.set_title(f"Numerical solution at t={t1:.4f}")
            plt.savefig(f"{sol_dir}/solution_t{step:03d}.png")
            plt.close(fig)

        # Adaptive refinement
        while error > theta:
            marked = space.mesh.refine_mark(eta, theta=refine_mark)
            mesh.refine_triangle_marked(marked)
            space = LagrangeFiniteElementSpace(mesh, p=degree)
            uh = space.interpolation(lambda x: uh[find_node_with_value(mesh, x)])
            bc = DirichletBC(space, pde.dirichlet, threshold=1e-14)
            grad_uh = space.grad_recovery(uh)
            eta = space.recovery_estimate(uh, grad_uh)
            error = np.sqrt(np.sum(eta**2))
            print(f"  Refinement: error={error:.6e}, mesh size={mesh.number_of_cells()}")
            fig = plt.figure()
            mesh.add_plot(fig.gca(), cellcolor='w')
            plt.title(f"Mesh after refinement at t={t1:.4f}")
            plt.savefig(f"{mesh_dir}/refine_t{step:03d}.png")
            plt.close(fig)

        # Adaptive coarsening
        marked = space.mesh.coarsen_mark(eta, theta=coarsen_mark)
        if np.any(marked):
            mesh.coarsen_triangle_marked(marked)
            space = LagrangeFiniteElementSpace(mesh, p=degree)
            uh = space.interpolation(lambda x: uh[find_node_with_value(mesh, x)])
            bc = DirichletBC(space, pde.dirichlet, threshold=1e-14)
            fig = plt.figure()
            mesh.add_plot(fig.gca(), cellcolor='w')
            plt.title(f"Mesh after coarsening at t={t1:.4f}")
            plt.savefig(f"{mesh_dir}/coarsen_t{step:03d}.png")
            plt.close(fig)

if __name__ == '__main__':
    main(
        nx=10,
        ny=10,
        nt=20,
        theta=0.01,
        refine_mark=0.3,
        coarsen_mark=0.05,
        T=0.1,
        plot_steps=[0, 5, 10, 15, 20],
        mesh_dir='mesh_plots',
        sol_dir='solution_plots'
    )
```