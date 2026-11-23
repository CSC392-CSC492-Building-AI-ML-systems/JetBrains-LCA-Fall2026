```python
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Placeholder imports for VEM and mesh handling
# from vem import VEMSpace, assemble_stiffness_matrix, assemble_rhs, apply_dirichlet_bc, solve_system, compute_error, mark_cells_for_refinement, refine_mesh
# from mesh import PolygonalMesh, plot_mesh

def poisson_rhs(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def dirichlet_bc(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def parse_args():
    parser = argparse.ArgumentParser(description="Solve Poisson equation with VEM on polygonal mesh.")
    parser.add_argument('--degree', type=int, default=1, help='Degree of VEM space')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of mesh refinement iterations')
    parser.add_argument('--adaptive_param', type=float, default=0.5, help='Adaptive refinement parameter')
    parser.add_argument('--dof_threshold', type=int, default=10000, help='Maximum number of degrees of freedom')
    return parser.parse_args()

def main():
    args = parse_args()
    degree = args.degree
    max_iter = args.max_iter
    adaptive_param = args.adaptive_param
    dof_threshold = args.dof_threshold

    # Initialize domain and mesh
    domain = [[0, 1], [0, 1]]
    mesh = PolygonalMesh(domain, initial_divisions=4)  # Placeholder

    error_matrix = []
    dof_list = []

    for it in range(max_iter):
        vem_space = VEMSpace(mesh, degree)
        u = np.zeros(vem_space.num_dofs)

        A = assemble_stiffness_matrix(vem_space)
        b = assemble_rhs(vem_space, poisson_rhs)

        apply_dirichlet_bc(A, b, vem_space, dirichlet_bc)

        u = solve_system(A, b)

        error = compute_error(vem_space, u, dirichlet_bc)
        error_matrix.append(error)
        dof_list.append(vem_space.num_dofs)

        print(f"Iteration {it+1}: DOFs = {vem_space.num_dofs}, Error = {error:.4e}")

        if vem_space.num_dofs > dof_threshold:
            print("DOF threshold reached. Stopping refinement.")
            break

        residual = compute_residual_estimate(vem_space, u, poisson_rhs)
        marked_cells = mark_cells_for_refinement(residual, adaptive_param)
        if not marked_cells:
            print("No cells marked for refinement. Stopping.")
            break
        mesh = refine_mesh(mesh, marked_cells)

    # Save results
    np.savetxt("dofs.txt", np.array(dof_list), fmt='%d')
    np.savetxt("error_matrix.txt", np.array(error_matrix))

    # Plot error
    plt.figure()
    plt.semilogy(dof_list, error_matrix, marker='o')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Error')
    plt.title('Error vs DOFs')
    plt.grid(True)
    plt.savefig("error_plot.png")
    plt.show()

    # Plot final mesh
    plt.figure()
    plot_mesh(mesh)
    plt.title('Final Mesh')
    plt.savefig("final_mesh.png")
    plt.show()

if __name__ == "__main__":
    main()
```