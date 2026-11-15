```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.pde.time_harmonic_maxwell_2d import CosSinData
from fealpy.tools.show import showmultirate

def recover_curl(space, uh):
    '''
    Recover the curl of the solution uh at each node using least squares fitting.
    '''
    mesh = space.mesh
    node_num = mesh.number_of_nodes()
    curl_recovered = np.zeros(node_num)
    for node in range(node_num):
        A, b = least_squares_matrix(space, uh, node)
        if A.shape[0] > 0:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            curl_recovered[node] = x[0]
    return curl_recovered

def least_squares_matrix(space, uh, node):
    '''
    Build the least squares matrix for recovering the curl at a given node.
    '''
    mesh = space.mesh
    cells = mesh.ds.node2cell[node]
    A = []
    b = []
    for cell in cells:
        ldof = space.cell_dof[cell]
        cell_uh = uh[ldof]
        curl_uh = space.curl(cell, cell_uh)
        A.append([1.0])
        b.append(curl_uh)
    return np.array(A), np.array(b)

def remove_fourth_quadrant(mesh):
    '''
    Remove the fourth quadrant (x>0, y<0) from the mesh.
    '''
    node = mesh.node
    cell = mesh.cell
    cell_center = node[cell].mean(axis=1)
    mask = ~((cell_center[:, 0] > 0) & (cell_center[:, 1] < 0))
    mesh.from_mesh(MeshFactory().build_from_node_cell(node, cell[mask]))
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=1, help='Degree of Nedelec element')
    parser.add_argument('--h', type=float, default=0.1, help='Initial mesh size')
    parser.add_argument('--maxit', type=int, default=5, help='Maximum adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for marking')
    args = parser.parse_args()

    pde = CosSinData()
    mf = MeshFactory()
    mesh = mf.boxmesh2d(pde.domain(), nx=int(1/args.h), ny=int(1/args.h), meshtype='tri')
    mesh = remove_fourth_quadrant(mesh)

    errorMatrix = np.zeros((args.maxit, 3))
    Ndof = np.zeros(args.maxit, dtype=np.int_)

    for it in range(args.maxit):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=args.degree)
        Ndof[it] = space.number_of_global_dofs()
        uh = space.function()
        bc = DirichletBC(space, pde.dirichlet, threshold=1e-14)
        A = space.stiff_matrix(lambda x: pde.mu(x))
        M = space.mass_matrix(lambda x: pde.epsilon(x))
        F = space.source_vector(pde.source)
        A = A + 1j * pde.omega * M
        A, F = bc.apply(A, F, uh)
        uh[:] = spsolve(A, F)

        # L2 error of solution
        errorMatrix[it, 0] = space.integralalg.L2_error(pde.solution, uh)
        # L2 error of curl
        errorMatrix[it, 1] = space.integralalg.L2_error(pde.curl, uh, q=2, curl=True)
        # Recovery error
        curl_recovered = recover_curl(space, uh)
        node = mesh.node
        exact_curl = pde.curl(node)
        errorMatrix[it, 2] = np.sqrt(np.sum((curl_recovered - exact_curl)**2 * mesh.entity_measure('node')))

        if it < args.maxit - 1:
            # Mark cells for refinement based on recovery error
            cell = mesh.cell
            cell_center = mesh.node[cell].mean(axis=1)
            cell_curl = space.curl(cell, uh[space.cell_dof[cell]])
            cell_exact_curl = pde.curl(cell_center)
            cell_error = np.abs(cell_curl - cell_exact_curl)
            eta = cell_error / cell_error.max()
            marked = eta > (1 - args.theta)
            mesh.refine(marked)

    showmultirate(plt, 0, Ndof, errorMatrix, 
                  labels=['$||u-u_h||_{L^2}$', '$||curl(u)-curl(u_h)||_{L^2}$', '$||curl(u_h)-curl^*_h||_{L^2}$'])
    plt.show()

if __name__ == '__main__':
    main()
```