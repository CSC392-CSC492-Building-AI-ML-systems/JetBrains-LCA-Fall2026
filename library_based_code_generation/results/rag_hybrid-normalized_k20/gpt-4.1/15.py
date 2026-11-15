```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
from fealpy.timeharmonic_model_2d import CosSinData

def recover_curl(space, uh):
    '''
    Recover the curl of the solution uh at each node using least squares fitting.
    '''
    mesh = space.mesh
    node_num = mesh.number_of_nodes()
    curl_recovered = np.zeros(node_num)
    for i in range(node_num):
        A, b = least_squares_matrix(space, uh, i)
        if A.size > 0:
            curl_recovered[i] = np.linalg.lstsq(A, b, rcond=None)[0][0]
    return curl_recovered

def least_squares_matrix(space, uh, node_idx):
    '''
    Build the least squares matrix for the node with index node_idx.
    '''
    mesh = space.mesh
    cells = mesh.ds.node2cell[node_idx]
    A = []
    b = []
    for ci in cells:
        cell = mesh.entity('cell', ci)
        bary = mesh.bc[ci]
        curl_uh = space.curl(uh, cell=ci, barycentric=bary)
        A.append([1.0])
        b.append(curl_uh)
    return np.array(A), np.array(b)

def remove_fourth_quadrant(mesh):
    '''
    Remove the fourth quadrant (x>0, y<0) from the mesh.
    '''
    node = mesh.node
    cell = mesh.cell
    bary = node[cell].mean(axis=1)
    mask = ~((bary[:, 0] > 0) & (bary[:, 1] < 0))
    mesh.from_mesh(MeshFactory().to_mesh(node, cell[mask]))
    return mesh

def main():
    parser = argparse.ArgumentParser(description='Adaptive time-harmonic equation solver with Nedelec elements.')
    parser.add_argument('--degree', type=int, default=1, help='Degree of Nedelec element')
    parser.add_argument('--h', type=float, default=0.2, help='Initial mesh size')
    parser.add_argument('--maxit', type=int, default=5, help='Maximum number of adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for marking')
    args = parser.parse_args()

    p = args.degree
    h = args.h
    maxit = args.maxit
    theta = args.theta

    data = CosSinData()
    mf = MeshFactory()
    mesh = mf.boxmesh2d(data.domain(), nx=int(1/h), ny=int(1/h), meshtype='tri')
    mesh = remove_fourth_quadrant(mesh)

    errorMatrix = np.zeros((maxit, 3))
    Ndof = np.zeros(maxit, dtype=int)

    for it in range(maxit):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)
        uh = space.function()
        A = space.stiff_matrix(c=data.c())
        F = space.source_vector(data.source)
        bc = DirichletBC(space, data.dirichlet, threshold=1e-14)
        A, F = bc.apply(A, F, uh)
        uh[:] = spsolve(A, F)

        Ndof[it] = space.number_of_dofs()
        # L2 error of solution
        errorMatrix[it, 0] = space.integralalg.L2_error(data.solution, uh)
        # L2 error of curl
        errorMatrix[it, 1] = space.integralalg.L2_error(data.curl_solution, lambda x: space.curl(uh, x))
        # Recovery error
        curl_recovered = recover_curl(space, uh)
        node = mesh.node
        exact_curl = data.curl_solution(node)
        errorMatrix[it, 2] = np.sqrt(np.sum((curl_recovered - exact_curl)**2 * mesh.entity_measure('node')))

        if it < maxit - 1:
            # Mark cells for refinement based on recovery error
            cell = mesh.cell
            bary = mesh.bc
            curl_uh = space.curl(uh, barycentric=bary)
            curl_exact = data.curl_solution(mesh.bc_to_point(bary))
            eta = np.abs(curl_uh - curl_exact)
            idx = np.argsort(-eta)
            total = np.sum(eta)
            s = 0.0
            marked = []
            for i in idx:
                marked.append(i)
                s += eta[i]
                if s >= theta * total:
                    break
            mesh.refine_cells(np.array(marked))

    showmultirate(plt, 0, Ndof, errorMatrix, 
                  labels=['$||u-u_h||_{L^2}$', '$||curl(u)-curl(u_h)||_{L^2}$', '$||curl(u_h)-curl^*_h||_{L^2}$'])
    plt.show()

if __name__ == '__main__':
    main()
```