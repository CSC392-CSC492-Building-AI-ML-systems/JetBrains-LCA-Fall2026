```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
from fealpy.decorator import cartesian
from fealpy.mesh.mesh_tools import find_cell_with_point
from fealpy.data.CosSinData import CosSinData

def recover_curl(space, uh):
    mesh = space.mesh
    NC = mesh.number_of_cells()
    curl_uh = np.zeros(NC)
    for i in range(NC):
        cell = mesh.entity('cell', i)
        bary = mesh.cell_barycenter(i)
        curl_uh[i] = space.curl(uh, bary.reshape(1, -1))[0]
    return curl_uh

def least_squares_matrix(space, uh):
    mesh = space.mesh
    NN = mesh.number_of_nodes()
    A = np.zeros((NN, NN))
    for i in range(NN):
        node = mesh.node[i]
        cells = mesh.ds.node2cell[i]
        for j in cells:
            bary = mesh.cell_barycenter(j)
            grad_phi = space.grad_basis(bary.reshape(1, -1))
            A[i, i] += np.sum(grad_phi**2)
    return A

def remove_fourth_quadrant(mesh):
    bary = mesh.cell_barycenter()
    mask = ~((bary[:, 0] > 0) & (bary[:, 1] < 0))
    mesh = mesh.submesh(mask)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=1, help='Degree of Nedelec element')
    parser.add_argument('--n', type=int, default=8, help='Initial mesh size')
    parser.add_argument('--maxit', type=int, default=5, help='Maximum adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for marking')
    args = parser.parse_args()

    p = args.degree
    n = args.n
    maxit = args.maxit
    theta = args.theta

    data = CosSinData()
    mf = MeshFactory()
    mesh = mf.boxmesh2d(data.domain(), nx=n, ny=n, meshtype='tri')
    mesh = remove_fourth_quadrant(mesh)

    errorL2 = []
    errorCurl = []
    errorRec = []
    ndof = []

    for it in range(maxit):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)
        uh = space.function()
        bc = DirichletBC(space, data.dirichlet, threshold=1e-14)
        A = space.stiff_matrix(c=data.c)
        M = space.mass_matrix()
        F = space.source_vector(data.source)
        A = A + M
        bc.apply(A, F, uh)
        uh[:] = spsolve(A, F)

        l2err = space.integralalg.L2_error(data.solution, uh)
        curl_uh = space.curl(uh)
        curl_exact = data.curl(mesh.cell_barycenter())
        curlerr = np.sqrt(np.sum((curl_uh - curl_exact)**2 * mesh.cell_area()))
        curl_rec = recover_curl(space, uh)
        recerr = np.sqrt(np.sum((curl_uh - curl_rec)**2 * mesh.cell_area()))

        errorL2.append(l2err)
        errorCurl.append(curlerr)
        errorRec.append(recerr)
        ndof.append(space.number_of_dofs())

        if it < maxit - 1:
            eta = np.abs(curl_uh - curl_rec)
            cell_mark = mesh.refine_marker(eta, theta=theta)
            mesh = mesh.refine(cell_mark)

    showmultirate(plt, 0, ndof, errorL2, errorCurl, errorRec, 
                  labels=['L2', 'Curl', 'Recovery'])
    plt.show()

if __name__ == '__main__':
    main()
```