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
    mesh = space.mesh
    cell2dof = space.cell2dof
    ncell = mesh.number_of_cells()
    curl_recovered = np.zeros(ncell)
    for ci in range(ncell):
        dofs = cell2dof[ci]
        cell_basis_curl = space.curl_basis(ci)
        curl_val = 0.0
        for i, dof in enumerate(dofs):
            curl_val += uh[dof] * cell_basis_curl[i]
        curl_recovered[ci] = curl_val
    return curl_recovered

def least_squares_matrix(space, uh):
    mesh = space.mesh
    nnode = mesh.number_of_nodes()
    A = np.zeros((nnode, nnode))
    for ci in range(mesh.number_of_cells()):
        nodes = mesh.cell_to_node()[ci]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                A[nodes[i], nodes[j]] += 1
    return A

def remove_fourth_quadrant(mesh):
    cellcenters = mesh.cell_centers()
    mask = ~((cellcenters[:, 0] > 0) & (cellcenters[:, 1] < 0))
    mesh = mesh.submesh(mask)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=1, help='Degree of Nedelec element')
    parser.add_argument('--h', type=float, default=0.1, help='Initial mesh size')
    parser.add_argument('--maxit', type=int, default=5, help='Maximum adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for marking')
    args = parser.parse_args()

    p = args.degree
    h = args.h
    maxit = args.maxit
    theta = args.theta

    data = CosSinData()
    box = [0, 1, 0, 1]
    mf = MeshFactory()
    mesh = mf.boxmesh2d(box, nx=int(1/h), ny=int(1/h))
    mesh = remove_fourth_quadrant(mesh)

    errorL2 = []
    errorCurl = []
    errorRecovery = []
    ndof = []

    for it in range(maxit):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)
        bc = DirichletBC(space, data.dirichlet, threshold=1e-14)
        A = space.stiff_matrix(c=data.c, q=data.q)
        F = space.source_vector(data.source)
        A, F = bc.apply(A, F)
        uh = spsolve(A, F)
        ndof.append(space.number_of_dofs())

        l2err = space.integralalg.L2_error(lambda x: data.solution(x), uh, q=5)
        errorL2.append(l2err)

        curl_uh = space.curl_solution(uh)
        curl_exact = data.curl_solution(mesh.cell_centers())
        curlerr = np.sqrt(np.sum((curl_uh - curl_exact)**2 * mesh.cell_measures()))
        errorCurl.append(curlerr)

        curl_rec = recover_curl(space, uh)
        recerr = np.sqrt(np.sum((curl_uh - curl_rec)**2 * mesh.cell_measures()))
        errorRecovery.append(recerr)

        if it < maxit - 1:
            eta = np.abs(curl_uh - curl_rec)
            cell_markers = np.zeros(mesh.number_of_cells(), dtype=bool)
            sorted_eta = np.argsort(-eta)
            sum_eta2 = np.sum(eta**2)
            s = 0.0
            for idx in sorted_eta:
                cell_markers[idx] = True
                s += eta[idx]**2
                if s >= theta * sum_eta2:
                    break
            mesh = mesh.refine(cell_markers)

    showmultirate(plt, 0, ndof, errorL2, errorCurl, errorRecovery, 
                  labels=['L2', 'Curl', 'Recovery'])
    plt.show()

if __name__ == '__main__':
    main()
```