```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.tools.show import showmultirate
from fealpy.decorator import cartesian
from fealpy.data.CosSinData import CosSinData

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive time-harmonic equation solver with Nedelec elements.")
    parser.add_argument('--degree', type=int, default=1, help='Degree of Nedelec element')
    parser.add_argument('--h', type=float, default=0.1, help='Initial mesh size')
    parser.add_argument('--maxit', type=int, default=5, help='Maximum number of adaptive iterations')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta parameter for adaptive refinement')
    return parser.parse_args()

def recover_curl(space, uh):
    '''
    Recover the curl of the solution uh using least squares fitting.
    '''
    mesh = space.mesh
    NC = mesh.number_of_cells()
    curl_recovered = np.zeros(NC)
    for ci in range(NC):
        cell = mesh.entity('cell', ci)
        bary = mesh.cell_barycenters()[ci]
        # Get local dofs and values
        ldof = space.cell_dof(ci)
        uh_local = uh[ldof]
        # Compute local curl at barycenter
        curl_val = space.curl(uh_local, bary.reshape(1, -1))
        curl_recovered[ci] = curl_val
    return curl_recovered

def least_squares_matrix(space, uh):
    '''
    Calculate the least squares matrix for each node of the mesh.
    '''
    mesh = space.mesh
    NV = mesh.number_of_vertices()
    ls_matrix = np.zeros((NV, 2, 2))
    for vi in range(NV):
        node = mesh.entity('node', vi)
        # Find cells sharing this node
        cells = mesh.ds.vertex_to_cell()[vi]
        A = np.zeros((len(cells), 2))
        b = np.zeros(len(cells))
        for i, ci in enumerate(cells):
            bary = mesh.cell_barycenters()[ci]
            ldof = space.cell_dof(ci)
            uh_local = uh[ldof]
            curl_val = space.curl(uh_local, bary.reshape(1, -1))
            A[i, :] = bary
            b[i] = curl_val
        # Least squares fit: curl = a*x + b*y + c
        if len(cells) >= 2:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            ls_matrix[vi, 0, :] = coeffs
    return ls_matrix

def remove_fourth_quadrant(mesh):
    '''
    Remove cells in the fourth quadrant (x>0, y<0).
    '''
    bary = mesh.cell_barycenters()
    mask = ~((bary[:, 0] > 0) & (bary[:, 1] < 0))
    mesh = mesh.submesh(mask)
    return mesh

def main():
    args = parse_args()
    degree = args.degree
    h = args.h
    maxit = args.maxit
    theta = args.theta

    # Problem data
    pde = CosSinData()
    mf = MeshFactory()
    mesh = mf.boxmesh2d(pde.domain(), nx=int(1/h), ny=int(1/h), meshtype='tri')
    mesh = remove_fourth_quadrant(mesh)

    errorL2 = []
    errorCurl = []
    errorRecovery = []
    ndof = []

    for it in range(maxit):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=degree)
        uh = space.function()
        A = space.stiff_matrix(curl=True)
        F = space.source_vector(pde.source)
        bc = DirichletBC(space, pde.dirichlet, threshold=1e-14)
        A, F = bc.apply(A, F, uh)

        uh[:] = spsolve(A, F)

        # L2 error
        l2err = space.integralalg.L2_error(pde.solution, uh)
        # Curl error
        curl_uh = space.curl(uh)
        curl_exact = pde.curl
        curlerr = space.integralalg.L2_error(curl_exact, curl_uh)
        # Recovery error
        curl_rec = recover_curl(space, uh)
        recoveryerr = np.linalg.norm(curl_uh - curl_rec) / np.linalg.norm(curl_uh)

        errorL2.append(l2err)
        errorCurl.append(curlerr)
        errorRecovery.append(recoveryerr)
        ndof.append(space.number_of_dofs())

        if it < maxit - 1:
            # Mark cells for refinement based on recovery error
            eta = np.abs(curl_uh - curl_rec)
            cell_markers = eta > theta * np.max(eta)
            mesh.refine(cell_markers)

    # Plot error rates
    showmultirate(plt, 0, ndof, errorL2, label='L2 Error')
    showmultirate(plt, 0, ndof, errorCurl, label='Curl Error')
    showmultirate(plt, 0, ndof, errorRecovery, label='Recovery Error')
    plt.xlabel('DOF')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
```