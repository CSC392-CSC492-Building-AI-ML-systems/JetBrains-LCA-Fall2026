```python
import numpy as np
import os
from mpi4py import MPI
from tacs import pyTACS, constitutive, elements, functions

# MPI communicator
comm = MPI.COMM_WORLD

# Path to BDF file
bdf_file = os.path.abspath('mesh.bdf')

# Load the mesh
FEAAssembler = pyTACS(bdf_file, comm=comm)

# Constitutive properties
rho = 2.7e3         # density, kg/m^3
E = 70e9            # elastic modulus, Pa
nu = 0.33           # poisson's ratio
kcorr = 5.0/6.0     # shear correction factor
ys = 350e6          # yield stress, Pa
thickness = 0.01    # thickness, m

# Loop over components and assign constitutive objects
def elemCallBack(design, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=0, kcorr=kcorr)
    transform = None
    elem = elements.Quad4Shell(transform, con)
    return elem

FEAAssembler.initialize(elemCallBack)

# Create TACS Assembler object
assembler = FEAAssembler.assembler

# Create KS function
ks_weight = 50.0
funcs = [functions.KSFailure(assembler, ksWeight=ks_weight)]

# Get design variable values
x = assembler.createDesignVec()
assembler.getDesignVars(x)

# Get node locations
Xpts = assembler.createNodeVec()
assembler.getNodes(Xpts)
Xpts_array = Xpts.getArray().copy()

# Create forces (e.g., uniform load in z-direction)
f = assembler.createVec()
f_array = f.getArray()
f_array[:] = 0.0
f_array[2::3] = -1000.0  # Apply -1000 N in z to each node
f.setArray(f_array)

# Set up and solve the analysis problem
res = assembler.createVec()
u = assembler.createVec()
assembler.assembleRes(u, f, res)
assembler.assembleJacobian(u, alpha=1.0, beta=0.0, gamma=0.0)
mat = assembler.getJacobianMatrix()
pc = assembler.getPreconditioner()
mat.factor()
mat.solve(res, u)

# Evaluate the function
func_vals = {}
assembler.evalFunctions(u, funcs, func_vals)
func_grad = {}
assembler.evalFunctionsSens(u, funcs, func_grad)

# Solve for adjoint variables
adjoint = assembler.createVec()
assembler.evalFunctionsSens(u, funcs, func_grad, adjoint=adjoint)

# Compute total derivative w.r.t. design variables and nodal locations
dfdx = assembler.createDesignVec()
assembler.evalDVSens(u, funcs, dfdx)
dfdX = assembler.createNodeVec()
assembler.evalXptSens(u, funcs, dfdX)

# Create random direction for nodal perturbation
np.random.seed(42)
perturb_dir = np.random.randn(*Xpts_array.shape)
perturb_dir /= np.linalg.norm(perturb_dir)

# Compute total derivative in this direction
proj_deriv = np.dot(dfdX.getArray(), perturb_dir)

# Set complex step and compute perturbed solution
h = 1e-30
Xpts_perturbed = Xpts_array.astype(np.complex128)
Xpts_perturbed += 1j * h * perturb_dir
Xpts_complex = assembler.createNodeVec()
Xpts_complex_array = Xpts_complex.getArray()
Xpts_complex_array[:] = Xpts_perturbed
Xpts_complex.setArray(Xpts_complex_array)

# Solve with perturbed nodes
assembler.setNodes(Xpts_complex)
u_complex = assembler.createVec()
assembler.assembleRes(u_complex, f, res)
assembler.assembleJacobian(u_complex, alpha=1.0, beta=0.0, gamma=0.0)
mat = assembler.getJacobianMatrix()
mat.factor()
mat.solve(res, u_complex)

# Evaluate function for perturbed solution
func_vals_complex = {}
assembler.evalFunctions(u_complex, funcs, func_vals_complex)
ks_complex = list(func_vals_complex.values())[0]
ks_real = list(func_vals.values())[0]
projected_deriv_cs = np.imag(ks_complex) / h

# Output results for visualization
if comm.rank == 0:
    print("KS function value (real):", ks_real)
    print("Projected derivative (adjoint):", proj_deriv)
    print("Projected derivative (complex step):", projected_deriv_cs)
    print("Relative error:", np.abs(proj_deriv - projected_deriv_cs) / np.abs(projected_deriv_cs))

# Write solution for visualization
assembler.setNodes(Xpts)
FEAAssembler.writeSolution(u, outputDir='output')
```