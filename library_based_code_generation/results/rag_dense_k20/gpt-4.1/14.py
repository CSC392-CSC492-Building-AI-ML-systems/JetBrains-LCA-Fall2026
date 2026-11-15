```python
import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, constitutive, elements, functions, assembler, pyTACS

# MPI communicator
comm = MPI.COMM_WORLD

# Path to BDF file
bdf_file = 'mesh.bdf'

# Load mesh using pyTACS
FEAAssembler = pyTACS.pyTACS(bdf_file, comm)

# Constitutive properties
rho = 2500.0         # density kg/m^3
E = 70e9             # Young's modulus Pa
nu = 0.3             # Poisson's ratio
kcorr = 5.0 / 6.0    # shear correction factor
ys = 350e6           # yield stress Pa
thickness = 0.01     # thickness m

# Material and element property setup function
def elemCallBack(design, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=0)
    transform = elements.ShellRefAxisTransform()
    elem = elements.Quad4Shell(transform, con)
    return elem

# Set up elements
FEAAssembler.initialize(elemCallBack)

# Create TACS Assembler object
assembler_obj = FEAAssembler.assembler

# Create KS function
ks_func = functions.KSFailure(assembler_obj, safetyFactor=1.0, ksWeight=50.0)

# Get design variable values
xDV = assembler_obj.getDesignVars()

# Get node locations
Xpts = assembler_obj.createNodeVec()
assembler_obj.getNodes(Xpts)
Xpts_array = Xpts.getArray().copy()

# Create force vector (e.g., uniform load in z)
force = assembler_obj.createVec()
force_array = force.getArray()
force_array[:] = 0.0
force_array[2::3] = -1000.0  # Apply -1000 N in z to all nodes
force.setArray(force_array)

# Set up solution and residual vectors
res = assembler_obj.createVec()
u = assembler_obj.createVec()

# Assemble Jacobian and factor
assembler_obj.assembleJacobian(1.0, 0.0, 0.0, u, res)
mat = assembler_obj.getJacobian()
mat.factor()

# Solve linear system
assembler_obj.assembleRes(u, res)
res.scale(-1.0)
mat.solve(res, u)

# Evaluate function and solve adjoint
funcs = {}
ks_func.evalFunctions(u, funcs)
func_val = funcs[ks_func.getName()]

# Solve adjoint
adjoint = assembler_obj.createVec()
ks_func.evalSVSens(u, adjoint)
mat.solve(adjoint, adjoint)

# Compute total derivative w.r.t. design variables
dfunc_dXpt = assembler_obj.createNodeVec()
dfunc_ddv = np.zeros(assembler_obj.getNumDesignVars())
ks_func.evalXptSens(u, Xpts, dfunc_dXpt)
ks_func.evalDesignVarSens(u, dfunc_ddv)

# Create random direction for nodal perturbation
np.random.seed(42)
perturb_dir = np.random.randn(Xpts_array.size)
perturb_dir /= np.linalg.norm(perturb_dir)

# Compute total derivative w.r.t. nodal locations
proj_deriv = np.dot(dfunc_dXpt.getArray(), perturb_dir)

# Complex step for verification
h = 1e-30
Xpts_cs = Xpts_array.astype(np.complex128)
Xpts_cs += 1j * h * perturb_dir
Xpts_cs_vec = assembler_obj.createNodeVec()
Xpts_cs_vec.setArray(Xpts_cs)
assembler_obj.setNodes(Xpts_cs_vec)

# Re-solve with perturbed nodes
u_cs = assembler_obj.createVec()
assembler_obj.assembleJacobian(1.0, 0.0, 0.0, u_cs, res)
mat_cs = assembler_obj.getJacobian()
mat_cs.factor()
assembler_obj.assembleRes(u_cs, res)
res.scale(-1.0)
mat_cs.solve(res, u_cs)

# Evaluate function for perturbed solution
funcs_cs = {}
ks_func.evalFunctions(u_cs, funcs_cs)
func_val_cs = funcs_cs[ks_func.getName()]
proj_deriv_cs = np.imag(func_val_cs) / h

# Output results for visualization
if comm.rank == 0:
    print("Function value:", func_val)
    print("Projected derivative (adjoint):", proj_deriv)
    print("Projected derivative (complex step):", proj_deriv_cs)

# Write solution for visualization
assembler_obj.setNodes(Xpts)
assembler_obj.setVariables(u)
assembler_obj.writeSolution('tacs_solution')
```