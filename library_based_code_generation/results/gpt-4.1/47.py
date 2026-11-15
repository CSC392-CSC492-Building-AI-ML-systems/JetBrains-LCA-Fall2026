```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from SimPEG import maps, data, data_misfit, regularization, optimization, inverse_problem, inversion, directives
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import mkvc

# Mesh parameters
cs, ncx, ncy, ncz = 5.0, 40, 40, 20
npad = 8
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz)]
mesh = TensorMesh([hx, hy, hz], x0="CCC")

# Model parameters
background_conductivity = 1e-2  # S/m
sphere1_center = np.array([0, -30, -40])
sphere1_radius = 12
sphere1_conductivity = 1e-1  # conductive sphere

sphere2_center = np.array([0, 30, -40])
sphere2_radius = 12
sphere2_conductivity = 1e-3  # resistive sphere

# Model definition
model = background_conductivity * np.ones(mesh.nC)
xyz = mesh.gridCC

r1 = np.linalg.norm(xyz - sphere1_center, axis=1)
r2 = np.linalg.norm(xyz - sphere2_center, axis=1)
model[r1 < sphere1_radius] = sphere1_conductivity
model[r2 < sphere2_radius] = sphere2_conductivity

# Define core region for inversion
core_inds = (
    (np.abs(xyz[:, 0]) < 80)
    & (np.abs(xyz[:, 1]) < 80)
    & (xyz[:, 2] < -5)
    & (xyz[:, 2] > -95)
)
active = core_inds

# Survey parameters
n_electrodes = 24
xmin, xmax = -60, 60
ymin, ymax = -60, 60
z_electrodes = 0.0

x_electrodes = np.linspace(xmin, xmax, n_electrodes)
y_electrodes = np.linspace(ymin, ymax, n_electrodes)
electrodes = np.c_[x_electrodes, np.zeros_like(x_electrodes), np.ones_like(x_electrodes) * z_electrodes]

source_list = []
for i in range(n_electrodes - 3):
    A = electrodes[i]
    B = electrodes[i + 1]
    for j in range(i + 2, n_electrodes - 1):
        M = electrodes[j]
        N = electrodes[j + 1]
        src = dc.sources.Dipole(
            [dc.receivers.Dipole(M, N)], A, B
        )
        source_list.append(src)

survey = dc.survey.Survey(source_list)

# Simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigma=model,
    solver=dc.utils.SolverLU,
)

# Synthetic data
synthetic_data = simulation.dpred(model)
std = 0.03
noise = std * np.abs(synthetic_data) * np.random.randn(synthetic_data.size)
dobs = synthetic_data + noise

data_obj = data.Data(survey, dobs=dobs, standard_deviation=std * np.abs(dobs) + 1e-5)

# Mappings
expmap = maps.ExpMap()
actmap = maps.InjectActiveCells(mesh, active, np.log(background_conductivity))
mapping = expmap * actmap

# Initial model
m0 = np.log(background_conductivity) * np.ones(active.sum())

# Inversion setup
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigma=mapping,
    solver=dc.utils.SolverLU,
)

dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)
reg = regularization.Simple(mesh, indActive=active)
opt = optimization.InexactGaussNewton(maxIter=15)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
inv = inversion.BaseInversion(inv_prob, directiveList=[betaest, target, save])

# Run inversion
recovered_model = inv.run(m0)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Vertical cross-section (Y=0)
ysec = np.argmin(np.abs(mesh.vectorCCy))
model_3d = np.ones(mesh.nC) * np.nan
model_3d[active] = np.exp(recovered_model)
truth_3d = model

im0 = mesh.plot_slice(
    truth_3d, normal="Y", ind=ysec, ax=axes[0, 0], grid=True, clim=(1e-3, 1e-1), pcolorOpts={"cmap": "viridis"}
)
axes[0, 0].set_title("True Model (Vertical Y=0)")
im1 = mesh.plot_slice(
    model_3d, normal="Y", ind=ysec, ax=axes[0, 1], grid=True, clim=(1e-3, 1e-1), pcolorOpts={"cmap": "viridis"}
)
axes[0, 1].set_title("Inverted Model (Vertical Y=0)")

# Horizontal slice (Z=-40)
zsec = np.argmin(np.abs(mesh.vectorCCz + 40))
im2 = mesh.plot_slice(
    truth_3d, normal="Z", ind=zsec, ax=axes[1, 0], grid=True, clim=(1e-3, 1e-1), pcolorOpts={"cmap": "viridis"}
)
axes[1, 0].set_title("True Model (Horizontal Z=-40)")
im3 = mesh.plot_slice(
    model_3d, normal="Z", ind=zsec, ax=axes[1, 1], grid=True, clim=(1e-3, 1e-1), pcolorOpts={"cmap": "viridis"}
)
axes[1, 1].set_title("Inverted Model (Horizontal Z=-40)")

for ax in axes.flatten():
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```