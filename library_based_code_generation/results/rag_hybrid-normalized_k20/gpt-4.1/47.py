```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from SimPEG import maps, data, data_misfit, regularization, optimization, inverse_problem, inversion, directives
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import mkvc

# Mesh parameters
cs, ncx, ncy, ncz, npad = 5.0, 40, 40, 20, 10
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz)]
mesh = TensorMesh([hx, hy, hz], x0="CCC")

# Model parameters
background_conductivity = 1e-2  # S/m
sphere1_center = np.array([0, -30, -30])
sphere1_radius = 15
sphere1_conductivity = 1e-1  # conductive sphere

sphere2_center = np.array([0, 30, -30])
sphere2_radius = 15
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
    (np.abs(xyz[:, 0]) < 60)
    & (np.abs(xyz[:, 1]) < 60)
    & (xyz[:, 2] < 0)
    & (xyz[:, 2] > -60)
)
active = core_inds

# Mapping: log conductivity in core, background elsewhere
expmap = maps.ExpMap()
actmap = maps.InjectActiveCells(mesh, active, np.log(background_conductivity))
mapping = expmap * actmap

# Survey geometry: Dipole-Dipole
n_electrodes = 24
xmin, xmax = -60, 60
y_electrodes = np.linspace(xmin, xmax, n_electrodes)
electrode_locations = np.c_[np.zeros(n_electrodes), y_electrodes, np.zeros(n_electrodes)]

source_list = []
for i in range(n_electrodes - 3):
    A = electrode_locations[i]
    B = electrode_locations[i + 1]
    for n in range(1, 4):
        M_idx = i + 1 + n
        N_idx = i + 2 + n
        if N_idx < n_electrodes:
            M = electrode_locations[M_idx]
            N = electrode_locations[N_idx]
            rx = dc.receivers.Dipole(M, N)
            src = dc.sources.Dipole([rx], A, B)
            source_list.append(src)

survey = dc.survey.Survey(source_list)

# Simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigma=background_conductivity * np.ones(mesh.nC),
    solver=dc.utils.SolverLU,
)

# Generate synthetic data
sigma_true = model
data_true = simulation.dpred(sigma_true)
std = 0.01 * np.abs(data_true) + 1e-5
noise = std * np.random.randn(data_true.size)
data_obs = data_true + noise

data_obj = data.Data(survey, dobs=data_obs, standard_deviation=std)

# Inversion setup
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigmaMap=mapping,
    solver=dc.utils.SolverLU,
)

dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_obj)
reg = regularization.Simple(mesh, indActive=active)
opt = optimization.InexactGaussNewton(maxIter=15)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

betaest = directives.BetaEstimate_ByEig()
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
update_IRLS = directives.Update_IRLS()
directives_list = [betaest, target, save]

inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Initial model
m0 = np.log(background_conductivity) * np.ones(active.sum())

# Run inversion
recovered_model = inv.run(m0)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Vertical cross-section (Y=0)
ind_y0 = np.argmin(np.abs(mesh.vectorCCy))
slice_y0 = mesh.gridCC[:, 1] == mesh.vectorCCy[ind_y0]

# Ground truth vertical
im0 = axes[0, 0].tricontourf(
    mesh.gridCC[slice_y0, 0],
    mesh.gridCC[slice_y0, 2],
    np.log10(model[slice_y0]),
    50,
    cmap="viridis",
)
axes[0, 0].set_title("Ground Truth (Vertical Y=0)")
axes[0, 0].set_xlabel("X (m)")
axes[0, 0].set_ylabel("Z (m)")
fig.colorbar(im0, ax=axes[0, 0], label="log10 Conductivity (S/m)")

# Inverted vertical
recovered_full = background_conductivity * np.ones(mesh.nC)
recovered_full[active] = np.exp(recovered_model)
im1 = axes[0, 1].tricontourf(
    mesh.gridCC[slice_y0, 0],
    mesh.gridCC[slice_y0, 2],
    np.log10(recovered_full[slice_y0]),
    50,
    cmap="viridis",
)
axes[0, 1].set_title("Inverted Model (Vertical Y=0)")
axes[0, 1].set_xlabel("X (m)")
axes[0, 1].set_ylabel("Z (m)")
fig.colorbar(im1, ax=axes[0, 1], label="log10 Conductivity (S/m)")

# Horizontal slice (Z=-30)
ind_z = np.argmin(np.abs(mesh.vectorCCz + 30))
slice_z = mesh.gridCC[:, 2] == mesh.vectorCCz[ind_z]

# Ground truth horizontal
im2 = axes[1, 0].tricontourf(
    mesh.gridCC[slice_z, 0],
    mesh.gridCC[slice_z, 1],
    np.log10(model[slice_z]),
    50,
    cmap="viridis",
)
axes[1, 0].set_title("Ground Truth (Horizontal Z=-30m)")
axes[1, 0].set_xlabel("X (m)")
axes[1, 0].set_ylabel("Y (m)")
fig.colorbar(im2, ax=axes[1, 0], label="log10 Conductivity (S/m)")

# Inverted horizontal
im3 = axes[1, 1].tricontourf(
    mesh.gridCC[slice_z, 0],
    mesh.gridCC[slice_z, 1],
    np.log10(recovered_full[slice_z]),
    50,
    cmap="viridis",
)
axes[1, 1].set_title("Inverted Model (Horizontal Z=-30m)")
axes[1, 1].set_xlabel("X (m)")
axes[1, 1].set_ylabel("Y (m)")
fig.colorbar(im3, ax=axes[1, 1], label="log10 Conductivity (S/m)")

plt.tight_layout()
plt.show()
```