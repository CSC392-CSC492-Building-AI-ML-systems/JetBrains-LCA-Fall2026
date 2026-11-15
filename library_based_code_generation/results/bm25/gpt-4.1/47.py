```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from SimPEG import maps, data, inversion, inverse_problem, optimization, directives, regularization, data_misfit
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
sphere1_center = np.array([0, -20, -30])
sphere1_radius = 12.0
sphere1_conductivity = 1e-1  # conductive sphere

sphere2_center = np.array([15, 20, -25])
sphere2_radius = 10.0
sphere2_conductivity = 1e-3  # resistive sphere

# Active cells (below surface)
z0 = mesh.cell_centers[:, 2].min() + cs/2
active = mesh.cell_centers[:, 2] < 0

# Model definition
model = background_conductivity * np.ones(mesh.nC)
cc = mesh.cell_centers

sphere1 = np.linalg.norm(cc - sphere1_center, axis=1) < sphere1_radius
sphere2 = np.linalg.norm(cc - sphere2_center, axis=1) < sphere2_radius

model[sphere1] = sphere1_conductivity
model[sphere2] = sphere2_conductivity

# Mapping: active cells + expMap (log conductivity)
actMap = maps.InjectActiveCells(mesh, active, np.log(background_conductivity))
expMap = maps.ExpMap()
mapping = expMap * actMap

# Survey: Dipole-Dipole
xmin, xmax, ymin, ymax = -80, 80, -80, 80
n_electrodes = 21
x_electrodes = np.linspace(xmin, xmax, n_electrodes)
y_electrodes = np.zeros_like(x_electrodes)
z_electrodes = np.zeros_like(x_electrodes)
electrodes = np.c_[x_electrodes, y_electrodes, z_electrodes]

source_list = []
for i in range(n_electrodes - 3):
    for n in range(1, 4):  # n-spacing
        A = electrodes[i]
        B = electrodes[i + n]
        for m in range(1, 4):  # m-spacing
            M_idx = i + n + m
            N_idx = i + n + m + 1
            if N_idx < n_electrodes:
                M = electrodes[M_idx]
                N = electrodes[N_idx]
                rx = dc.receivers.Dipole(M, N)
                src = dc.sources.Dipole([rx], A, B)
                source_list.append(src)

survey = dc.survey.Survey(source_list)

# Simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh, survey=survey, sigmaMap=mapping, solver=None, 
    storeJ=True, 
    sigma=background_conductivity * np.ones(mesh.nC)
)

# Synthetic data
m_true = np.log(model[active])
d_true = simulation.dpred(m_true)
noise = 0.01 * np.abs(d_true) * np.random.randn(*d_true.shape)
d_obs = d_true + noise

data_obj = data.Data(survey, dobs=d_obs, standard_deviation=0.01 * np.abs(d_obs) + 1e-5)

# Inversion
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_obj)
reg = regularization.Simple(mesh, indActive=active, mapping=maps.IdentityMap(nP=int(active.sum())))
opt = optimization.InexactGaussNewton(maxIter=15)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
betaest = directives.BetaEstimate_ByEig()
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
inv = inversion.BaseInversion(inv_prob, directiveList=[betaest, target, save])

m0 = np.log(background_conductivity) * np.ones(active.sum())
recovered_model = inv.run(m0)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Vertical cross-section (Y=0)
y_section = np.argmin(np.abs(mesh.cell_centers[:, 1]))
ind_y = np.where(np.abs(mesh.cell_centers[:, 1]) < cs)[0]
x = mesh.cell_centers[ind_y, 0]
z = mesh.cell_centers[ind_y, 2]

# Ground truth vertical
im0 = axes[0, 0].tricontourf(
    x, z, np.log10(model[ind_y]), 20, cmap="viridis"
)
axes[0, 0].set_title("Ground Truth (Vertical, Y=0)")
axes[0, 0].set_xlabel("x (m)")
axes[0, 0].set_ylabel("z (m)")
fig.colorbar(im0, ax=axes[0, 0], label="log10 Conductivity (S/m)")

# Inverted vertical
inv_model_full = background_conductivity * np.ones(mesh.nC)
inv_model_full[active] = np.exp(recovered_model)
im1 = axes[0, 1].tricontourf(
    x, z, np.log10(inv_model_full[ind_y]), 20, cmap="viridis"
)
axes[0, 1].set_title("Inverted Model (Vertical, Y=0)")
axes[0, 1].set_xlabel("x (m)")
axes[0, 1].set_ylabel("z (m)")
fig.colorbar(im1, ax=axes[0, 1], label="log10 Conductivity (S/m)")

# Horizontal slice (z = -30 m)
z_slice = -30
ind_z = np.where(np.abs(mesh.cell_centers[:, 2] - z_slice) < cs)[0]
x = mesh.cell_centers[ind_z, 0]
y = mesh.cell_centers[ind_z, 1]

# Ground truth horizontal
im2 = axes[1, 0].tricontourf(
    x, y, np.log10(model[ind_z]), 20, cmap="viridis"
)
axes[1, 0].set_title(f"Ground Truth (Horizontal, z={z_slice} m)")
axes[1, 0].set_xlabel("x (m)")
axes[1, 0].set_ylabel("y (m)")
fig.colorbar(im2, ax=axes[1, 0], label="log10 Conductivity (S/m)")

# Inverted horizontal
im3 = axes[1, 1].tricontourf(
    x, y, np.log10(inv_model_full[ind_z]), 20, cmap="viridis"
)
axes[1, 1].set_title(f"Inverted Model (Horizontal, z={z_slice} m)")
axes[1, 1].set_xlabel("x (m)")
axes[1, 1].set_ylabel("y (m)")
fig.colorbar(im3, ax=axes[1, 1], label="log10 Conductivity (S/m)")

plt.tight_layout()
plt.show()
```