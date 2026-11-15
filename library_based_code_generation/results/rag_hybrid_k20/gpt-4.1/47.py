```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from SimPEG import maps, data, data_misfit, regularization, optimization, inverse_problem, inversion, directives
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import model_builder

# Mesh parameters
cs, ncx, ncy, ncz, npad = 5.0, 40, 40, 20, 10
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz)]
mesh = TensorMesh([hx, hy, hz], x0="CCC")

# Model parameters
background_conductivity = 1e-2  # S/m
sphere1_center = np.array([0, -20, -30])
sphere1_radius = 12
sphere1_conductivity = 1e-1  # conductive sphere

sphere2_center = np.array([0, 20, -30])
sphere2_radius = 12
sphere2_conductivity = 1e-3  # resistive sphere

# Build model
model = background_conductivity * np.ones(mesh.nC)
ind_sphere1 = model_builder.getIndicesSphere(sphere1_center, sphere1_radius, mesh.gridCC)
ind_sphere2 = model_builder.getIndicesSphere(sphere2_center, sphere2_radius, mesh.gridCC)
model[ind_sphere1] = sphere1_conductivity
model[ind_sphere2] = sphere2_conductivity

# Define core region for inversion
core_z = mesh.vectorCCz < 0
core_x = np.logical_and(mesh.vectorCCx > -60, mesh.vectorCCx < 60)
core_y = np.logical_and(mesh.vectorCCy > -60, mesh.vectorCCy < 60)
active = np.outer(np.outer(core_x, core_y), core_z).reshape(mesh.vnC, order="F").flatten()

# Survey parameters
xmin, xmax, ymin, ymax = -40, 40, -40, 40
n_electrodes = 21
x_electrodes = np.linspace(xmin, xmax, n_electrodes)
y_electrodes = np.linspace(ymin, ymax, n_electrodes)
z_electrodes = np.zeros_like(x_electrodes)

# Build dipole-dipole survey
source_list = []
a = 10  # dipole length
n_spacing = 8  # number of spacings

for i in range(n_electrodes - n_spacing - 1):
    for j in range(n_electrodes - n_spacing - 1):
        A = np.array([x_electrodes[i], y_electrodes[j], 0.0])
        B = np.array([x_electrodes[i + 1], y_electrodes[j], 0.0])
        receivers = []
        for n in range(1, n_spacing + 1):
            M = np.array([x_electrodes[i + n], y_electrodes[j], 0.0])
            N = np.array([x_electrodes[i + n + 1], y_electrodes[j], 0.0])
            receivers.append(dc.receivers.Dipole(M, N, data_type="volt"))
        if receivers:
            source_list.append(dc.sources.Dipole(receivers, A, B))

survey = dc.survey.Survey(source_list)

# Simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigma=model,
    solver=None,
)

# Generate synthetic data
synthetic_data = simulation.dpred(model)
std = 0.01 * np.abs(synthetic_data) + 1e-5
noise = std * np.random.randn(*synthetic_data.shape)
dobs = synthetic_data + noise

data_obj = data.Data(survey, dobs=dobs, standard_deviation=std)

# Mapping: active cells + expMap (log conductivity)
act_map = maps.InjectActiveCells(mesh, active, np.log(background_conductivity))
exp_map = maps.ExpMap(mesh)
mapping = exp_map * act_map

# Initial model
m0 = np.log(background_conductivity) * np.ones(active.sum())

# Data misfit
dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=dc.simulation.Simulation3DNodal(
    mesh, survey=survey, sigmaMap=mapping, solver=None
))

# Regularization
reg = regularization.Simple(mesh, indActive=active, mapping=maps.IdentityMap(nP=int(active.sum())))

# Optimization
opt = optimization.InexactGaussNewton(maxIter=15)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = directives.TargetMisfit()
save_every = directives.SaveEveryIteration()
update_Jacobi = directives.UpdatePreconditioner()

# Inversion
inv = inversion.BaseInversion(inv_prob, directiveList=[betaest, target, save_every, update_Jacobi])
recovered_model = inv.run(m0)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Vertical cross-section (Y=0)
y_ind = np.argmin(np.abs(mesh.vectorCCy))
model_3d = np.log10(model.reshape(mesh.vnC, order="F"))
recovered_3d = np.log10(mapping * recovered_model).reshape(mesh.vnC, order="F")

im0 = axs[0, 0].imshow(
    model_3d[:, y_ind, :].T,
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCz[0], mesh.vectorCCz[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
axs[0, 0].set_title("True Model (Vertical, Y=0)")
axs[0, 0].set_xlabel("X (m)")
axs[0, 0].set_ylabel("Z (m)")
plt.colorbar(im0, ax=axs[0, 0], label="log10 Conductivity (S/m)")

im1 = axs[0, 1].imshow(
    recovered_3d[:, y_ind, :].T,
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCz[0], mesh.vectorCCz[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
axs[0, 1].set_title("Inverted Model (Vertical, Y=0)")
axs[0, 1].set_xlabel("X (m)")
axs[0, 1].set_ylabel("Z (m)")
plt.colorbar(im1, ax=axs[0, 1], label="log10 Conductivity (S/m)")

# Horizontal slice (Z closest to -30)
z_ind = np.argmin(np.abs(mesh.vectorCCz + 30))
im2 = axs[1, 0].imshow(
    model_3d[:, :, z_ind].T,
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCy[0], mesh.vectorCCy[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
axs[1, 0].set_title("True Model (Horizontal, Z=-30m)")
axs[1, 0].set_xlabel("X (m)")
axs[1, 0].set_ylabel("Y (m)")
plt.colorbar(im2, ax=axs[1, 0], label="log10 Conductivity (S/m)")

im3 = axs[1, 1].imshow(
    recovered_3d[:, :, z_ind].T,
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCy[0], mesh.vectorCCy[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
axs[1, 1].set_title("Inverted Model (Horizontal, Z=-30m)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
plt.colorbar(im3, ax=axs[1, 1], label="log10 Conductivity (S/m)")

plt.tight_layout()
plt.show()
```