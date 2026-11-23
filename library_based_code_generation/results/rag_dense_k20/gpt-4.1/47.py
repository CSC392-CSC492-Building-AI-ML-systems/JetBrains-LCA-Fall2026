```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from SimPEG import maps, utils
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.data import Data
from SimPEG.inversion import Inversion
from SimPEG.inverse_problem import InverseProblem
from SimPEG.optimization import InexactGaussNewton
from SimPEG.directives import (
    BetaEstimate_ByEig, BetaSchedule, Update_IRLS, UpdatePreconditioner
)
from SimPEG.regularization import Tikhonov
from SimPEG.data_misfit import L2DataMisfit

# Mesh parameters
cs = 5.0
ncx, ncy, ncz = 40, 40, 20
npad = 8
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz)]
mesh = TensorMesh([hx, hy, hz], x0="CCC")

# Model parameters
background_conductivity = 1e-2  # S/m
sphere1_center = np.array([0, -20, -30])
sphere1_radius = 12.0
sphere1_conductivity = 1e-1  # conductive sphere

sphere2_center = np.array([20, 20, -25])
sphere2_radius = 10.0
sphere2_conductivity = 1e-3  # resistive sphere

# Model definition
xyz = mesh.gridCC
model = background_conductivity * np.ones(mesh.nC)

r1 = np.linalg.norm(xyz - sphere1_center, axis=1)
r2 = np.linalg.norm(xyz - sphere2_center, axis=1)
model[r1 < sphere1_radius] = sphere1_conductivity
model[r2 < sphere2_radius] = sphere2_conductivity

# Define active cells (below surface)
actv = mesh.gridCC[:, 2] < 0
n_active = int(actv.sum())

# Mapping: active cells + expMap (log conductivity)
actvMap = maps.InjectActiveCells(mesh, actv, np.log(background_conductivity))
expMap = maps.ExpMap()
mapping = expMap * actvMap

# Survey geometry (Dipole-Dipole)
xmin, xmax = -80, 80
ymin, ymax = -80, 80
n_electrodes = 21
x_electrodes = np.linspace(xmin, xmax, n_electrodes)
y_electrodes = np.zeros_like(x_electrodes)
z_electrodes = np.zeros_like(x_electrodes)

electrodes = np.c_[x_electrodes, y_electrodes, z_electrodes]

source_list = []
a = 10  # dipole spacing
n_spacing = 6  # number of spacings for receiver dipole

for i in range(n_electrodes - 3 - n_spacing):
    A = electrodes[i]
    B = electrodes[i + 1]
    for n in range(1, n_spacing + 1):
        M = electrodes[i + 1 + n]
        N = electrodes[i + 2 + n]
        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], A, B)
        source_list.append(src)

survey = dc.survey.Survey(source_list)

# Simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh,
    survey=survey,
    sigmaMap=mapping,
    solver=utils.SolverLU,
)

# Synthetic data
np.random.seed(42)
sigma_true = np.log(model[actv])
data = simulation.dpred(sigma_true)
noise_level = 0.01  # 1% noise
noise = noise_level * np.abs(data) * np.random.randn(data.size)
data_noisy = data + noise
dobs = Data(survey, dobs=data_noisy, standard_deviation=np.abs(noise))

# Inversion setup
m0 = np.log(background_conductivity) * np.ones(n_active)
reg = Tikhonov(mesh, mapping=maps.IdentityMap(nP=n_active), active_cells=actv)
dmis = L2DataMisfit(simulation=simulation, data=dobs)
opt = InexactGaussNewton(maxIter=15)
inv_prob = InverseProblem(dmis, reg, opt)
betaest = BetaEstimate_ByEig()
beta_schedule = BetaSchedule(coolingFactor=2, coolingRate=2)
update_irls = Update_IRLS(f_min_change=1e-4, max_irls_iterations=10)
update_Jacobi = UpdatePreconditioner()
directives = [betaest, beta_schedule, update_irls, update_Jacobi]

inv = Inversion(inv_prob, directives=directives)
recovered_model = inv.run(m0)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Vertical cross-section (Y=0)
ind_y = np.argmin(np.abs(mesh.vectorCCy))
slice_true = model.reshape(mesh.vnC)[:, ind_y, :]
slice_rec = np.exp(mapping * recovered_model).reshape(mesh.vnC)[:, ind_y, :]

im0 = axes[0, 0].imshow(
    np.log10(slice_true.T),
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCz[-1], mesh.vectorCCz[0]],
    aspect="auto",
    cmap="viridis"
)
axes[0, 0].set_title("True Model (Vertical, Y=0)")
axes[0, 0].set_xlabel("X (m)")
axes[0, 0].set_ylabel("Z (m)")
plt.colorbar(im0, ax=axes[0, 0], label="log10 Conductivity (S/m)")

im1 = axes[0, 1].imshow(
    np.log10(slice_rec.T),
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCz[-1], mesh.vectorCCz[0]],
    aspect="auto",
    cmap="viridis"
)
axes[0, 1].set_title("Inverted Model (Vertical, Y=0)")
axes[0, 1].set_xlabel("X (m)")
axes[0, 1].set_ylabel("Z (m)")
plt.colorbar(im1, ax=axes[0, 1], label="log10 Conductivity (S/m)")

# Horizontal slice (Z = -30 m)
ind_z = np.argmin(np.abs(mesh.vectorCCz + 30))
slice_true_h = model.reshape(mesh.vnC)[:, :, ind_z]
slice_rec_h = np.exp(mapping * recovered_model).reshape(mesh.vnC)[:, :, ind_z]

im2 = axes[1, 0].imshow(
    np.log10(slice_true_h.T),
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCy[0], mesh.vectorCCy[-1]],
    aspect="auto",
    cmap="viridis"
)
axes[1, 0].set_title("True Model (Horizontal, Z=-30m)")
axes[1, 0].set_xlabel("X (m)")
axes[1, 0].set_ylabel("Y (m)")
plt.colorbar(im2, ax=axes[1, 0], label="log10 Conductivity (S/m)")

im3 = axes[1, 1].imshow(
    np.log10(slice_rec_h.T),
    extent=[mesh.vectorCCx[0], mesh.vectorCCx[-1], mesh.vectorCCy[0], mesh.vectorCCy[-1]],
    aspect="auto",
    cmap="viridis"
)
axes[1, 1].set_title("Inverted Model (Horizontal, Z=-30m)")
axes[1, 1].set_xlabel("X (m)")
axes[1, 1].set_ylabel("Y (m)")
plt.colorbar(im3, ax=axes[1, 1], label="log10 Conductivity (S/m)")

plt.tight_layout()
plt.show()
```