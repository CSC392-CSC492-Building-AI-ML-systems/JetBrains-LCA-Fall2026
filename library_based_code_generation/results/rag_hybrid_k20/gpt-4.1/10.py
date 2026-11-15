```python
import numpy as np
import matplotlib.pyplot as plt

from discretize import CylMesh
from SimPEG import (
    maps, data, inversion, inverse_problem, optimization, directives, regularization, data_misfit
)
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.utils import mkvc

# Set plotting flag
plot_results = True

# Mesh parameters
ncz, npad, cs, h, pad_rate = 40, 20, 2.0, 2.0, 1.3
hz = [(cs, ncz), (cs * pad_rate, npad)]
mesh = CylMesh([[(1, 1)], hz], x0="00C")

# Geologic parameters
n_layer = 3
layer_thicknesses = [10, 20, 30]  # meters
sigma_layers = [1e-2, 5e-3, 1e-2]  # S/m (fixed)
mu_layers = [1.0, 1.5, 2.0]  # relative magnetic permeability

# Assign conductivity and permeability models
sigma = np.ones(mesh.nC) * sigma_layers[-1]
mu = np.ones(mesh.nC) * mu_layers[-1]
z_edges = mesh.vectorCCz
z0 = 0.0
for i, (thick, sig, mu_r) in enumerate(zip(layer_thicknesses, sigma_layers, mu_layers)):
    ind = (z_edges >= z0) & (z_edges < z0 + thick)
    sigma[ind] = sig
    mu[ind] = mu_r
    z0 += thick

# Fixed conductivity mapping
sigma_map = maps.IdentityMap(nP=mesh.nC)
sigma_model = np.log(sigma)

# Magnetic susceptibility mapping (to invert for)
chi_true = mu - 1.0
chi_map = maps.IdentityMap(nP=mesh.nC)
chi_start = np.log(np.ones(mesh.nC) * 0.1)

# Survey setup
frequencies = np.logspace(2, 4, 10)  # 100 Hz to 10 kHz
source_height = 30.0
receiver_height = 0.0
src_loc = np.array([[0.0, 0.0, source_height]])
rx_loc = np.array([[0.0, 0.0, receiver_height]])

rx = fdem.receivers.PointMagneticFluxDensitySecondary(rx_loc, orientation="z", component="real")
src = fdem.sources.CircularLoop([rx], location=src_loc[0], frequency=frequencies)
survey = fdem.Survey([src])

# Problem setup
try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except ImportError:
    from SimPEG import SolverLU
    Solver = SolverLU

problem = fdem.Simulation3DMagneticField(
    mesh,
    survey=survey,
    sigmaMap=sigma_map,
    chiMap=chi_map,
    Solver=Solver,
    sigma=sigma,
    chi=chi_true,
    storeJ=True,
)

# Generate synthetic data
d_true = problem.dpred(np.log(chi_true))
noise = 0.03 * np.abs(d_true) * np.random.randn(*d_true.shape)
d_obs = d_true + noise
dobs = data.Data(survey, dobs=d_obs, noise_floor=0.03 * np.abs(d_true))

# Inversion setup
reg = regularization.Simple(mesh, mapping=maps.ExpMap(nP=mesh.nC))
opt = optimization.InexactGaussNewton(maxIter=15)
dmis = data_misfit.L2DataMisfit(simulation=problem, data=dobs)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig()
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
update_jac = directives.UpdatePreconditioner()

inv = inversion.BaseInversion(
    inv_prob,
    directiveList=[betaest, target, save, update_jac]
)

# Run inversion
chi_inv_log = inv.run(chi_start)
chi_inv = np.exp(chi_inv_log)

# Plotting
if plot_results:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot conductivity model
    axs[0].plot(sigma, mesh.vectorCCz, label="Conductivity (S/m)")
    axs[0].set_xlabel("Conductivity (S/m)")
    axs[0].set_ylabel("Depth (m)")
    axs[0].set_title("Conductivity Model")
    axs[0].invert_yaxis()
    axs[0].grid(True)

    # Plot permeability model
    axs[1].plot(mu, mesh.vectorCCz, label="True $\mu_r$")
    axs[1].plot(chi_inv + 1.0, mesh.vectorCCz, label="Inverted $\mu_r$")
    axs[1].set_xlabel("Relative Permeability")
    axs[1].set_ylabel("Depth (m)")
    axs[1].set_title("Permeability Model")
    axs[1].invert_yaxis()
    axs[1].legend()
    axs[1].grid(True)

    # Plot data misfit
    axs[2].plot(d_obs, 'k.', label="Observed")
    axs[2].plot(problem.dpred(chi_inv_log), 'r-', label="Predicted")
    axs[2].set_xlabel("Data Index")
    axs[2].set_ylabel("Magnetic Flux Density (T)")
    axs[2].set_title("Data Misfit")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
```