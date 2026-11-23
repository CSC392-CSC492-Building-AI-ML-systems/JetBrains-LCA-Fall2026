```python
import numpy as np
import matplotlib.pyplot as plt
from discretize import CylindricalMesh
from SimPEG import (
    maps, data, data_misfit, regularization, optimization, inverse_problem,
    inversion, directives, utils
)
from SimPEG.electromagnetics.frequency_domain import simulation, survey, sources, receivers
from SimPEG.utils import mkvc

# Parameters
plot_results = True  # Set to False to skip plotting
n_layer = 3
layer_thickness = [10, 20, 70]  # meters
sigma_layers = [1e-2, 5e-3, 1e-1]  # S/m (fixed)
mu_layers = [1.0, 1.5, 2.0]  # relative magnetic permeability (to invert)
mu0 = 4 * np.pi * 1e-7  # H/m

# Mesh
ncz, npad, csz, csz_pad = 40, 20, 2, 5
hz = csz * np.ones(ncz)
hz = np.r_[hz, csz_pad * np.ones(npad)]
mesh = CylindricalMesh(
    [
        (1, 1),  # single cell in radial direction (cylindrical symmetry)
        (hz, 0)
    ],
    x0="00"
)

# Model parameters
sigma = np.ones(mesh.nCz) * sigma_layers[-1]
mu = np.ones(mesh.nCz) * mu_layers[-1]
z_edges = np.cumsum([0] + layer_thickness)
for i in range(n_layer):
    ind = (mesh.vectorCCz >= z_edges[i]) & (mesh.vectorCCz < z_edges[i+1])
    sigma[ind] = sigma_layers[i]
    mu[ind] = mu_layers[i]

# Frequency and source/receiver locations
frequencies = np.logspace(2, 4, 5)  # 100 Hz to 10 kHz
src_height = 1.0  # m above surface
rx_height = 1.0  # m above surface
src_loc = np.array([0.0, 0.0, src_height])
rx_loc = np.array([0.0, 0.0, rx_height])

# Receivers and sources
receiver_list = []
for freq in frequencies:
    receiver_list.append(
        receivers.PointMagneticFluxDensitySecondary(
            locations=np.array([rx_loc]),
            orientation="z",
            component="real",
            frequency=freq
        )
    )
    receiver_list.append(
        receivers.PointMagneticFluxDensitySecondary(
            locations=np.array([rx_loc]),
            orientation="z",
            component="imag",
            frequency=freq
        )
    )

source_list = [
    sources.MagDipole(
        receiver_list,
        location=src_loc,
        orientation="z",
        frequency=freq
    ) for freq in frequencies
]

survey_obj = survey.Survey(source_list)

# Mapping: model is log(mu_r)
act_map = maps.IdentityMap(nP=mesh.nCz)
mu_map = maps.ExpMap(nP=mesh.nCz)  # model is log(mu_r)

# Fixed conductivity
sigma_map = maps.ConstantMap(mesh.nCz)
sigma_const = sigma_layers[-1]  # background value

# Try to use PardisoSolver, else fallback to SolverLU
try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except ImportError:
    from pymatsolver import SolverLU
    Solver = SolverLU

# Simulation
sim = simulation.Simulation3DMagneticFluxDensity(
    mesh=mesh,
    survey=survey_obj,
    sigmaMap=maps.IdentityMap(nP=mesh.nCz),
    muMap=mu_map,
    Solver=Solver,
    sigma=sigma,
    mu0=mu0
)

# True model and synthetic data
log_mu_true = np.log(mu)
d_true = sim.dpred(log_mu_true)

# Add noise
np.random.seed(42)
noise_level = 0.03
noise = noise_level * np.abs(d_true) * np.random.randn(*d_true.shape)
d_obs = d_true + noise
dobs = data.Data(survey_obj, dobs=d_obs, noise_floor=noise_level * np.abs(d_true) + 1e-15)

# Inversion setup
m0 = np.log(np.ones(mesh.nCz))  # starting model: log(mu_r)=0 (mu_r=1)
reg = regularization.Simple(mesh, mapping=act_map)
opt = optimization.InexactGaussNewton(maxIter=15)
dmis = data_misfit.L2DataMisfit(simulation=sim, data=dobs)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
update_Jacobi = directives.UpdatePreconditioner()
dir_list = [betaest, target, save, update_Jacobi]

# Inversion
inv = inversion.BaseInversion(inv_prob, directiveList=dir_list)
log_mu_rec = inv.run(m0)

# Plotting
if plot_results:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot conductivity
    axs[0].plot(sigma, mesh.vectorCCz, label="Conductivity (S/m)")
    axs[0].set_xlabel("Conductivity (S/m)")
    axs[0].set_ylabel("Depth (m)")
    axs[0].set_title("Conductivity Model")
    axs[0].invert_yaxis()
    axs[0].grid(True)

    # Plot permeability
    axs[1].plot(np.exp(log_mu_true), mesh.vectorCCz, label="True", lw=2)
    axs[1].plot(np.exp(log_mu_rec), mesh.vectorCCz, label="Recovered", lw=2, ls="--")
    axs[1].set_xlabel("Relative Permeability")
    axs[1].set_title("Magnetic Permeability Model")
    axs[1].invert_yaxis()
    axs[1].legend()
    axs[1].grid(True)

    # Plot data misfit
    axs[2].plot(np.abs(d_true), "k-", label="True")
    axs[2].plot(np.abs(d_obs), "ro", label="Observed")
    dpred = sim.dpred(log_mu_rec)
    axs[2].plot(np.abs(dpred), "b--", label="Predicted")
    axs[2].set_xlabel("Data Index")
    axs[2].set_ylabel("|Bz| (T)")
    axs[2].set_title("Data Misfit")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
```