```python
import numpy as np
import matplotlib.pyplot as plt

from discretize import CylMesh
from SimPEG import (
    maps, data, inversion, inverse_problem, optimization, directives, regularization, data_misfit
)
from SimPEG.electromagnetics import frequency_domain as fdem

# Parameters
plot_results = True  # Set to True to plot results

# Mesh
ncz, npad, cs, csz = 40, 20, 2.0, 2.0
hz = [(cs, ncz), (csz, npad)]
mesh = CylMesh([[(1, 1)], hz], x0="00C")

# Geologic parameters
n_layer = 3
layer_thicknesses = [10, 20, None]  # Last layer is half-space
sigma_layers = [1e-2, 1e-1, 5e-2]   # S/m, fixed
mu_r_layers = [1.0, 2.0, 1.5]       # Relative magnetic permeability (to invert for)

# Assign conductivity model (fixed)
sigma = np.ones(mesh.nC) * sigma_layers[-1]
z_edges = mesh.vectorCCz
z0 = 0.0
z = np.abs(z_edges)
cum_thick = 0
for i, thick in enumerate(layer_thicknesses[:-1]):
    idx = (z >= cum_thick) & (z < cum_thick + thick)
    sigma[idx] = sigma_layers[i]
    cum_thick += thick

# Assign initial permeability model (to invert for)
mu0 = 4 * np.pi * 1e-7
mu_r = np.ones(mesh.nC) * mu_r_layers[-1]
cum_thick = 0
for i, thick in enumerate(layer_thicknesses[:-1]):
    idx = (z >= cum_thick) & (z < cum_thick + thick)
    mu_r[idx] = mu_r_layers[i]
    cum_thick += thick

# Map for permeability (log-space inversion)
actv = np.ones(mesh.nC, dtype=bool)
permeability_map = maps.ExpMap(mesh)  # invert for log(mu_r)
permeability_model = np.log(mu_r)

# Fixed conductivity map
conductivity_map = maps.IdentityMap(mesh)

# Frequencies and source/receiver locations
frequencies = np.logspace(2, 4, 5)  # 100 Hz to 10 kHz
src_height = 30.0
rx_height = 30.0
src_loc = np.array([[0.0, 0.0, src_height]])
rx_loc = np.array([[0.0, 0.0, rx_height]])

# Survey setup
source_list = []
for freq in frequencies:
    rx = fdem.receivers.PointMagneticFluxDensitySecondary(
        locations=rx_loc, orientation="z", component="real"
    )
    src = fdem.sources.CircularLoop(
        receiver_list=[rx], location=src_loc[0], frequency=freq, radius=10.0
    )
    source_list.append(src)
survey = fdem.Survey(source_list)

# Problem setup
try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except ImportError:
    from SimPEG.utils import SolverLU
    Solver = SolverLU

problem = fdem.Simulation3DMagneticFluxDensity(
    mesh,
    survey=survey,
    sigmaMap=conductivity_map,
    muMap=permeability_map,
    Solver=Solver,
    sigma=sigma,
)

# Synthetic data (generate with true model)
d_true = problem.dpred(np.log(mu_r))

# Add noise
noise_level = 0.03
noise = noise_level * np.abs(d_true) * np.random.randn(*d_true.shape)
d_obs = d_true + noise

# Data object
data_obj = data.Data(survey, dobs=d_obs, noise_floor=noise_level * np.abs(d_obs) + 1e-15)

# Data misfit
dmis = data_misfit.L2DataMisfit(simulation=problem, data=data_obj)

# Regularization
reg = regularization.Simple(mesh, mapping=maps.IdentityMap(nP=mesh.nC), active_cells=actv)

# Optimization
opt = optimization.InexactGaussNewton(maxIter=15)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
update_jacobi = directives.UpdatePreconditioner()

# Inversion
inv = inversion.BaseInversion(
    inv_prob, directiveList=[betaest, target, save, update_jacobi]
)

# Initial model (log(mu_r) = 0)
m0 = np.zeros(mesh.nC)

# Run inversion
recovered_model = inv.run(m0)

# Plotting
if plot_results:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot conductivity
    axes[0].plot(sigma, mesh.vectorCCz, label="Conductivity (S/m)")
    axes[0].set_xlabel("Conductivity (S/m)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Conductivity Model")
    axes[0].invert_yaxis()

    # Plot permeability
    axes[1].plot(np.exp(recovered_model), mesh.vectorCCz, label="Recovered mu_r")
    axes[1].plot(mu_r, mesh.vectorCCz, "--", label="True mu_r")
    axes[1].set_xlabel("Relative Permeability")
    axes[1].set_ylabel("Depth (m)")
    axes[1].set_title("Permeability Model")
    axes[1].legend()
    axes[1].invert_yaxis()

    # Plot data misfit
    axes[2].plot(d_obs, "k.", label="Observed")
    axes[2].plot(problem.dpred(recovered_model), "r-", label="Predicted")
    axes[2].set_xlabel("Data Index")
    axes[2].set_ylabel("Magnetic Flux Density (T)")
    axes[2].set_title("Data Misfit")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
```