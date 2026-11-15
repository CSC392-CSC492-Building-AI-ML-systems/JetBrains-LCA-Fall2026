```python
import numpy as np
import matplotlib.pyplot as plt

from discretize import CylindricalMesh
from SimPEG import maps, data, inversion, inverse_problem, optimization, directives, regularization, data_misfit
from SimPEG.electromagnetics.frequency_domain import simulation, survey, sources, receivers
from SimPEG.utils import mkvc

# Set plotting flag
plot_results = True

# Mesh parameters
ncz, npad, cs, h, pad_rate = 40, 20, 2.0, 2.0, 1.3
hz = [(cs, ncz), (h, npad, pad_rate)]
hr = [(cs, 10), (h, npad, pad_rate)]
mesh = CylindricalMesh([hr, 1, hz], x0="00C")

# Geologic parameters
n_layer = 3
layer_tops = np.r_[-10., -30., -100.]
chi_true = np.r_[0.01, 0.05, 0.1]  # Magnetic susceptibility (SI)
sigma_true = 1e-2  # S/m, fixed

# Assign susceptibility and conductivity
chi_model = np.zeros(mesh.nC)
sigma_model = np.ones(mesh.nC) * sigma_true

z = mesh.cell_centers_z
for i in range(n_layer):
    if i == 0:
        ind = z > layer_tops[i]
    elif i == n_layer - 1:
        ind = z <= layer_tops[i-1]
    else:
        ind = (z <= layer_tops[i-1]) & (z > layer_tops[i])
    chi_model[ind] = chi_true[i]

# Set up mapping for inversion (invert for log(chi))
active = np.ones(mesh.nC, dtype=bool)
chi_map = maps.ExpMap(nP=mesh.nC)
sigma_map = maps.IdentityMap(nP=mesh.nC)

# Set up relative permeability (mu_r = 1 + chi)
mu_r = 1 + chi_model

# FDEM survey setup
rx_offset = 10.0
frequencies = np.logspace(2, 4, 10)
rx = receivers.PointMagneticFluxDensitySecondary(
    locations=np.array([[rx_offset, 0, 0]]),
    orientation="z",
    component="real"
)
src = sources.CircularLoop(
    receiver_list=[rx],
    location=np.array([0, 0, 0]),
    frequency=frequencies
)
survey_fdem = survey.Survey([src])

# Set up FDEM simulation
try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except ImportError:
    from SimPEG import SolverLU
    Solver = SolverLU

sim = simulation.Simulation3DMagneticFluxDensity(
    mesh=mesh,
    survey=survey_fdem,
    sigma=sigma_model,
    mu=mu_r,
    Solver=Solver,
    storeJ=True
)

# Generate synthetic data
d_true = sim.dpred(chi_model)

# Add noise
np.random.seed(42)
noise_level = 0.05
noise = noise_level * np.abs(d_true) * np.random.randn(*d_true.shape)
d_obs = d_true + noise

data_obj = data.Data(survey_fdem, dobs=d_obs, noise_floor=noise_level * np.abs(d_obs))

# Inversion setup
m0 = np.log(np.ones(mesh.nC) * 0.01)
reg = regularization.Simple(mesh, mapping=maps.IdentityMap(nP=mesh.nC))
dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj, mapping=chi_map)
opt = optimization.InexactGaussNewton(maxIter=15)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig()
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()
update_Jacobi = directives.UpdatePreconditioner()

inv = inversion.BaseInversion(
    inv_prob,
    directiveList=[betaest, target, save, update_Jacobi]
)

# Run inversion
recovered_model = inv.run(m0)

# Plotting
if plot_results:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot conductivity
    axes[0].plot(sigma_model, z, label="True")
    axes[0].set_xlabel("Conductivity (S/m)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Conductivity Model")
    axes[0].invert_yaxis()
    # Plot susceptibility
    axes[1].plot(np.exp(recovered_model), z, label="Recovered")
    axes[1].plot(chi_model, z, "--", label="True")
    axes[1].set_xlabel("Magnetic Susceptibility (SI)")
    axes[1].set_title("Magnetic Susceptibility")
    axes[1].legend()
    axes[1].invert_yaxis()
    # Plot data misfit
    axes[2].plot(d_obs, label="Observed")
    axes[2].plot(sim.dpred(np.exp(recovered_model)), label="Predicted")
    axes[2].set_xlabel("Data Index")
    axes[2].set_ylabel("Magnetic Flux Density (T)")
    axes[2].set_title("Data Misfit")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
```