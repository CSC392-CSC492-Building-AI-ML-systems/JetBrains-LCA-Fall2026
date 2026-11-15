```python
import numpy as np
import matplotlib.pyplot as plt

from discretize import CylMesh
from SimPEG import maps, data, optimization, inverse_problem, inversion, directives, regularization, data_misfit, utils
from SimPEG.electromagnetics import frequency_domain as fdem

# Parameters
plot_results = True  # Set to True to plot results

# Mesh
ncz, npad, cs, h = 40, 20, 2.0, 1.3
hz = [(cs, ncz), (cs * h, npad)]
mesh = CylMesh([[(1, 1)], hz], x0="00")

# Model parameters
n_layer = 3
layer_thicknesses = [10, 20, None]  # Last layer is half-space
sigma_layers = [1e-2, 1e-1, 1e-3]   # S/m
mu_r_layers = [1.0, 2.0, 1.0]       # Relative magnetic permeability

# Assign conductivity and permeability
sigma = np.ones(mesh.nC) * sigma_layers[-1]
mu_r = np.ones(mesh.nC) * mu_r_layers[-1]
z = mesh.gridCC[:, 1]
z0 = 0
for i, (thick, sig, mur) in enumerate(zip(layer_thicknesses, sigma_layers, mu_r_layers)):
    if thick is not None:
        ind = (z < z0 + thick)
        sigma[ind] = sig
        mu_r[ind] = mur
        z0 += thick

# Fixed conductivity mapping
sigma_map = maps.IdentityMap(nP=mesh.nC)
sigma_model = sigma.copy()

# Magnetic susceptibility mapping (to invert for)
chi_true = mu_r - 1.0
chi_map = maps.IdentityMap(nP=mesh.nC)
chi_start = np.zeros(mesh.nC) + 0.01  # Initial guess

# Survey setup
frequencies = np.logspace(2, 4, 5)  # 100 Hz to 10 kHz
source_height = 5.0
receiver_height = 5.0
rx_offset = 10.0

source_list = []
for freq in frequencies:
    rx = fdem.receivers.PointMagneticFluxDensity(
        locations=np.array([[rx_offset, 0, receiver_height]]),
        orientation="z",
        component="real"
    )
    src = fdem.sources.CircularLoop(
        receiver_list=[rx],
        location=np.array([0, 0, source_height]),
        radius=5.0,
        frequency=freq
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

simulation = fdem.Simulation1DLayered(
    survey=survey,
    thicknesses=np.array([t for t in layer_thicknesses if t is not None]),
    sigmaMap=maps.IdentityMap(nP=n_layer),
    muMap=maps.IdentityMap(nP=n_layer),
    solver=Solver
)

# True model for data generation
sigma_1d = np.array(sigma_layers)
mu_1d = np.array(mu_r_layers) * 4 * np.pi * 1e-7
data_true = simulation.dpred(np.r_[sigma_1d, mu_1d / (4 * np.pi * 1e-7)])

# Add noise
noise_level = 0.05
noise = noise_level * np.abs(data_true) * np.random.randn(*data_true.shape)
dobs = data_true + noise

# Data object
data_obj = data.Data(survey, dobs=dobs, noise_floor=noise_level * np.abs(dobs))

# Inversion mappings
# Only invert for mu_r (magnetic susceptibility), fix sigma
def model_to_sigma(model):
    return sigma_1d

def model_to_mu(model):
    return (model + 1.0) * 4 * np.pi * 1e-7

sigma_map = maps.IdentityMap(nP=n_layer)
mu_map = maps.IdentityMap(nP=n_layer)

simulation = fdem.Simulation1DLayered(
    survey=survey,
    thicknesses=np.array([t for t in layer_thicknesses if t is not None]),
    sigmaMap=maps.InjectActiveCells(None, None, sigma_1d, nC=n_layer),
    muMap=maps.ExpMap(nP=n_layer),
    solver=Solver
)

# Data misfit
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_obj)

# Regularization
reg = regularization.Simple(mesh=mesh, mapping=maps.IdentityMap(nP=n_layer))

# Optimization
opt = optimization.InexactGaussNewton(maxIter=15)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Directives
betaest = directives.BetaEstimate_ByEig()
target = directives.TargetMisfit()
save = directives.SaveOutputEveryIteration()

# Inversion
inv = inversion.BaseInversion(inv_prob, directiveList=[betaest, target, save])

# Initial model (log(mu_r - 1))
m0 = np.log(np.ones(n_layer) * 0.01)

# Run inversion
recovered_model = inv.run(m0)

# Plotting
if plot_results:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot conductivity model
    axes[0].plot(sigma_1d, np.r_[0, np.cumsum([t for t in layer_thicknesses if t is not None])], 'o-')
    axes[0].set_xlabel('Conductivity (S/m)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title('Conductivity Model')
    axes[0].invert_yaxis()

    # Plot permeability model
    mu_r_rec = np.exp(recovered_model) + 1.0
    axes[1].plot(mu_r_rec, np.r_[0, np.cumsum([t for t in layer_thicknesses if t is not None])], 'o-')
    axes[1].set_xlabel('Relative Permeability')
    axes[1].set_ylabel('Depth (m)')
    axes[1].set_title('Recovered Permeability')
    axes[1].invert_yaxis()

    # Plot data misfit
    axes[2].plot(dobs, 'k.', label='Observed')
    dpred = simulation.dpred(recovered_model)
    axes[2].plot(dpred, 'r-', label='Predicted')
    axes[2].set_xlabel('Data Index')
    axes[2].set_ylabel('Magnetic Flux Density (T)')
    axes[2].set_title('Data Misfit')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
```