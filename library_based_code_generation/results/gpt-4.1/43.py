```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pysteps import io, motion, nowcasts, verification, utils
from pysteps.visualization import plot_precip_field
from pysteps.utils import transformation, conversion, dimension
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

# --- PARAMETERS ---
data_path = "path_to_meteoswiss_data/"  # Update with your data path
file_pattern = "AQC_*.h5"  # Update with your file pattern
n_ens_members = 8
n_lead_times = 6  # e.g., 30 min at 5-min steps
target_resolution_km = 2.0

# --- READ DATA ---
# Read a sequence of radar composites
filenames = io.find_files(data_path, file_pattern)
importer = io.get_method("mch")
R, _, metadata = io.read_timeseries(filenames, importer, num_prev_files=2, num_next_files=0)

# --- UPSCALE TO 2 KM ---
orig_resolution = metadata["xpixelsize"] / 1000.0  # meters to km
zoom_factor = orig_resolution / target_resolution_km
R_upscaled = zoom(R, (1, zoom_factor, zoom_factor), order=1)

# --- CONVERT TO RAIN RATE ---
R_rate = conversion.to_rainrate(R_upscaled, metadata)

# --- LOG-TRANSFORM & HANDLE MISSING VALUES ---
R_masked, R_min = transformation.dB_transform(R_rate, threshold=0.1, zerovalue=-15.0, inverse=False, return_threshold=True)
R_masked = np.ma.masked_invalid(R_masked)

# --- ESTIMATE MOTION FIELD ---
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R_masked[-2:], metadata)

# --- ENSEMBLE NOWCAST (STEPS) ---
steps_method = nowcasts.get_method("steps")
R_filled = np.ma.filled(R_masked, fill_value=R_min)
R_filled = np.maximum(R_filled, R_min)
R_filled = np.expand_dims(R_filled[-1], axis=0)  # Use last frame as initial condition

# STEPS parameters
steps_kwargs = dict(
    n_ens_members=n_ens_members,
    n_cascade_levels=6,
    kmperpixel=target_resolution_km,
    timestep=5,
    n_timesteps=n_lead_times,
    R_thr=0.1,
    mask_method="incremental",
    probmatching="cdf",
    seed=42,
)

R_ens = steps_method(R_filled, V, **steps_kwargs)

# --- BACK-TRANSFORM TO RAIN RATE ---
R_ens_rate = []
for ens in range(n_ens_members):
    R_ens_rate.append(transformation.dB_transform(R_ens[ens], threshold=0.1, zerovalue=-15.0, inverse=True))
R_ens_rate = np.stack(R_ens_rate, axis=0)  # shape: (n_ens, n_lead_times, H, W)

# --- PLOT SOME REALIZATIONS ---
plt.figure(figsize=(15, 6))
for i in range(min(4, n_ens_members)):
    plt.subplot(2, 2, i+1)
    plot_precip_field(R_ens_rate[i, -1], geodata=metadata, title=f"Ensemble {i+1}, t+{n_lead_times*5}min")
plt.tight_layout()
plt.show()

# --- VERIFICATION ---
# Read corresponding observations for verification
obs_filenames = filenames[-n_lead_times:]
obs_R, _, _ = io.read_timeseries(obs_filenames, importer)
obs_R_upscaled = zoom(obs_R, (1, zoom_factor, zoom_factor), order=1)
obs_R_rate = conversion.to_rainrate(obs_R_upscaled, metadata)

# --- ROC CURVE ---
threshold = 1.0  # mm/h
obs_bin = (obs_R_rate >= threshold).astype(int)
fcst_prob = (R_ens_rate >= threshold).mean(axis=0)  # shape: (n_lead_times, H, W)

fpr, tpr, _ = roc_curve(obs_bin.ravel(), fcst_prob.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --- RELIABILITY DIAGRAM ---
from pysteps.verification.ens import reliability
rel = reliability(fcst_prob, obs_bin, n_bins=10)
plt.figure()
plt.plot(rel["mean_forecast_prob"], rel["observed_freq"], marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Forecast Probability")
plt.ylabel("Observed Relative Frequency")
plt.title("Reliability Diagram")
plt.show()

# --- RANK HISTOGRAM ---
from pysteps.verification.ens import rankhist
hist, bin_edges = rankhist(R_ens_rate, obs_R_rate)
plt.figure()
plt.bar(range(len(hist)), hist, width=0.8)
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Rank Histogram")
plt.show()
```