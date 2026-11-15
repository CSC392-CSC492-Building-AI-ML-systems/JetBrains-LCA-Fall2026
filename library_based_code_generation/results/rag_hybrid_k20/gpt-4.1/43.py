```python
import numpy as np
import matplotlib.pyplot as plt
from pysteps import io, motion, nowcasts, verification, utils
from pysteps.visualization import plot_precip_field
from pysteps.utils import transformation, conversion, dimension
from pysteps.verification import roc, reliability, rank
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Read precipitation field data (MeteoSwiss radar) ---
# Example: Use pysteps' demo data loader for MeteoSwiss
root_path = io.examples.get_path("mch")
importer = io.get_method("mch")
fns = io.find_files(root_path, "*.mch")
dates = io.get_precipitation_dates(fns)
input_files = io.get_precipitation_files(fns, dates[-12:])  # last 12 timesteps

R, _, metadata = io.read_timeseries(input_files, importer, **io.read_mch_kwargs)

# --- 2. Upscale to 2 km resolution ---
R_upscaled = dimension.aggregate_fields_space(R, metadata, 2000)

# --- 3. Convert to rain rate (mm/h) ---
R_rainrate = conversion.to_rainrate(R_upscaled, metadata)

# --- 4. Log-transform and handle missing values ---
R_log, R_thr = transformation.dB_transform(R_rainrate, threshold=0.1, zerovalue=-15.0, inverse=False)
R_log = np.where(np.isnan(R_log), -15.0, R_log)  # Fill missing with low value

# --- 5. Estimate motion field ---
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R_log[-3:])  # Use last 3 frames for motion estimation

# --- 6. Perform ensemble nowcast using STEPS ---
n_ens_members = 8
n_lead_times = 6
steps = nowcasts.get_method("steps")
R_f = R_log[-3:]  # Use last 3 frames as input

# STEPS parameters
steps_kwargs = dict(
    n_ens_members=n_ens_members,
    n_cascade_levels=6,
    kmperpixel=metadata["xpixelsize"] / 1000.0,
    timestep=metadata["accutime"],
    seed=42,
    R_thr=R_thr,
    mask_method="incremental",
    conditional=True,
    probmatching="cdf"
)

nowcast_generator = steps(R_f, V, **steps_kwargs)
R_ens_log = []
for i, R_ens_t in enumerate(nowcast_generator):
    if i >= n_lead_times:
        break
    R_ens_log.append(R_ens_t)
R_ens_log = np.stack(R_ens_log, axis=0)  # shape: (n_lead_times, n_ens_members, H, W)

# --- 7. Back-transform to rain rates ---
R_ens = transformation.dB_transform(R_ens_log, threshold=0.1, zerovalue=-15.0, inverse=True)

# --- 8. Plot some realizations ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    ax = axes[0, i]
    plot_precip_field(R_ens[0, i], ax=ax, title=f"Lead 1, Ens {i+1}")
    ax = axes[1, i]
    plot_precip_field(R_ens[-1, i], ax=ax, title=f"Lead {n_lead_times}, Ens {i+1}")
plt.tight_layout()
plt.show()

# --- 9. Verification: ROC curve, reliability diagram, rank histogram ---
# Prepare reference (observed) field for each lead time
R_obs = R_rainrate[-n_lead_times:]

# ROC curve for threshold 1 mm/h
threshold = 1.0
probs = (R_ens >= threshold).mean(axis=1)  # shape: (n_lead_times, H, W)
obs_bin = (R_obs >= threshold).astype(int)

fig, axes = plt.subplots(1, n_lead_times, figsize=(18, 4))
for t in range(n_lead_times):
    fpr, tpr, _ = roc.roc_curve(obs_bin[t].flatten(), probs[t].flatten())
    axes[t].plot(fpr, tpr, label=f"Lead {t+1}")
    axes[t].plot([0, 1], [0, 1], "k--")
    axes[t].set_title(f"ROC Lead {t+1}")
    axes[t].set_xlabel("FPR")
    axes[t].set_ylabel("TPR")
plt.tight_layout()
plt.show()

# Reliability diagram
fig, axes = plt.subplots(1, n_lead_times, figsize=(18, 4))
for t in range(n_lead_times):
    rel, obs_freq, _ = reliability.reliability_curve(obs_bin[t].flatten(), probs[t].flatten(), n_bins=10)
    axes[t].plot(rel, obs_freq, marker="o")
    axes[t].plot([0, 1], [0, 1], "k--")
    axes[t].set_title(f"Reliability Lead {t+1}")
    axes[t].set_xlabel("Forecast Probability")
    axes[t].set_ylabel("Observed Frequency")
plt.tight_layout()
plt.show()

# Rank histogram
fig, axes = plt.subplots(1, n_lead_times, figsize=(18, 4))
for t in range(n_lead_times):
    ranks = rank.rank_histogram(R_ens[t], R_obs[t])
    axes[t].bar(np.arange(len(ranks)), ranks)
    axes[t].set_title(f"Rank Histogram Lead {t+1}")
    axes[t].set_xlabel("Rank")
    axes[t].set_ylabel("Frequency")
plt.tight_layout()
plt.show()
```