```python
import numpy as np
import matplotlib.pyplot as plt
from pysteps import data, io, motion, nowcasts, utils, verification
from pysteps.utils import transformation, conversion, dimension
from pysteps.visualization import plot_precip_field
from pysteps.verification import roc, reliability, rank

# 1. Read MeteoSwiss radar data
root_path = "data"  # Adjust path as needed
data_source = data.get_source("mch")
fns = data_source.find_files(root_path, "201806010600", "201806010700")
importer = data_source.get_method("importer")
R, _, metadata = io.read_timeseries(fns, importer, **data_source.importer_kwargs)

# 2. Upscale to 2 km resolution
R_upscaled = dimension.aggregate_fields_space(R, metadata, 2.0)

# 3. Convert to rain rate (mm/h)
R_rainrate, metadata = conversion.to_rainrate(R_upscaled, metadata)

# 4. Log-transform and handle missing values
R_log, R_thr = transformation.dB_transform(R_rainrate, threshold=0.1, zerovalue=-15.0)
R_log = np.where(np.isnan(R_log), -15.0, R_log)

# 5. Estimate motion field
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R_log[-3:])

# 6. STEPS ensemble nowcast
n_ens_members = 8
n_lead_times = 6
nowcast_method = nowcasts.get_method("steps")
R_f = R_log[-3:]
nowcast = nowcast_method(R_f, V, n_lead_times, n_ens_members, R_thr, 
                         kmperpixel=metadata["xpixelsize"]/1000.0, 
                         timestep=metadata["accutime"], 
                         seed=42)

# 7. Back-transform to rain rates
R_nowcast_ens = []
for i in range(n_ens_members):
    R_nowcast_member = []
    for t in range(n_lead_times):
        R_inv = transformation.dB_transform_inverse(nowcast[t, ..., i], threshold=R_thr, zerovalue=0.0)
        R_nowcast_member.append(R_inv)
    R_nowcast_ens.append(np.stack(R_nowcast_member))
R_nowcast_ens = np.stack(R_nowcast_ens, axis=-1)  # shape: (n_lead_times, H, W, n_ens_members)

# 8. Plot some realizations
plt.figure(figsize=(15, 8))
for i in range(min(4, n_ens_members)):
    plt.subplot(2, 2, i+1)
    plot_precip_field(R_nowcast_ens[2, ..., i], geodata=metadata, title=f"Ensemble member {i+1}, T+3")
plt.tight_layout()
plt.show()

# 9. Verification: ROC, reliability, rank histogram
# Prepare observations (truth) for verification
R_obs = R_rainrate[-n_lead_times:]

# ROC curve
thresholds = [0.5, 2.0, 5.0]  # mm/h
for thr in thresholds:
    fpr, tpr, _ = roc.roc_curve(R_nowcast_ens, R_obs, thr)
    plt.plot(fpr, tpr, label=f"{thr} mm/h")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Reliability diagram
for thr in thresholds:
    rel, obs_freq, _ = reliability.reliability_curve(R_nowcast_ens, R_obs, thr)
    plt.plot(rel, obs_freq, label=f"{thr} mm/h")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Forecast Probability")
plt.ylabel("Observed Frequency")
plt.title("Reliability Diagram")
plt.legend()
plt.show()

# Rank histogram
for t in range(n_lead_times):
    rhist = rank.rank_histogram(R_nowcast_ens[t], R_obs[t])
    plt.bar(np.arange(len(rhist)), rhist)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Rank Histogram, Lead time {t+1}")
    plt.show()
```