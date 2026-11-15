```python
import numpy as np
import matplotlib.pyplot as plt
from pysteps import io, motion, nowcasts, verification, utils
from pysteps.preprocessing import transformation
from pysteps.visualization import plot_precip_field
from pysteps.utils import conversion, dimension
from pysteps.verification import roc, reliability, rank

# 1. Read precipitation field data
root_path = "/path/to/meteoSwiss/data"
data_source = "mch"
importer = io.get_method(data_source)
fns = io.find_files(root_path, data_source, "*.hdf5")
R, _, metadata = io.read_timeseries(fns, importer, num_prev_files=3)

# 2. Upscale to 2 km resolution
R_upscaled, metadata_upscaled = dimension.aggregate_fields_space(R, metadata, 2.0)

# 3. Convert to rain rate
R_rainrate = conversion.to_rainrate(R_upscaled, metadata_upscaled)

# 4. Log-transform and handle missing values
R_log, R_thr = transformation.dB_transform(R_rainrate, threshold=0.1, zerovalue=-15.0, inverse=False)
R_log = np.where(np.isnan(R_log), -15.0, R_log)

# 5. Estimate motion field
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R_log[-3:])

# 6. STEPS ensemble nowcast
n_ens_members = 8
n_lead_times = 12
steps_method = nowcasts.get_method("steps")
R_f = steps_method(R_log[-1], V, n_lead_times, n_ens_members, R_thr, 
                   noise_method="nonparametric", 
                   ar_order=2, 
                   mask=~np.isnan(R_log[-1]))

# 7. Back-transform to rain rates
R_f_rainrate = transformation.dB_transform(R_f, threshold=0.1, zerovalue=-15.0, inverse=True)

# 8. Plot some realizations
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plot_precip_field(R_f_rainrate[0, i], geodata=metadata_upscaled, title=f"Ensemble member {i+1}")
plt.tight_layout()
plt.show()

# 9. Verification: ROC curve, reliability diagram, rank histogram
# Prepare observations and forecasts for verification
obs = R_rainrate[1:]  # Next time steps as "truth"
fcst = R_f_rainrate.transpose(1, 0, 2, 3)  # (n_lead_times, n_ens_members, ...)

# ROC curve
thresholds = [0.5, 1.0, 2.0, 5.0]
for thr in thresholds:
    roc_curve = roc.roc_curve(fcst, obs, thr)
    plt.plot(roc_curve["FAR"], roc_curve["POD"], label=f"{thr} mm/h")
plt.xlabel("False Alarm Rate")
plt.ylabel("Probability of Detection")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Reliability diagram
for thr in thresholds:
    rel = reliability.reliability_diagram(fcst, obs, thr)
    plt.plot(rel["mean_forecast"], rel["observed_frequency"], label=f"{thr} mm/h")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Forecast Probability")
plt.ylabel("Observed Frequency")
plt.title("Reliability Diagram")
plt.legend()
plt.show()

# Rank histogram
for lt in range(n_lead_times):
    rh = rank.rank_histogram(fcst[lt], obs[lt])
    plt.bar(np.arange(len(rh)), rh)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Rank Histogram (Lead time {lt+1})")
    plt.show()
```