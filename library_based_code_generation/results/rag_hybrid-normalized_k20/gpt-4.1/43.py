```python
import numpy as np
import matplotlib.pyplot as plt
from pysteps import io, motion, nowcasts, verification, utils
from pysteps.visualization import plot_precip_field
from pysteps.utils import to_rainrate
from pysteps.preprocessing.transform import BoxCoxTransformer

# 1. Read precipitation field data (MeteoSwiss radar)
# Example: Use pysteps sample data
root_path = io.examples.get_path("mch")
importer = io.get_method("mch")
fns = io.find_files(root_path, "*.mch", provider="mch")[0]
R, metadata = io.read_timeseries(fns, importer, **io.read_mch_kwargs)

# 2. Upscale to 2 km resolution
R = utils.dimension.aggregate_fields_space(R, metadata, 2.0)

# 3. Convert to rain rate
R = to_rainrate(R, metadata)

# 4. Log-transform the data and handle missing values
transformer = BoxCoxTransformer()
R, params = transformer.transform(R, return_lambda=True)
R[np.isnan(R)] = np.nanmin(R)  # Fill missing values with min

# 5. Estimate the motion field
oflow_method = motion.get_method("lucaskanade")
V = oflow_method(R[-3:])

# 6. Perform ensemble nowcast using STEPS
n_ens_members = 8
n_lead_times = 12
nowcast_method = nowcasts.get_method("steps")
R_f = nowcast_method(R[-3:], V, n_lead_times, n_ens_members, R_thr=0.1, kmperpixel=2.0, seed=42)

# 7. Back-transform the nowcast to rain rates
R_f = transformer.inverse_transform(R_f)

# 8. Plot some of the realizations
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plot_precip_field(R_f[i, 0], geodata=metadata, title=f"Ensemble member {i+1}, t+1")
plt.tight_layout()
plt.show()

# 9. Verification: ROC curve, reliability diagram, rank histogram
# Prepare reference (observed) field
R_obs = R[-1]
R_obs = transformer.inverse_transform(R_obs)

# ROC curve
fct = R_f[:, :, :, :]
obs = np.stack([R_obs]*n_ens_members)
roc = verification.ROC()
roc.compute(fct[:, 0], obs, threshold=1.0)
roc.plot()
plt.title("ROC Curve")
plt.show()

# Reliability diagram
rel = verification.Reliability()
rel.compute(fct[:, 0], obs, threshold=1.0)
rel.plot()
plt.title("Reliability Diagram")
plt.show()

# Rank histogram
rank = verification.RankHistogram()
rank.compute(fct[:, 0], obs)
rank.plot()
plt.title("Rank Histogram")
plt.show()
```