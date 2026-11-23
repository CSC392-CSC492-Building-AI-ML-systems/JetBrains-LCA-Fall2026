```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metpy.plots import SkewT
from metpy.units import units
from metpy.calc import lcl, parcel_profile, cape_cin

# Load sample dataset (using MetPy's example data)
from metpy.cbook import get_test_data
import xarray as xr

# Load the example sounding data
ds = xr.open_dataset(get_test_data('sounding_data.nc', as_file_obj=False))
df = ds.to_dataframe().reset_index()

# Clean data: drop rows with NaN in required columns
df = df.dropna(subset=['pressure', 'temperature', 'dewpoint', 'u_wind', 'v_wind'])

# Assign units
pressure = df['pressure'].values * units.hPa
temperature = df['temperature'].values * units.degC
dewpoint = df['dewpoint'].values * units.degC
u_wind = df['u_wind'].values * units.knots
v_wind = df['v_wind'].values * units.knots

# Create figure with specific aspect ratio
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot temperature and dewpoint
skew.plot(pressure, temperature, 'r', linewidth=2, label='Temperature')
skew.plot(pressure, dewpoint, 'g', linewidth=2, label='Dewpoint')

# Plot wind barbs
skew.plot_barbs(pressure, u_wind, v_wind)

# Set custom labels
skew.ax.set_xlabel('Temperature (°C)', fontsize=14)
skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)

# Calculate LCL
p_lcl, t_lcl = lcl(pressure[0], temperature[0], dewpoint[0])
skew.plot(p_lcl, t_lcl, 'ko', markerfacecolor='black', label='LCL')

# Calculate parcel profile
parcel_prof = parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
skew.plot(pressure, parcel_prof, 'k', linewidth=2, label='Parcel Profile')

# Calculate CAPE and CIN
cape, cin, cape_prof = cape_cin(pressure, temperature, dewpoint, parcel_prof)

# Shade CAPE and CIN areas
skew.shade_cape(pressure, temperature, parcel_prof)
skew.shade_cin(pressure, temperature, parcel_prof)

# Add special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Add legend
skew.ax.legend(loc='best')

# Display plot
plt.tight_layout()
plt.show()
```