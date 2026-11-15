```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from metpy.calc import lcl, parcel_profile, cape_cin
from metpy.cbook import get_test_data

# Load sample dataset (e.g., from MetPy's test data)
data = pd.read_csv(get_test_data('sounding_data.txt', as_file_obj=False), delim_whitespace=True, comment='#')

# Clean data: drop rows with NaN in required columns
data = data.dropna(subset=['pressure', 'temperature', 'dewpoint', 'wind_speed', 'wind_direction'])

# Assign units
p = data['pressure'].values * units.hPa
T = data['temperature'].values * units.degC
Td = data['dewpoint'].values * units.degC
u, v = None, None
if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
    wind_speed = data['wind_speed'].values * units.knots
    wind_dir = data['wind_direction'].values * units.degrees
    from metpy.calc import wind_components
    u, v = wind_components(wind_speed, wind_dir)

# Create figure with specific aspect ratio
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot temperature and dewpoint
skew.plot(p, T, 'r', linewidth=2, label='Temperature')
skew.plot(p, Td, 'g', linewidth=2, label='Dew Point')

# Plot wind barbs if wind data is available
if u is not None and v is not None:
    skew.plot_barbs(p, u, v)

# Set custom labels
skew.ax.set_xlabel('Temperature (°C)', fontsize=14)
skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)

# Calculate LCL
p_lcl, T_lcl = lcl(p[0], T[0], Td[0])
skew.plot(p_lcl, T_lcl, 'ko', markerfacecolor='black', label='LCL')

# Calculate parcel profile
parcel_prof = parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, parcel_prof, 'k', linewidth=2, label='Parcel Profile')

# Calculate CAPE and CIN
cape, cin, cape_prof, cin_prof = cape_cin(p, T, Td, parcel_prof)

# Shade CAPE and CIN areas
skew.shade_cape(p, T, parcel_prof)
skew.shade_cin(p, T, parcel_prof)

# Add special lines
skew.plot_dry_adiabats(linewidth=0.75)
skew.plot_moist_adiabats(linewidth=0.75)
skew.plot_mixing_lines(linewidth=0.75)

# Add legend
skew.ax.legend(loc='best')

plt.tight_layout()
plt.show()
```