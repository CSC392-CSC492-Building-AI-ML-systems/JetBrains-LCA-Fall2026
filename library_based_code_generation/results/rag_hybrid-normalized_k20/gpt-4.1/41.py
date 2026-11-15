```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from metpy.calc import lcl, parcel_profile, cape_cin
from metpy.cbook import get_test_data

# Load sample dataset
datafile = get_test_data('sounding_data.txt')
df = pd.read_csv(datafile, delim_whitespace=True, comment='#', header=None,
                 names=['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed'])

# Clean data: drop rows with NaN in pressure, temperature, dewpoint
df = df.dropna(subset=['pressure', 'temperature', 'dewpoint'])

# Assign units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC

# Create figure and SkewT plot with specific aspect ratio
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot temperature and dewpoint
skew.plot(p, T, 'r', linewidth=2, label='Temperature')
skew.plot(p, Td, 'g', linewidth=2, label='Dewpoint')

# Set custom labels
skew.ax.set_xlabel('Temperature (°C)', fontsize=14)
skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)

# Calculate LCL
lcl_pressure, lcl_temperature = lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', label='LCL')

# Calculate parcel profile
parcel_prof = parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, parcel_prof, 'k', linewidth=2, label='Parcel Profile')

# Shade CAPE and CIN
cape, cin, cape_prof, cin_prof = cape_cin(p, T, Td, parcel_prof)
skew.shade_cape(p, T, parcel_prof)
skew.shade_cin(p, T, parcel_prof)

# Add special lines
skew.plot_dry_adiabats(linewidth=0.75, alpha=0.7)
skew.plot_moist_adiabats(linewidth=0.75, alpha=0.7)
skew.plot_mixing_lines(linewidth=0.75, alpha=0.7)

# Add legend
skew.ax.legend(loc='best')

plt.tight_layout()
plt.show()
```