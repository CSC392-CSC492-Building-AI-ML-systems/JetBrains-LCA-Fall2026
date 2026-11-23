```python
import numpy as np
import pandas as pd
from metpy.units import units
from metpy.calc import (
    wind_components, parcel_profile, lcl, lfc, el, cape_cin, mean_layer_parcel,
    most_unstable_parcel, bulk_shear, bunkers_storm_motion, critical_angle,
    mixed_layer_cape_cin, significant_tornado, supercell_composite
)
from metpy.calc import dewpoint_from_relative_humidity
from metpy.io import parse_metar_to_dataframe

def effective_inflow_layer(p, T, Td, u, v, cape_thresh=100, cin_thresh=-250):
    """
    Determine the effective inflow layer following Thompson et al. (2004).
    Returns the pressure bounds (top, bottom) of the effective inflow layer.
    """
    nlev = len(p)
    eff_top = None
    eff_bot = None
    for i in range(nlev-1):
        # Compute CAPE and CIN for a parcel starting at this level
        prof = parcel_profile(p, T, Td, p[i])
        this_cape, this_cin = cape_cin(p, T, Td, prof)
        if this_cape >= cape_thresh and this_cin > cin_thresh:
            if eff_bot is None:
                eff_bot = p[i]
            eff_top = p[i]
    return eff_bot, eff_top

# Sample data (replace with actual sounding data as needed)
# For demonstration, we'll use a simple synthetic sounding
data = {
    'pressure': [1000, 925, 850, 700, 500, 400, 300, 250, 200],  # hPa
    'temperature': [20, 15, 10, 0, -10, -20, -30, -40, -50],     # degC
    'dewpoint': [18, 13, 8, -2, -20, -30, -40, -50, -60],        # degC
    'wind_speed': [10, 15, 20, 25, 30, 35, 40, 45, 50],          # knots
    'wind_dir': [180, 190, 200, 210, 220, 230, 240, 250, 260],   # degrees
}
df = pd.DataFrame(data)

# Attach units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
ws = df['wind_speed'].values * units.knots
wd = df['wind_dir'].values * units.degrees

# Compute wind components
u, v = wind_components(ws, wd)

# Compute common sounding index parameters
# Surface-based parcel
prof = parcel_profile(p, T, Td)
sb_cape, sb_cin = cape_cin(p, T, Td, prof)
sb_lcl_p, sb_lcl_T = lcl(p[0], T[0], Td[0])
sb_lfc_p, sb_lfc_T = lfc(p, T, Td)
sb_el_p, sb_el_T = el(p, T, Td)

# Mean layer parcel (lowest 100 hPa)
ml_p, ml_T, ml_Td = mean_layer_parcel(p, T, Td, depth=100 * units.hPa)
ml_prof = parcel_profile(p, T, Td, ml_p)
ml_cape, ml_cin = cape_cin(p, T, Td, ml_prof)

# Most unstable parcel (lowest 300 hPa)
mu_p, mu_T, mu_Td = most_unstable_parcel(p, T, Td, depth=300 * units.hPa)
mu_prof = parcel_profile(p, T, Td, mu_p)
mu_cape, mu_cin = cape_cin(p, T, Td, mu_prof)

# Bunkers storm motion
right_mover, left_mover, mean_wind = bunkers_storm_motion(p, u, v, depth=6000 * units.meter)

# Critical angle (using surface and 500m wind)
crit_angle = critical_angle(p, u, v)

# Bulk shear 0-6 km
shear_0_6km = bulk_shear(p, u, v, height=None, depth=6000 * units.meter)

# Significant tornado parameter (using surface, 1km, 3km, 6km, etc.)
stp = significant_tornado(p, T, Td, u, v)

# Supercell composite parameter
scp = supercell_composite(p, T, Td, u, v)

# Effective inflow layer
eff_bot, eff_top = effective_inflow_layer(p, T, Td, u, v)

# Print important sounding parameters
print(f"Surface-based CAPE: {sb_cape:.1f}")
print(f"Surface-based CIN: {sb_cin:.1f}")
print(f"Surface-based LCL: {sb_lcl_p:.1f}, {sb_lcl_T:.1f}")
print(f"Surface-based LFC: {sb_lfc_p:.1f}, {sb_lfc_T:.1f}")
print(f"Surface-based EL: {sb_el_p:.1f}, {sb_el_T:.1f}")
print(f"Mean-layer CAPE: {ml_cape:.1f}")
print(f"Mean-layer CIN: {ml_cin:.1f}")
print(f"Most unstable CAPE: {mu_cape:.1f}")
print(f"Most unstable CIN: {mu_cin:.1f}")
print(f"Bunkers right-mover: {right_mover}")
print(f"Bunkers left-mover: {left_mover}")
print(f"Critical angle: {crit_angle:.1f}")
print(f"Bulk shear 0-6 km: {shear_0_6km}")
print(f"Significant Tornado Parameter: {stp}")
print(f"Supercell Composite Parameter: {scp}")
print(f"Effective inflow layer: {eff_bot} to {eff_top}")
```