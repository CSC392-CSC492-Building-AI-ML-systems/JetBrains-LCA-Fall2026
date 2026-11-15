```python
import numpy as np
import pandas as pd
from metpy.units import units
from metpy.calc import (
    wind_components, wind_speed, wind_direction, lcl, lfc, el, parcel_profile,
    surface_based_cape_cin, most_unstable_parcel, mean_parcel, cape_cin,
    bulk_shear, bunkers_storm_motion, critical_angle, significant_tornado, supercell_composite,
    dewpoint_from_relative_humidity, mixed_layer, get_layer
)
from metpy.io import get_upper_air_data
from datetime import datetime

def effective_inflow_layer(p, T, Td, u, v, min_cape=100, max_cin=-250):
    """
    Determine the effective inflow layer following Thompson et al. (2004).
    Returns the pressure bounds (top, bottom) of the effective inflow layer.
    """
    eff_top = None
    eff_bot = None
    for i in range(len(p)-1, -1, -1):
        prof_p = p[i:]
        prof_T = T[i:]
        prof_Td = Td[i:]
        prof_u = u[i:]
        prof_v = v[i:]
        if len(prof_p) < 3:
            continue
        cape, cin = cape_cin(prof_p, prof_T, prof_Td, prof_p[0])
        if cape >= min_cape and cin >= max_cin:
            eff_bot = prof_p[0]
            break
    for i in range(len(p)):
        prof_p = p[i:]
        prof_T = T[i:]
        prof_Td = Td[i:]
        prof_u = u[i:]
        prof_v = v[i:]
        if len(prof_p) < 3:
            continue
        cape, cin = cape_cin(prof_p, prof_T, prof_Td, prof_p[0])
        if cape >= min_cape and cin >= max_cin:
            eff_top = prof_p[0]
    if eff_top is not None and eff_bot is not None and eff_bot >= eff_top:
        return eff_top, eff_bot
    else:
        return None, None

# Sample data: Use metpy's get_upper_air_data for a real sounding
df = get_upper_air_data(datetime(2017, 5, 18, 0), 'OUN')

# Isolate variables and attach units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
u, v = wind_components(df['u_wind'].values * units.knots, df['v_wind'].values * units.knots)

# Compute wind speed and direction
ws = wind_speed(u, v)
wd = wind_direction(u, v)

# Compute common sounding index parameters
ml_p, ml_T, ml_Td = mean_parcel(p, T, Td, depth=100 * units.hPa)
mu_p, mu_T, mu_Td = most_unstable_parcel(p, T, Td, depth=300 * units.hPa)

# Surface-based parcel profile
prof = parcel_profile(p, T[0], Td[0])

# Surface-based CAPE, CIN, LCL, LFC, EL
sfc_cape, sfc_cin = surface_based_cape_cin(p, T, Td)
sfc_lcl_p, sfc_lcl_T = lcl(p[0], T[0], Td[0])
sfc_lfc_p, sfc_lfc_T = lfc(p, T, Td)
sfc_el_p, sfc_el_T = el(p, T, Td)

# Mean-layer parcel (0-100 hPa)
ml_cape, ml_cin = cape_cin(p, T, Td, ml_p)
ml_lcl_p, ml_lcl_T = lcl(ml_p, ml_T, ml_Td)
ml_lfc_p, ml_lfc_T = lfc(p, T, Td, parcel_pressure=ml_p)
ml_el_p, ml_el_T = el(p, T, Td, parcel_pressure=ml_p)

# Most unstable parcel (0-300 hPa)
mu_cape, mu_cin = cape_cin(p, T, Td, mu_p)
mu_lcl_p, mu_lcl_T = lcl(mu_p, mu_T, mu_Td)
mu_lfc_p, mu_lfc_T = lfc(p, T, Td, parcel_pressure=mu_p)
mu_el_p, mu_el_T = el(p, T, Td, parcel_pressure=mu_p)

# Effective inflow layer
eff_top, eff_bot = effective_inflow_layer(p, T, Td, u, v)

# Bulk shear (0-6 km)
shear_0_6km = bulk_shear(p, u, v, height=None, depth=6000 * units.m)

# Bunkers storm motion and critical angle
rm, lm, mean_wind = bunkers_storm_motion(p, u, v, T, Td)
crit_angle = critical_angle(p, u, v, T, Td)

# Significant tornado parameter (STP)
stp = significant_tornado(p, T, Td, u, v)

# Supercell composite parameter (SCP)
scp = supercell_composite(p, T, Td, u, v)

# Print important sounding parameters
print(f"Surface CAPE: {sfc_cape:.0f}")
print(f"Surface CIN: {sfc_cin:.0f}")
print(f"Surface LCL: {sfc_lcl_p:.0f}")
print(f"Surface LFC: {sfc_lfc_p:.0f}")
print(f"Surface EL: {sfc_el_p:.0f}")
print(f"Mean-layer CAPE: {ml_cape:.0f}")
print(f"Mean-layer CIN: {ml_cin:.0f}")
print(f"Most Unstable CAPE: {mu_cape:.0f}")
print(f"Most Unstable CIN: {mu_cin:.0f}")
print(f"Effective Inflow Layer: {eff_top} - {eff_bot}")
print(f"0-6 km Bulk Shear: {shear_0_6km[0]:.1f}, {shear_0_6km[1]:.1f}")
print(f"Bunkers RM: {rm}")
print(f"Bunkers LM: {lm}")
print(f"Critical Angle: {crit_angle:.1f}")
print(f"Significant Tornado Parameter: {stp:.2f}")
print(f"Supercell Composite Parameter: {scp:.2f}")
```