```python
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.io import parse_metar_to_dataframe
from metpy.cbook import get_test_data
from metpy.calc import (wind_components, lcl, lfc, el, parcel_profile, surface_based_cape_cin,
                        mixed_layer_cape_cin, most_unstable_parcel, bunkers_storm_motion,
                        critical_angle, significant_tornado, supercell_composite)
import pandas as pd

def effective_inflow_layer(p, T, Td, u, v, cape_thresh=100 * units('J/kg'), cin_thresh=-250 * units('J/kg')):
    """
    Determine the effective inflow layer following Thompson et al. (2004).
    Returns the pressure bounds (top, bottom) of the effective inflow layer.
    """
    # Compute parcel profile for each level
    eff_layer = []
    for i in range(len(p)):
        # Only consider parcels within the lowest 300 hPa
        if p[i] < (p[0] - 300 * units.hPa):
            continue
        prof = parcel_profile(p[i:], T[i:], Td[i:]).to('degC')
        cape, cin = mpcalc.cape_cin(p[i:], T[i:], Td[i:], prof)
        if cape >= cape_thresh and cin >= cin_thresh:
            eff_layer.append(p[i].m)
    if not eff_layer:
        return None, None
    return max(eff_layer) * units.hPa, min(eff_layer) * units.hPa

# Read in sample data (using MetPy's test data)
data = pd.read_csv(get_test_data('sounding_data.txt', as_file_obj=False), delim_whitespace=True, comment='#', header=None,
                   names=['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed'])

# Isolate needed variables and attach units
p = data['pressure'].values * units.hPa
T = data['temperature'].values * units.degC
Td = data['dewpoint'].values * units.degC
wind_speed = data['speed'].values * units.knots
wind_dir = data['direction'].values * units.degrees

# Compute wind components
u, v = wind_components(wind_speed, wind_dir)

# Compute common sounding index parameters
ml_cape, ml_cin = mixed_layer_cape_cin(p, T, Td)
mu_parcel = most_unstable_parcel(p, T, Td)
mu_p, mu_T, mu_Td = mu_parcel
mu_prof = parcel_profile(p, mu_T, mu_Td)
mu_cape, mu_cin = mpcalc.cape_cin(p, mu_T, mu_Td, mu_prof)

# Surface-based parcel
sb_prof = parcel_profile(p, T, Td)
sb_cape, sb_cin = surface_based_cape_cin(p, T, Td)
sb_lcl_p, sb_lcl_T = lcl(p[0], T[0], Td[0])
sb_lfc_p, sb_lfc_T = lfc(p, T, Td)
sb_el_p, sb_el_T = el(p, T, Td)

# Mean layer parcel (lowest 100 hPa)
ml_p = p[p >= (p[0] - 100 * units.hPa)]
ml_T = T[p >= (p[0] - 100 * units.hPa)]
ml_Td = Td[p >= (p[0] - 100 * units.hPa)]
mean_T = np.mean(ml_T)
mean_Td = np.mean(ml_Td)
mean_p = np.mean(ml_p)
ml_prof = parcel_profile(p, mean_T, mean_Td)
ml_cape, ml_cin = mpcalc.cape_cin(p, mean_T, mean_Td, ml_prof)

# Bunkers Storm Motion vector and critical angle
right_mover, left_mover = bunkers_storm_motion(p, u, v, T)
crit_angle = critical_angle(p, u, v, T)

# Significant Tornado Parameter (STP)
stp = significant_tornado(p, T, Td, u, v)

# Supercell Composite Parameter (SCP)
scp = supercell_composite(p, T, Td, u, v)

# Effective inflow layer
eff_top, eff_bottom = effective_inflow_layer(p, T, Td, u, v)

# Print out important sounding parameters
print(f"Surface-based CAPE: {sb_cape:.1f}")
print(f"Surface-based CIN: {sb_cin:.1f}")
print(f"Surface-based LCL: {sb_lcl_p:.1f}")
print(f"Surface-based LFC: {sb_lfc_p:.1f}")
print(f"Surface-based EL: {sb_el_p:.1f}")
print(f"Mean-layer CAPE: {ml_cape:.1f}")
print(f"Mean-layer CIN: {ml_cin:.1f}")
print(f"Most Unstable CAPE: {mu_cape:.1f}")
print(f"Most Unstable CIN: {mu_cin:.1f}")
print(f"Bunkers Right Mover: {right_mover}")
print(f"Bunkers Left Mover: {left_mover}")
print(f"Critical Angle: {crit_angle:.1f}")
print(f"Significant Tornado Parameter: {stp:.2f}")
print(f"Supercell Composite Parameter: {scp:.2f}")
print(f"Effective Inflow Layer Top: {eff_top}")
print(f"Effective Inflow Layer Bottom: {eff_bottom}")
```