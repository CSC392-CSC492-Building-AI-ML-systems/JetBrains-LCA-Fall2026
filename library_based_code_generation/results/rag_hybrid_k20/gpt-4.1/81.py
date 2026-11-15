```python
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.io import parse_metar_to_dataframe
from metpy.cbook import get_test_data
from metpy.calc import (parcel_profile, lcl, lfc, el, cape_cin, mixed_layer, most_unstable_parcel,
                        bulk_shear, bunkers_storm_motion, critical_angle, significant_tornado_parameter,
                        supercell_composite, wind_components, mean_pressure, mean_temperature, mean_dewpoint)

def effective_inflow_layer(p, T, Td, cape_thresh=100 * units('J/kg'), cin_thresh=-250 * units('J/kg')):
    """
    Determine the effective inflow layer for a convective sounding.
    Returns the pressure bounds (top, bottom) of the effective inflow layer.
    """
    eff_layer_top = None
    eff_layer_bottom = None
    for i in range(len(p)):
        prof = parcel_profile(p[i:], T[i:], Td[i:])
        this_cape, this_cin = cape_cin(p[i:], T[i:], Td[i:], prof)
        if this_cape >= cape_thresh and this_cin >= cin_thresh:
            eff_layer_bottom = p[i]
            break
    for j in range(i, len(p)):
        prof = parcel_profile(p[j:], T[j:], Td[j:])
        this_cape, this_cin = cape_cin(p[j:], T[j:], Td[j:], prof)
        if this_cape < cape_thresh or this_cin < cin_thresh:
            eff_layer_top = p[j-1]
            break
    if eff_layer_top is None:
        eff_layer_top = p[-1]
    return eff_layer_top, eff_layer_bottom

# Read in sample data (using MetPy's test data)
data = np.genfromtxt(get_test_data('sounding_data.txt', as_file_obj=False), skip_header=5)
p = data[:,0] * units.hPa
T = data[:,2] * units.degC
Td = data[:,3] * units.degC
u, v = mpcalc.wind_components(data[:,6] * units('knots'), data[:,7] * units('degrees'))

# Compute wind components
wind_speed = data[:,6] * units('knots')
wind_dir = data[:,7] * units('degrees')
u, v = wind_components(wind_speed, wind_dir)

# Compute common sounding index parameters
ml_p, ml_T, ml_Td = mixed_layer(p, T, Td, depth=100 * units.hPa)
mu_p, mu_T, mu_Td = most_unstable_parcel(p, T, Td, depth=300 * units.hPa)

# Surface-based parcel profile
prof_sfc = parcel_profile(p, T[0], Td[0])

# Surface-based CAPE, CIN, LCL, LFC, EL
cape_sfc, cin_sfc = cape_cin(p, T, Td, prof_sfc)
lcl_p, lcl_T = lcl(p[0], T[0], Td[0])
lfc_p, lfc_T = lfc(p, T, Td, prof_sfc)
el_p, el_T = el(p, T, Td, prof_sfc)
li_sfc = mpcalc.lifted_index(p, T, Td, p[0])

# Mean layer parcel (100 hPa)
ml_prof = parcel_profile(p, ml_T, ml_Td)
cape_ml, cin_ml = cape_cin(p, T, Td, ml_prof)
lcl_ml_p, lcl_ml_T = lcl(ml_p, ml_T, ml_Td)
lfc_ml_p, lfc_ml_T = lfc(p, T, Td, ml_prof)
el_ml_p, el_ml_T = el(p, T, Td, ml_prof)

# Most unstable parcel
mu_prof = parcel_profile(p, mu_T, mu_Td)
cape_mu, cin_mu = cape_cin(p, T, Td, mu_prof)
lcl_mu_p, lcl_mu_T = lcl(mu_p, mu_T, mu_Td)
lfc_mu_p, lfc_mu_T = lfc(p, T, Td, mu_prof)
el_mu_p, el_mu_T = el(p, T, Td, mu_prof)

# Bunkers Storm Motion and Critical Angle
right_mover, left_mover, mean_wind = bunkers_storm_motion(p, u, v)
crit_angle = critical_angle(p, u, v, T, Td)

# Significant Tornado Parameter (STP)
stp = significant_tornado_parameter(p, T, Td, u, v)

# Supercell Composite Parameter
scp = supercell_composite(p, T, Td, u, v)

# Effective inflow layer
eff_top, eff_bottom = effective_inflow_layer(p, T, Td)

# Print important sounding parameters
print(f"Surface-based CAPE: {cape_sfc:.1f}")
print(f"Surface-based CIN: {cin_sfc:.1f}")
print(f"Surface-based LI: {li_sfc:.1f}")
print(f"Surface LCL: {lcl_p:.1f}")
print(f"Surface LFC: {lfc_p:.1f}")
print(f"Surface EL: {el_p:.1f}")
print(f"Mean Layer CAPE: {cape_ml:.1f}")
print(f"Mean Layer CIN: {cin_ml:.1f}")
print(f"Most Unstable CAPE: {cape_mu:.1f}")
print(f"Most Unstable CIN: {cin_mu:.1f}")
print(f"Bunkers Right Mover: {right_mover}")
print(f"Bunkers Left Mover: {left_mover}")
print(f"Critical Angle: {crit_angle:.1f}")
print(f"Significant Tornado Parameter: {stp:.2f}")
print(f"Supercell Composite Parameter: {scp:.2f}")
print(f"Effective Inflow Layer: {eff_top:.1f} - {eff_bottom:.1f}")
```