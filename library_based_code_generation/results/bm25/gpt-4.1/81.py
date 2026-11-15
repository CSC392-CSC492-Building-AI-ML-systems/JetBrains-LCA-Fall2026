```python
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.io import parse_metar_to_dataframe
from metpy.io.sounding import get_upper_air_data
from datetime import datetime
import pandas as pd

def effective_inflow_layer(p, T, Td, u, v, cape_thresh=100 * units('joule / kilogram'), cin_thresh=-250 * units('joule / kilogram')):
    """
    Determine the effective inflow layer for a convective sounding.
    Returns the pressure bounds (top, bottom) of the effective inflow layer.
    """
    # Compute CAPE and CIN for each level as a parcel
    cape = []
    cin = []
    for i in range(len(p)):
        prof_p = p[i:]
        prof_T = T[i:]
        prof_Td = Td[i:]
        if len(prof_p) < 3:
            cape.append(np.nan)
            cin.append(np.nan)
            continue
        prof_parcel_prof = mpcalc.parcel_profile(prof_p, prof_T[0], prof_Td[0])
        this_cape, this_cin = mpcalc.cape_cin(prof_p, prof_T, prof_Td, prof_parcel_prof)
        cape.append(this_cape)
        cin.append(this_cin)
    cape = np.array(cape) * units('joule / kilogram')
    cin = np.array(cin) * units('joule / kilogram')
    # Find levels where CAPE >= threshold and CIN >= threshold
    mask = (cape >= cape_thresh) & (cin >= cin_thresh)
    if not np.any(mask):
        return None, None
    indices = np.where(mask)[0]
    bottom = p[indices[0]]
    top = p[indices[-1]]
    return bottom, top

# Sample data: Use a sample sounding from Wyoming upper air archive
date = datetime(2020, 4, 12, 12)
station = 'OUN'
df = get_upper_air_data(date, station)

# Isolate needed variables and attach units
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
u, v = mpcalc.wind_components(df['u_wind'].values * units('knots'), df['wind_dir'].values * units.degree)

# Compute wind components (already done above)
# Compute common sounding index parameters
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
el_pressure, el_temperature = mpcalc.el(p, T, Td)

# Compute the parcel profile for a surface-based parcel
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])

# Compute corresponding LI, CAPE, CIN values for a surface parcel
li = mpcalc.lifted_index(p, T, Td, p[0], T[0], Td[0])
cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)

# Determine the LCL, LFC, and EL for the surface parcel (already computed above)

# Compute characteristics of a mean layer parcel (0-100 hPa mean layer)
ml_p, ml_T, ml_Td = mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa)
ml_parcel_prof = mpcalc.parcel_profile(p, ml_T, ml_Td)
ml_cape, ml_cin = mpcalc.cape_cin(p, T, Td, ml_parcel_prof)

# Compute characteristics of the most unstable parcel
mu_p, mu_T, mu_Td = mpcalc.most_unstable_parcel(p, T, Td, depth=300 * units.hPa)
mu_parcel_prof = mpcalc.parcel_profile(p, mu_T, mu_Td)
mu_cape, mu_cin = mpcalc.cape_cin(p, T, Td, mu_parcel_prof)

# Compute Bunkers Storm Motion vector and critical angle
right_mover, left_mover = mpcalc.bunkers_storm_motion(p, u, v, T, Td)
mean_wind = mpcalc.mean_wind(p, u, v, depth=6000 * units.meter)
shear_0_6km = mpcalc.bulk_shear(p, u, v, height=df['height'].values * units.meter, depth=6000 * units.meter)
shear_vec = shear_0_6km[0] + 1j * shear_0_6km[1]
bunkers_vec = right_mover[0] + 1j * right_mover[1]
critical_angle = np.degrees(np.arccos(np.real((shear_vec * np.conj(bunkers_vec)) / (abs(shear_vec) * abs(bunkers_vec)))))

# Compute characteristics needed for significant tornado parameter (STP)
srh_0_1km = mpcalc.storm_relative_helicity(p, u, v, df['height'].values * units.meter, depth=1000 * units.meter, storm_u=right_mover[0], storm_v=right_mover[1])[0]
stp = mpcalc.significant_tornado_parameter(p, T, Td, u, v, df['height'].values * units.meter, right_mover[0], right_mover[1])

# Compute supercell composite parameter if possible
scp = mpcalc.supercell_composite(p, T, Td, u, v, df['height'].values * units.meter, right_mover[0], right_mover[1])

# Determine effective inflow layer
eff_bottom, eff_top = effective_inflow_layer(p, T, Td, u, v)

# Print out important sounding parameters
print(f"Surface LCL: {lcl_pressure:.2f}, {lcl_temperature:.2f}")
print(f"Surface LFC: {lfc_pressure:.2f}, {lfc_temperature:.2f}")
print(f"Surface EL: {el_pressure:.2f}, {el_temperature:.2f}")
print(f"Surface CAPE: {cape:.2f}")
print(f"Surface CIN: {cin:.2f}")
print(f"Surface LI: {li:.2f}")
print(f"Mean Layer CAPE: {ml_cape:.2f}")
print(f"Mean Layer CIN: {ml_cin:.2f}")
print(f"Most Unstable CAPE: {mu_cape:.2f}")
print(f"Most Unstable CIN: {mu_cin:.2f}")
print(f"Bunkers Right Mover: {right_mover}")
print(f"Bunkers Left Mover: {left_mover}")
print(f"Critical Angle: {critical_angle:.2f} degrees")
print(f"0-1 km SRH: {srh_0_1km:.2f}")
print(f"Significant Tornado Parameter (STP): {stp:.2f}")
print(f"Supercell Composite Parameter (SCP): {scp:.2f}")
print(f"Effective Inflow Layer: {eff_bottom} to {eff_top}")
```