```python
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# City data: (name, lat, lon)
cities = [
    ('New York', 40.7128, -74.0060),
    ('Los Angeles', 34.0522, -118.2437),
    ('Chicago', 41.8781, -87.6298),
    ('Denver', 39.7392, -104.9903),
    ('Toronto', 43.651070, -79.347015)
]

# Generate lat/lon grid and some data
lons = np.linspace(-180, 180, 181)
lats = np.linspace(-90, 90, 91)
lon2d, lat2d = np.meshgrid(lons, lats)
data = np.sin(np.radians(lat2d)) * np.cos(2 * np.radians(lon2d))

# Map backgrounds to plot
backgrounds = [
    ('Filled Continent', 'fillcontinents'),
    ('Land-Sea Mask', 'lsmask'),
    ('Blue Marble', 'bluemarble'),
    ('Shaded Relief', 'shadedrelief'),
    ('Etopo', 'etopo'),
    ('Etopo Transparent Land', 'etopo_transparent')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'aspect': 'equal'})
axes = axes.flatten()

for idx, (title, bg) in enumerate(backgrounds):
    ax = axes[idx]
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0], linewidth=0.5)
    m.drawmeridians(np.arange(-180, 181, 30), labels=[0,0,0,1], linewidth=0.5)
    m.drawmapboundary(fill_color='aqua')
    
    if bg == 'fillcontinents':
        m.fillcontinents(color='coral', lake_color='aqua')
    elif bg == 'lsmask':
        m.drawlsmask(land_color='coral', ocean_color='aqua', lakes=True)
    elif bg == 'bluemarble':
        m.bluemarble()
    elif bg == 'shadedrelief':
        m.shadedrelief()
    elif bg == 'etopo':
        m.etopo()
    elif bg == 'etopo_transparent':
        m.etopo(alpha=0.5)
        m.drawlsmask(land_color=(0,0,0,0), ocean_color='aqua', lakes=True)
    
    # Contour data
    x, y = m(lon2d, lat2d)
    cs = m.contourf(x, y, data, 10, cmap='RdYlBu', alpha=0.5)
    
    # Plot cities
    for city, lat, lon in cities:
        cx, cy = m(lon, lat)
        m.plot(cx, cy, 'o', color='black', markersize=8)
        ax.text(cx+100000, cy+100000, city, fontsize=9, fontweight='bold', color='black', ha='left', va='bottom')
    
    ax.set_title(title, fontsize=14)

plt.tight_layout()
plt.show()
```