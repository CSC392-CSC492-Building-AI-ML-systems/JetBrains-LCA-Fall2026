```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# City data: (name, lat, lon)
cities = [
    ('New York', 40.7128, -74.0060),
    ('Los Angeles', 34.0522, -118.2437),
    ('Chicago', 41.8781, -87.6298),
    ('Denver', 39.7392, -104.9903),
    ('Toronto', 43.651070, -79.347015)
]

# Generate lat/lon grid and some data
lats = np.linspace(-90, 90, 181)
lons = np.linspace(-180, 180, 361)
lon2d, lat2d = np.meshgrid(lons, lats)
data = np.sin(np.radians(lat2d)) * np.cos(2 * np.radians(lon2d))

# Map backgrounds to display
backgrounds = [
    ('Filled Continent', 'fillcontinents'),
    ('Land-Sea Mask', 'drawlsmask'),
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
    
    if bg == 'fillcontinents':
        m.fillcontinents(color='coral', lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bg == 'drawlsmask':
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
    
    # Plot city locations
    for name, lat, lon in cities:
        x, y = m(lon, lat)
        m.plot(x, y, 'o', markersize=8, color='red', markeredgecolor='k')
        ax.text(x + 100000, y + 100000, name, fontsize=9, fontweight='bold', color='black')
    
    # Contour data
    xg, yg = m(lon2d, lat2d)
    cs = m.contour(xg, yg, data, 8, linewidths=1, colors='k', alpha=0.5)
    m.contourf(xg, yg, data, 8, cmap='RdYlBu_r', alpha=0.5)
    
    ax.set_title(title, fontsize=14)

plt.tight_layout()
plt.show()
```