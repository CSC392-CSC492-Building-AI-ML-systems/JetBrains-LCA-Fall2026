```python
import pyvista as pv
import numpy as np

# Planet data: name, radius (relative), distance from sun (relative), texture function
planets = [
    {
        "name": "Mercury",
        "radius": 0.383,
        "distance": 0.39,
        "color": [0.8, 0.7, 0.6],
        "texture": None
    },
    {
        "name": "Venus",
        "radius": 0.949,
        "distance": 0.72,
        "color": [0.9, 0.8, 0.5],
        "texture": pv.examples.download_venus_surface
    },
    {
        "name": "Earth",
        "radius": 1.0,
        "distance": 1.0,
        "color": [0.2, 0.4, 1.0],
        "texture": pv.examples.download_topo_global
    },
    {
        "name": "Mars",
        "radius": 0.532,
        "distance": 1.52,
        "color": [0.8, 0.4, 0.2],
        "texture": pv.examples.download_mars_surface
    },
    {
        "name": "Jupiter",
        "radius": 11.21,
        "distance": 5.20,
        "color": [0.9, 0.8, 0.7],
        "texture": pv.examples.download_jupiter_surface
    },
    {
        "name": "Saturn",
        "radius": 9.45,
        "distance": 9.58,
        "color": [0.9, 0.9, 0.7],
        "texture": None
    },
    {
        "name": "Uranus",
        "radius": 4.01,
        "distance": 19.18,
        "color": [0.6, 0.8, 0.9],
        "texture": None
    },
    {
        "name": "Neptune",
        "radius": 3.88,
        "distance": 30.07,
        "color": [0.3, 0.5, 0.9],
        "texture": None
    }
]

# Scale factors for visualization
DIST_SCALE = 5
RADIUS_SCALE = 0.3

# Create the plotter for the solar system
plotter = pv.Plotter(window_size=[1600, 900], lighting='three lights')
plotter.set_background("black")

# Add the Sun (as a light source and a sphere)
sun_radius = 1.5 * RADIUS_SCALE * 10
sun = pv.Sphere(radius=sun_radius, center=(0, 0, 0), theta_resolution=64, phi_resolution=64)
plotter.add_mesh(sun, color='yellow', smooth_shading=True, specular=1.0, name="Sun")

# Add a light source at the sun's position
light = pv.Light(position=(0, 0, 0), focal_point=(1, 0, 0), color='white', intensity=2.0, light_type='scene light')
plotter.add_light(light)

planet_meshes = []
planet_textures = []

# Add planets
for i, planet in enumerate(planets):
    radius = planet["radius"] * RADIUS_SCALE
    distance = planet["distance"] * DIST_SCALE + sun_radius + radius
    center = (distance, 0, 0)
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=64, phi_resolution=64)
    texture = None
    if planet["texture"]:
        try:
            tex = planet["texture"]()
            if hasattr(tex, 'texture'):
                texture = tex.texture
            else:
                texture = tex
        except Exception:
            texture = None
    if texture is not None:
        plotter.add_mesh(sphere, texture=texture, smooth_shading=True, name=planet["name"])
    else:
        plotter.add_mesh(sphere, color=planet["color"], smooth_shading=True, name=planet["name"])
    planet_meshes.append(sphere)
    planet_textures.append(texture)

# Show orbits (circles)
for planet in planets:
    distance = planet["distance"] * DIST_SCALE + sun_radius + planet["radius"] * RADIUS_SCALE
    theta = np.linspace(0, 2 * np.pi, 200)
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.zeros_like(theta)
    orbit = np.column_stack((x, y, z))
    plotter.add_lines(orbit, color='white', width=1, opacity=0.2)

plotter.add_text("Solar System", font_size=20, color='white', position='upper_left')

# Subplots for individual planets
n_planets = len(planets)
n_cols = 4
n_rows = int(np.ceil(n_planets / n_cols))
subplotter = pv.Plotter(shape=(n_rows, n_cols), window_size=[1600, 900])
for idx, planet in enumerate(planets):
    row = idx // n_cols
    col = idx % n_cols
    subplotter.subplot(row, col)
    radius = planet["radius"] * RADIUS_SCALE * 2
    sphere = pv.Sphere(radius=radius, theta_resolution=64, phi_resolution=64)
    texture = planet_textures[idx]
    if texture is not None:
        subplotter.add_mesh(sphere, texture=texture, smooth_shading=True)
    else:
        subplotter.add_mesh(sphere, color=planet["color"], smooth_shading=True)
    subplotter.add_text(planet["name"], font_size=12, color='white', position='lower_left')
    subplotter.set_background("black")
    subplotter.camera.zoom(1.5)

# Venus with and without atmosphere
venus_surface = pv.examples.load_venus()
venus_texture = pv.examples.download_venus_surface().texture

# Venus surface only
venus_plotter = pv.Plotter(shape=(1, 2), window_size=[1000, 500])
venus_plotter.subplot(0, 0)
venus_plotter.add_mesh(venus_surface, texture=venus_texture, smooth_shading=True)
venus_plotter.add_text("Venus Surface", font_size=14, color='white')
venus_plotter.set_background("black")

# Venus with atmosphere (add a semi-transparent sphere)
venus_plotter.subplot(0, 1)
venus_plotter.add_mesh(venus_surface, texture=venus_texture, smooth_shading=True)
# Atmosphere: slightly larger, semi-transparent sphere
atmosphere = pv.Sphere(radius=venus_surface.length / 2 * 1.03, theta_resolution=64, phi_resolution=64)
venus_plotter.add_mesh(atmosphere, color=[1.0, 0.9, 0.7], opacity=0.3, smooth_shading=True)
venus_plotter.add_text("Venus with Atmosphere", font_size=14, color='white')
venus_plotter.set_background("black")

# Show all visualizations
plotter.show(title="Solar System 3D Visualization")
subplotter.show(title="Planets Textures")
venus_plotter.show(title="Venus: Surface and Atmosphere")
```