```python
import pyvista as pv
import numpy as np
import urllib.request
import os

# Planet data: name, radius (relative), distance from sun (relative), texture url
planets = [
    {
        "name": "Mercury",
        "radius": 0.383,
        "distance": 0.4,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_mercury.jpg"
    },
    {
        "name": "Venus",
        "radius": 0.949,
        "distance": 0.7,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_venus_surface.jpg",
        "atmosphere": "https://www.solarsystemscope.com/textures/download/2k_venus_atmosphere.jpg"
    },
    {
        "name": "Earth",
        "radius": 1.0,
        "distance": 1.0,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
    },
    {
        "name": "Mars",
        "radius": 0.532,
        "distance": 1.5,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_mars.jpg"
    },
    {
        "name": "Jupiter",
        "radius": 11.21,
        "distance": 5.2,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_jupiter.jpg"
    },
    {
        "name": "Saturn",
        "radius": 9.45,
        "distance": 9.5,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_saturn.jpg"
    },
    {
        "name": "Uranus",
        "radius": 4.01,
        "distance": 19.2,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_uranus.jpg"
    },
    {
        "name": "Neptune",
        "radius": 3.88,
        "distance": 30.1,
        "texture": "https://www.solarsystemscope.com/textures/download/2k_neptune.jpg"
    }
]

# Sun data
sun_radius = 10.0
sun_texture = "https://www.solarsystemscope.com/textures/download/2k_sun.jpg"

# Download textures if not present
def download_texture(url, name):
    folder = "textures"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, os.path.basename(url))
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

# Scale factors for visualization
distance_scale = 10
radius_scale = 0.5

# Create plotter
plotter = pv.Plotter(window_size=[1600, 900], lighting='three lights')

# Add Sun
sun_texture_file = download_texture(sun_texture, "sun")
sun = pv.Sphere(radius=sun_radius * radius_scale, theta_resolution=64, phi_resolution=64)
plotter.add_mesh(sun, texture=sun_texture_file, smooth_shading=True)
plotter.add_light(pv.Light(position=(0, 0, 0), focal_point=(1, 0, 0), color='white', intensity=2.0, light_type='scene light'))

# Add planets
planet_meshes = []
for planet in planets:
    texture_file = download_texture(planet["texture"], planet["name"])
    radius = planet["radius"] * radius_scale
    distance = planet["distance"] * distance_scale + sun_radius * radius_scale + radius * 1.5
    sphere = pv.Sphere(radius=radius, theta_resolution=64, phi_resolution=64, center=(distance, 0, 0))
    plotter.add_mesh(sphere, texture=texture_file, smooth_shading=True)
    planet_meshes.append((planet, sphere, texture_file))

# Set camera
plotter.set_background("black")
plotter.camera_position = [(80, 40, 40), (0, 0, 0), (0, 0, 1)]

# Show the solar system
plotter.show(title="Solar System 3D Visualization")

# Subplots for individual planets
n_planets = len(planets)
n_cols = 4
n_rows = int(np.ceil(n_planets / n_cols))
subplotter = pv.Plotter(shape=(n_rows, n_cols), window_size=[1600, 900])
for idx, (planet, sphere, texture_file) in enumerate(planet_meshes):
    row = idx // n_cols
    col = idx % n_cols
    subplotter.subplot(row, col)
    planet_sphere = pv.Sphere(radius=planet["radius"] * radius_scale, theta_resolution=64, phi_resolution=64)
    subplotter.add_mesh(planet_sphere, texture=texture_file, smooth_shading=True)
    subplotter.add_text(planet["name"], font_size=14, color='white')
    subplotter.set_background("black")
    subplotter.camera_position = 'xy'
subplotter.show(title="Individual Planets")

# Venus with and without atmosphere
venus = next(p for p in planets if p["name"] == "Venus")
venus_surface_texture = download_texture(venus["texture"], "venus_surface")
venus_atmosphere_texture = download_texture(venus["atmosphere"], "venus_atmosphere")

venus_radius = venus["radius"] * radius_scale
atmosphere_radius = venus_radius * 1.05

venus_sphere = pv.Sphere(radius=venus_radius, theta_resolution=64, phi_resolution=64)
venus_atmosphere_sphere = pv.Sphere(radius=atmosphere_radius, theta_resolution=64, phi_resolution=64)

venus_plotter = pv.Plotter(shape=(1, 2), window_size=[1000, 500])
venus_plotter.subplot(0, 0)
venus_plotter.add_mesh(venus_sphere, texture=venus_surface_texture, smooth_shading=True)
venus_plotter.add_text("Venus Surface", font_size=14, color='white')
venus_plotter.set_background("black")
venus_plotter.camera_position = 'xy'

venus_plotter.subplot(0, 1)
venus_plotter.add_mesh(venus_sphere, texture=venus_surface_texture, smooth_shading=True)
venus_plotter.add_mesh(venus_atmosphere_sphere, texture=venus_atmosphere_texture, opacity=0.5, smooth_shading=True)
venus_plotter.add_text("Venus with Atmosphere", font_size=14, color='white')
venus_plotter.set_background("black")
venus_plotter.camera_position = 'xy'

venus_plotter.show(title="Venus: Surface vs Atmosphere")
```