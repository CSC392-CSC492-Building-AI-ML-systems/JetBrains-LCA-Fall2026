```python
import pyvista as pv
from pyvista import examples
import numpy as np

# Planet data: name, radius (relative), distance from sun (AU), texture function
planets = [
    {
        "name": "Mercury",
        "radius": 0.383,
        "distance": 0.39,
        "texture": examples.planets.download_mercury_surface
    },
    {
        "name": "Venus",
        "radius": 0.949,
        "distance": 0.72,
        "texture": examples.planets.download_venus_surface
    },
    {
        "name": "Earth",
        "radius": 1.0,
        "distance": 1.0,
        "texture": examples.planets.download_earth_surface
    },
    {
        "name": "Mars",
        "radius": 0.532,
        "distance": 1.52,
        "texture": examples.planets.download_mars_surface
    },
    {
        "name": "Jupiter",
        "radius": 11.21,
        "distance": 5.20,
        "texture": examples.planets.download_jupiter_surface
    },
    {
        "name": "Saturn",
        "radius": 9.45,
        "distance": 9.58,
        "texture": examples.planets.download_saturn_surface
    },
    {
        "name": "Uranus",
        "radius": 4.01,
        "distance": 19.18,
        "texture": examples.planets.download_uranus_surface
    },
    {
        "name": "Neptune",
        "radius": 3.88,
        "distance": 30.07,
        "texture": examples.planets.download_neptune_surface
    }
]

# Scale factors for visualization
AU = 10.0  # Astronomical Unit in visualization units
RADIUS_SCALE = 0.5  # To make planets visible

# Create plotter for solar system
plotter = pv.Plotter(window_size=(1600, 900), title="Solar System 3D Visualization", lighting=None)
plotter.set_background("black")

# Add a light source to simulate the sun
sun_light = pv.Light(position=(0, 0, 0), color='white', intensity=2.0, light_type='scene light')
plotter.add_light(sun_light)

# Create the sun (as a glowing sphere)
sun = pv.Sphere(radius=1.5, center=(0, 0, 0), theta_resolution=60, phi_resolution=60)
plotter.add_mesh(sun, color='yellow', emissive=True, smooth_shading=True, opacity=0.8, name="Sun")

# Store planet meshes for subplots
planet_meshes = {}

# Add planets
for planet in planets:
    radius = planet["radius"] * RADIUS_SCALE
    distance = planet["distance"] * AU + 2.5  # Offset so Mercury isn't inside the sun
    # Position planets along x-axis for simplicity
    center = (distance, 0, 0)
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=60, phi_resolution=60)
    try:
        texture = planet["texture"]()
    except Exception:
        texture = None
    plotter.add_mesh(sphere, texture=texture, smooth_shading=True, name=planet["name"])
    planet_meshes[planet["name"]] = (sphere, texture)

# Show orbits (circles)
for planet in planets:
    distance = planet["distance"] * AU + 2.5
    theta = np.linspace(0, 2 * np.pi, 200)
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.zeros_like(theta)
    orbit = np.column_stack((x, y, z))
    plotter.add_lines(orbit, color='white', width=1, opacity=0.2)

# Show the solar system
plotter.show(auto_close=False)

# Create subplots for individual planets
n_planets = len(planets)
n_cols = 4
n_rows = int(np.ceil(n_planets / n_cols))
subplotter = pv.Plotter(shape=(n_rows, n_cols), window_size=(1600, 900), title="Planet Textures")

for idx, planet in enumerate(planets):
    row = idx // n_cols
    col = idx % n_cols
    subplotter.subplot(row, col)
    sphere, texture = planet_meshes[planet["name"]]
    subplotter.add_mesh(sphere.copy().translate(-np.array(sphere.center)), texture=texture, smooth_shading=True)
    subplotter.add_text(planet["name"], font_size=14, color='white')
    subplotter.set_background("black")
    subplotter.camera.zoom(1.5)

subplotter.show(auto_close=False)

# Venus with and without atmosphere
venus_surface = examples.planets.download_venus_surface()
venus_atmosphere = examples.planets.download_venus_atmosphere()

venus_sphere = pv.Sphere(radius=0.949 * RADIUS_SCALE, theta_resolution=60, phi_resolution=60)
venus_atmo_sphere = pv.Sphere(radius=0.949 * RADIUS_SCALE * 1.05, theta_resolution=60, phi_resolution=60)

venus_plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500), title="Venus: Surface vs Atmosphere")
venus_plotter.subplot(0, 0)
venus_plotter.add_mesh(venus_sphere, texture=venus_surface, smooth_shading=True)
venus_plotter.add_text("Venus Surface", font_size=14, color='white')
venus_plotter.set_background("black")
venus_plotter.camera.zoom(1.5)

venus_plotter.subplot(0, 1)
venus_plotter.add_mesh(venus_sphere, texture=venus_surface, smooth_shading=True)
venus_plotter.add_mesh(venus_atmo_sphere, texture=venus_atmosphere, opacity=0.6, smooth_shading=True)
venus_plotter.add_text("Venus with Atmosphere", font_size=14, color='white')
venus_plotter.set_background("black")
venus_plotter.camera.zoom(1.5)

venus_plotter.show()
```