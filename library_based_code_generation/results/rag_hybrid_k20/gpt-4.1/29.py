```python
import pyvista as pv
import numpy as np

# Planet data: name, radius (relative), distance from sun (AU), texture url or file
planets = [
    {
        "name": "Mercury",
        "radius": 0.383,
        "distance": 0.39,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/2/2e/Mercury_in_true_color.jpg"
    },
    {
        "name": "Venus",
        "radius": 0.949,
        "distance": 0.72,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg"
    },
    {
        "name": "Earth",
        "radius": 1.0,
        "distance": 1.0,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg"
    },
    {
        "name": "Mars",
        "radius": 0.532,
        "distance": 1.52,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg"
    },
    {
        "name": "Jupiter",
        "radius": 11.21,
        "distance": 5.20,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/e/e2/Jupiter.jpg"
    },
    {
        "name": "Saturn",
        "radius": 9.45,
        "distance": 9.58,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Saturn_during_Equinox.jpg"
    },
    {
        "name": "Uranus",
        "radius": 4.01,
        "distance": 19.18,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Uranus2.jpg"
    },
    {
        "name": "Neptune",
        "radius": 3.88,
        "distance": 30.07,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/5/56/Neptune_Full.jpg"
    }
]

# Scale factors for visualization
AU = 10.0  # Astronomical Unit in visualization units
RADIUS_SCALE = 0.5  # To make planets visible

# Create plotter for solar system
plotter = pv.Plotter(window_size=(1600, 900), lighting='three lights')

# Add the Sun (as a light source and a sphere)
sun_radius = 2.0
sun = pv.Sphere(radius=sun_radius, center=(0, 0, 0), theta_resolution=64, phi_resolution=64)
plotter.add_mesh(sun, color='yellow', smooth_shading=True, specular=1.0, name='Sun')

# Add a light source at the Sun's position
light = pv.Light(position=(0, 0, 0), focal_point=(1, 0, 0), color='white', intensity=2.0, light_type='scene light')
plotter.add_light(light)

planet_meshes = []

# Add planets
for planet in planets:
    radius = planet["radius"] * RADIUS_SCALE
    distance = planet["distance"] * AU + sun_radius + radius
    sphere = pv.Sphere(radius=radius, center=(distance, 0, 0), theta_resolution=64, phi_resolution=64)
    try:
        texture = pv.read_texture(planet["texture"])
    except Exception:
        texture = None
    plotter.add_mesh(sphere, texture=texture, smooth_shading=True, name=planet["name"])
    planet_meshes.append((planet["name"], sphere, texture))

# Set camera position for a nice view
plotter.camera_position = [(60, 40, 30), (0, 0, 0), (0, 0, 1)]

# Show the solar system
plotter.show(title="Solar System 3D Visualization")

# Create subplots for individual planets
n_planets = len(planets)
n_cols = 4
n_rows = int(np.ceil(n_planets / n_cols))
subplotter = pv.Plotter(shape=(n_rows, n_cols), window_size=(1600, 900))

for idx, (name, sphere, texture) in enumerate(planet_meshes):
    row = idx // n_cols
    col = idx % n_cols
    subplotter.subplot(row, col)
    # Center planet at origin for subplot
    planet_centered = pv.Sphere(radius=sphere.radius, center=(0, 0, 0), theta_resolution=64, phi_resolution=64)
    subplotter.add_mesh(planet_centered, texture=texture, smooth_shading=True)
    subplotter.add_text(name, font_size=14, position='upper_edge')

subplotter.show(title="Individual Planets with Textures")

# Venus with and without atmosphere
venus_surface = pv.download_venus_surface()
venus_no_atmo = pv.load_venus()
venus_texture = pv.read_texture("https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg")
venus_atmo_texture = pv.read_texture("https://upload.wikimedia.org/wikipedia/commons/8/85/Venus_globe.jpg")

venus_plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500))
venus_plotter.subplot(0, 0)
venus_plotter.add_mesh(venus_no_atmo, texture=venus_texture, smooth_shading=True)
venus_plotter.add_text("Venus Surface", font_size=14, position='upper_edge')
venus_plotter.subplot(0, 1)
venus_plotter.add_mesh(venus_surface, texture=venus_atmo_texture, smooth_shading=True)
venus_plotter.add_text("Venus with Atmosphere", font_size=14, position='upper_edge')
venus_plotter.show(title="Venus: Surface vs Atmosphere")
```