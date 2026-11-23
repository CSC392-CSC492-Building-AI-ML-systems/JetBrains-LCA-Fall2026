```python
import pyvista as pv
import numpy as np

# Planet data: name, radius (in 1000 km), distance from sun (in 10^7 km), texture url
planets = [
    {
        "name": "Mercury",
        "radius": 2.44,
        "distance": 5.8,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/2/2e/Mercury_in_true_color.jpg"
    },
    {
        "name": "Venus",
        "radius": 6.05,
        "distance": 10.8,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/e/e5/Venus-real_color.jpg",
        "atmosphere_texture": "https://upload.wikimedia.org/wikipedia/commons/8/85/Venus_globe.jpg"
    },
    {
        "name": "Earth",
        "radius": 6.37,
        "distance": 15.0,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg"
    },
    {
        "name": "Mars",
        "radius": 3.39,
        "distance": 22.8,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg"
    },
    {
        "name": "Jupiter",
        "radius": 69.9,
        "distance": 77.8,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/e/e2/Jupiter.jpg"
    },
    {
        "name": "Saturn",
        "radius": 58.2,
        "distance": 143.4,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Saturn_during_Equinox.jpg"
    },
    {
        "name": "Uranus",
        "radius": 25.4,
        "distance": 287.1,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Uranus2.jpg"
    },
    {
        "name": "Neptune",
        "radius": 24.6,
        "distance": 449.5,
        "texture": "https://upload.wikimedia.org/wikipedia/commons/5/56/Neptune_Full.jpg"
    }
]

# Scale factors for visualization
distance_scale = 0.2
radius_scale = 0.5

# Create plotter for solar system
plotter = pv.Plotter(window_size=(1600, 900), lighting='three lights')
plotter.set_background("black")

# Add a light source at the sun's position
sun_position = (0, 0, 0)
light = pv.Light(position=sun_position, focal_point=(1, 0, 0), color='white', intensity=2.0, light_type='scene light')
plotter.add_light(light)

# Create the sun
sun_radius = 109 * radius_scale  # Sun's radius ~109x Earth's
sun = pv.Sphere(radius=sun_radius, center=sun_position, theta_resolution=64, phi_resolution=64)
plotter.add_mesh(sun, color='yellow', smooth_shading=True, name="Sun", specular=1.0, specular_power=20)

# Add planets
planet_meshes = {}
for planet in planets:
    radius = planet["radius"] * radius_scale
    distance = planet["distance"] * distance_scale + sun_radius + radius * 2
    position = (distance, 0, 0)
    sphere = pv.Sphere(radius=radius, center=position, theta_resolution=64, phi_resolution=64)
    texture = pv.read_texture(planet["texture"])
    plotter.add_mesh(sphere, texture=texture, smooth_shading=True, name=planet["name"])
    planet_meshes[planet["name"]] = (sphere, texture, position, radius)

# Add orbits (circles)
for planet in planets:
    distance = planet["distance"] * distance_scale + sun_radius + planet["radius"] * radius_scale * 2
    theta = np.linspace(0, 2 * np.pi, 200)
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.zeros_like(theta)
    orbit = np.column_stack((x, y, z))
    plotter.add_lines(orbit, color='white', width=1, opacity=0.2)

# Show the solar system
plotter.show(title="Solar System 3D Visualization")

# Create subplots for individual planets
n_planets = len(planets)
n_cols = 4
n_rows = int(np.ceil(n_planets / n_cols))
subplotter = pv.Plotter(shape=(n_rows, n_cols), window_size=(1600, 900))
subplotter.set_background("black")

for idx, planet in enumerate(planets):
    row = idx // n_cols
    col = idx % n_cols
    subplotter.subplot(row, col)
    sphere = pv.Sphere(radius=planet["radius"] * radius_scale, theta_resolution=64, phi_resolution=64)
    texture = pv.read_texture(planet["texture"])
    subplotter.add_mesh(sphere, texture=texture, smooth_shading=True)
    subplotter.add_text(planet["name"], font_size=14, color='white')
    subplotter.camera_position = 'xy'

subplotter.show(title="Individual Planets with Textures")

# Venus with and without atmosphere
venus = next(p for p in planets if p["name"] == "Venus")
venus_radius = venus["radius"] * radius_scale
venus_texture = pv.read_texture(venus["texture"])
venus_atmo_texture = pv.read_texture(venus["atmosphere_texture"])

venus_plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500))
venus_plotter.set_background("black")

# Venus without atmosphere
venus_plotter.subplot(0, 0)
venus_sphere = pv.Sphere(radius=venus_radius, theta_resolution=64, phi_resolution=64)
venus_plotter.add_mesh(venus_sphere, texture=venus_texture, smooth_shading=True)
venus_plotter.add_text("Venus (Surface)", font_size=14, color='white')
venus_plotter.camera_position = 'xy'

# Venus with atmosphere
venus_plotter.subplot(0, 1)
venus_atmo_sphere = pv.Sphere(radius=venus_radius * 1.05, theta_resolution=64, phi_resolution=64)
venus_plotter.add_mesh(venus_atmo_sphere, texture=venus_atmo_texture, smooth_shading=True, opacity=0.8)
venus_plotter.add_text("Venus (Atmosphere)", font_size=14, color='white')
venus_plotter.camera_position = 'xy'

venus_plotter.show(title="Venus: Surface vs Atmosphere")
```