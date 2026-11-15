```python
import logging
import numpy as np
import capytaine as cpt
from capytaine.io.xarray import plot_free_surface_elevation
import matplotlib.pyplot as plt

# 1. Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Create a mesh of a sphere and floating body
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = 16
sphere_mesh = cpt.meshes.mesh_sphere(radius=radius, center=center, n_theta=resolution, n_phi=resolution)
floating_sphere = cpt.FloatingBody(sphere_mesh, name="sphere")
floating_sphere.add_translation_dof(name="Heave", direction=(0, 0, 1))

# 3. Extract the immersed part of the mesh
immersed_mesh = sphere_mesh.immersed_part()

# 4. Set up a BEMSolver
solver = cpt.BEMSolver()

# 5. Define and solve a diffraction problem
omega = 2.0
wave_direction = 0.0  # radians
diffraction_problem = cpt.DiffractionProblem(
    body=cpt.FloatingBody(immersed_mesh),
    omega=omega,
    wave_direction=wave_direction
)
diffraction_result = solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = "Heave"
radiation_problem = cpt.RadiationProblem(
    body=cpt.FloatingBody(immersed_mesh),
    omega=omega,
    radiating_dof=radiating_dof
)
radiation_result = solver.solve(radiation_problem)

# 7. Define a free surface
x_range = (-5, 5)
y_range = (-5, 5)
nx = 100
ny = 100
x = np.linspace(*x_range, nx)
y = np.linspace(*y_range, ny)
X, Y = np.meshgrid(x, y)
free_surface = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)

# 8. Compute the free surface elevation
diffraction_eta = diffraction_result.compute_free_surface_elevation_at_points(free_surface)
radiation_eta = radiation_result.compute_free_surface_elevation_at_points(free_surface)

# 9. Add incoming waves to the diffraction elevation
k = omega**2 / 9.81
incoming_wave = np.exp(1j * (k * (X * np.cos(wave_direction) + Y * np.sin(wave_direction))))
diffraction_eta_total = diffraction_eta.reshape((ny, nx)) + incoming_wave

# 10. Create and run two animations
from capytaine.io.xarray import animate_free_surface_elevation

camera_position = (10, 10, 10)

# Animation for diffraction
animate_free_surface_elevation(
    eta=diffraction_eta_total,
    X=X,
    Y=Y,
    mesh=sphere_mesh,
    face_motions=None,
    camera_position=camera_position,
    title="Diffraction Free Surface Elevation"
)

# Animation for radiation
animate_free_surface_elevation(
    eta=radiation_eta.reshape((ny, nx)),
    X=X,
    Y=Y,
    mesh=sphere_mesh,
    face_motions=None,
    camera_position=camera_position,
    title="Radiation Free Surface Elevation"
)
```