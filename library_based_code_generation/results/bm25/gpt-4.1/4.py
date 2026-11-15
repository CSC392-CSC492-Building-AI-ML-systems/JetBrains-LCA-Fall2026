```python
import logging
from capytaine import mesh_sphere, FloatingBody, BEMSolver, DiffractionProblem, RadiationProblem, FreeSurface, animate_elevation
import numpy as np

# 1. Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# 2. Create a mesh of a sphere and floating body
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = 16
sphere_mesh = mesh_sphere(radius=radius, center=center, n_theta=resolution, n_phi=resolution)
body = FloatingBody(sphere_mesh)
body.add_translation_dof(name="Heave", direction=(0, 0, 1))

# 3. Extract the immersed part of the mesh
immersed_mesh = body.immersed_part

# 4. Set up a BEMSolver
solver = BEMSolver()

# 5. Define and solve a diffraction problem
wave_direction = 0.0  # radians
omega = 2.0  # rad/s
diffraction_problem = DiffractionProblem(
    body=immersed_mesh,
    omega=omega,
    wave_direction=wave_direction
)
diffraction_result = solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = ("Heave",)
radiation_problem = RadiationProblem(
    body=immersed_mesh,
    omega=omega,
    radiating_dof=radiating_dof
)
radiation_result = solver.solve(radiation_problem)

# 7. Define a free surface
x_range = (-5, 5)
y_range = (-5, 5)
nx = 100
ny = 100
free_surface = FreeSurface(
    x=np.linspace(*x_range, nx),
    y=np.linspace(*y_range, ny)
)

# 8. Compute the free surface elevation
diffraction_elevation = diffraction_result.get_free_surface_elevation(free_surface)
radiation_elevation = radiation_result.get_free_surface_elevation(free_surface)

# 9. Add incoming waves to the diffraction elevation
diffraction_elevation_with_incident = diffraction_elevation + diffraction_problem.incoming_wave_elevation(free_surface)

# 10. Create and run two animations
camera_position = (10, 10, 10)

# Diffraction animation
animate_elevation(
    mesh=sphere_mesh,
    free_surface=free_surface,
    elevation=diffraction_elevation_with_incident,
    face_motions=diffraction_result.body_motion,
    camera_position=camera_position,
    title="Diffraction"
)

# Radiation animation
animate_elevation(
    mesh=sphere_mesh,
    free_surface=free_surface,
    elevation=radiation_elevation,
    face_motions=radiation_result.body_motion,
    camera_position=camera_position,
    title="Radiation"
)
```