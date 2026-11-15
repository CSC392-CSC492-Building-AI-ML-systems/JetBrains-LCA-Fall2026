```python
import logging
import numpy as np
import capytaine as cpt

# 1. Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# 2. Create a mesh of a sphere and floating body
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = 16
sphere_mesh = cpt.meshes.mesh_sphere(radius=radius, center=center, n_theta=resolution, n_phi=resolution)
sphere_body = cpt.FloatingBody(sphere_mesh)
sphere_body.add_translation_dof(name="Heave", direction=(0, 0, 1))

# 3. Extract the immersed part of the mesh
immersed_mesh = sphere_mesh.immersed_part()

# 4. Set up a BEMSolver
bem_solver = cpt.BEMSolver()

# 5. Define and solve a diffraction problem
wave_direction = 0.0  # radians
omega = 2 * np.pi  # rad/s
diffraction_problem = cpt.DiffractionProblem(
    body=cpt.FloatingBody(immersed_mesh),
    omega=omega,
    wave_direction=wave_direction
)
diffraction_result = bem_solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = ("Heave",)
radiation_problem = cpt.RadiationProblem(
    body=cpt.FloatingBody(immersed_mesh),
    omega=omega,
    radiating_dof=radiating_dof
)
radiation_result = bem_solver.solve(radiation_problem)

# 7. Define a free surface
x_range = (-5, 5)
y_range = (-5, 5)
nx = 100
ny = 100
x = np.linspace(*x_range, nx)
y = np.linspace(*y_range, ny)
free_surface = cpt.FreeSurface(x, y)

# 8. Compute the free surface elevation
diffraction_elevation = diffraction_result.get_free_surface_elevation(free_surface)
radiation_elevation = radiation_result.get_free_surface_elevation(free_surface)

# 9. Add incoming waves to the diffraction elevation
incoming_wave = diffraction_problem.incoming_waves_on_free_surface(free_surface)
total_diffraction_elevation = diffraction_elevation + incoming_wave

# 10. Create and run two animations
camera_position = (10, 10, 5)

# Diffraction animation
cpt.animation.animate_elevation_and_body(
    body=sphere_body,
    elevation=total_diffraction_elevation,
    face_motions={"Heave": 1.0},
    camera_position=camera_position,
    title="Diffraction Animation"
)

# Radiation animation
cpt.animation.animate_elevation_and_body(
    body=sphere_body,
    elevation=radiation_elevation,
    face_motions={"Heave": 1.0},
    camera_position=camera_position,
    title="Radiation Animation"
)
```