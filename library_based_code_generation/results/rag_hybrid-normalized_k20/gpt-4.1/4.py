```python
import logging
import numpy as np
import capytaine as cpt
from capytaine.meshes.geometry import mesh_sphere
from capytaine.bem.solver import BEMSolver
from capytaine.post_pro.airy_waves import airy_waves_free_surface_elevation_at_meshgrid
from capytaine.post_pro.free_surface import compute_free_surface_elevation_at_meshgrid
from capytaine.visualization.animation import animate_free_surface

# 1. Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# 2. Create a mesh of a sphere and floating body
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = 32
sphere_mesh = mesh_sphere(radius=radius, center=center, n_theta=resolution, n_phi=resolution)
body = cpt.FloatingBody(sphere_mesh, name="sphere")
body.add_translation_dof(name="Heave", direction=(0, 0, 1))

# 3. Extract the immersed part of the mesh
immersed_mesh = body.immersed_part()

# 4. Set up a BEMSolver
solver = BEMSolver()

# 5. Define and solve a diffraction problem
omega = 2.0  # rad/s
wave_direction = (1.0, 0.0)
diffraction_problem = cpt.DiffractionProblem(
    body=immersed_mesh,
    omega=omega,
    wave_direction=wave_direction
)
diffraction_result = solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = "Heave"
radiation_problem = cpt.RadiationProblem(
    body=immersed_mesh,
    omega=omega,
    radiating_dof=radiating_dof
)
radiation_result = solver.solve(radiation_problem)

# 7. Define a free surface
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
free_surface = {"X": X, "Y": Y}

# 8. Compute the free surface elevation
diffraction_eta = compute_free_surface_elevation_at_meshgrid(
    diffraction_result, X, Y, t=0.0
)
radiation_eta = compute_free_surface_elevation_at_meshgrid(
    radiation_result, X, Y, t=0.0
)

# 9. Add incoming waves to the diffraction elevation
incoming_eta = airy_waves_free_surface_elevation_at_meshgrid(
    omega=omega,
    wave_direction=wave_direction,
    X=X,
    Y=Y,
    t=0.0
)
total_diffraction_eta = diffraction_eta + incoming_eta

# 10. Create and run two animations
camera_position = (10, 10, 10)

# Diffraction animation
animate_free_surface(
    mesh=sphere_mesh,
    free_surface=(X, Y, total_diffraction_eta),
    face_motions={"Heave": 1.0},
    camera_position=camera_position,
    title="Diffraction Free Surface Animation"
)

# Radiation animation
animate_free_surface(
    mesh=sphere_mesh,
    free_surface=(X, Y, radiation_eta),
    face_motions={"Heave": 1.0},
    camera_position=camera_position,
    title="Radiation Free Surface Animation"
)
```