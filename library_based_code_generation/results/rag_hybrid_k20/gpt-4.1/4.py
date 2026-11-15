```python
import logging
import numpy as np
import capytaine as cpt
from capytaine.io.xarray import free_surface_to_mesh
from capytaine.visualization import animate_free_surface

# 1. Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# 2. Create a mesh of a sphere and floating body
radius = 1.0
center = (0.0, 0.0, 0.0)
resolution = 32
sphere_mesh = cpt.meshes.mesh_sphere(radius=radius, center=center, n_theta=resolution, n_phi=resolution)
body = cpt.FloatingBody(sphere_mesh, name="sphere")
body.add_translation_dof(name="Heave", direction=(0, 0, 1), center=center)

# 3. Extract the immersed part of the mesh
immersed_mesh = body.immersed_part

# 4. Set up a BEMSolver
solver = cpt.BEMSolver()

# 5. Define and solve a diffraction problem
omega = 2.0
wave_direction = 0.0  # radians
diffraction_problem = cpt.DiffractionProblem(
    body=immersed_mesh,
    omega=omega,
    wave_direction=wave_direction
)
diffraction_result = solver.solve(diffraction_problem)

# 6. Define and solve a radiation problem
radiating_dof = ("sphere", "Heave")
radiation_problem = cpt.RadiationProblem(
    body=immersed_mesh,
    dof_name=radiating_dof[1],
    omega=omega
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
free_surface = {"x": X, "y": Y}

# 8. Compute the free surface elevation
diffraction_eta = cpt.post_pro.free_surface.compute_free_surface_elevation(
    diffraction_result, X, Y, t=0.0
)
radiation_eta = cpt.post_pro.free_surface.compute_free_surface_elevation(
    radiation_result, X, Y, t=0.0
)

# 9. Add incoming waves to the diffraction elevation
incoming_eta = cpt.post_pro.free_surface.airy_waves_free_surface_elevation(
    amplitude=1.0,
    omega=omega,
    wave_direction=wave_direction,
    x=X,
    y=Y,
    t=0.0
)
diffraction_eta_total = diffraction_eta + incoming_eta

# 10. Create and run two animations
camera_position = (10, 10, 10)

# Animation for diffraction
animate_free_surface(
    mesh=sphere_mesh,
    free_surface=(X, Y, diffraction_eta_total),
    face_motions={"sphere": ("Heave", 1.0)},
    camera_position=camera_position,
    title="Diffraction Free Surface Elevation"
)

# Animation for radiation
animate_free_surface(
    mesh=sphere_mesh,
    free_surface=(X, Y, radiation_eta),
    face_motions={"sphere": ("Heave", 1.0)},
    camera_position=camera_position,
    title="Radiation Free Surface Elevation"
)
```