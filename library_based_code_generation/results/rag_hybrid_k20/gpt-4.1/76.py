```python
import pybamm

# Define model options
options = {
    "external circuit": "explicit",
    "current collector": "potential pair",
    "thermal": "lumped",
    "porosity": "true",
    "electrolyte diffusion": "full",
    "electrolyte conductivity": "full",
    "SEI": "ec reaction limited",
    "SEI on cracks": "true",
    "lithium plating": "reversible",
    "particle mechanics": "swelling and cracking",
    "interface utilisation": "true",
}

# Create the custom model
model = pybamm.lithium_ion.BaseModel(options=options)
model.name = "my li-ion model"

# Build the model
model.build_model()

# Create geometry
geometry = pybamm.geometry.get_geometry(model)

# Process model and geometry
param = pybamm.ParameterValues("Chen2020")
param.process_model(model)
param.process_geometry(geometry)

# Set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# Discretise the model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# Solve the model
t_eval = pybamm.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver(mode="fast")
solution = solver.solve(model, t_eval)

# Plot the solution dynamically
pybamm.dynamic_plot(solution)
```