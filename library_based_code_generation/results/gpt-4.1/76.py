```python
import pybamm

# Create a custom model
model = pybamm.BaseModel(name="my li-ion model")

# Parameter values
param = pybamm.ParameterValues("Chen2020")

# Submodels for external circuit, current collector, thermal, porosity, electrolyte
model.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(model)
model.submodels["current collector"] = pybamm.current_collector.Uniform(model)
model.submodels["thermal"] = pybamm.thermal.lumped.Lumped(model)
model.submodels["porosity"] = pybamm.porosity.Full(model)
model.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(model)
model.submodels["electrolyte conductivity"] = pybamm.electrolyte_conductivity.Full(model)
model.submodels["SEI"] = pybamm.sei.Constant(model, "Negative")
model.submodels["SEI on cracks"] = pybamm.sei.SEIOnCracks(model, "Negative")
model.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating(model, "Negative")

# Negative electrode submodels
neg_domain = "Negative"
model.submodels["negative active material"] = pybamm.active_material.Constant(model, neg_domain)
model.submodels["negative electrode potential"] = pybamm.electrode_potential.Full(model, neg_domain)
model.submodels["negative particle"] = pybamm.particle.FDdiffusion(model, neg_domain)
model.submodels["negative total particle concentration"] = pybamm.total_particle_concentration.Full(model, neg_domain)
model.submodels["negative open-circuit potential"] = pybamm.open_circuit_potential.Full(model, neg_domain)
model.submodels["negative interface"] = pybamm.interface.InverseButlerVolmer(model, neg_domain)
model.submodels["negative interface utilisation"] = pybamm.interface_utilisation.Constant(model, neg_domain)
model.submodels["negative interface current"] = pybamm.interface_current.Full(model, neg_domain)
model.submodels["negative surface potential difference"] = pybamm.surface_potential_difference.Full(model, neg_domain)
model.submodels["negative particle mechanics"] = pybamm.particle_mechanics.NoMechanics(model, neg_domain)

# Positive electrode submodels
pos_domain = "Positive"
model.submodels["positive active material"] = pybamm.active_material.Constant(model, pos_domain)
model.submodels["positive electrode potential"] = pybamm.electrode_potential.Full(model, pos_domain)
model.submodels["positive particle"] = pybamm.particle.FDdiffusion(model, pos_domain)
model.submodels["positive total particle concentration"] = pybamm.total_particle_concentration.Full(model, pos_domain)
model.submodels["positive open-circuit potential"] = pybamm.open_circuit_potential.Full(model, pos_domain)
model.submodels["positive interface"] = pybamm.interface.InverseButlerVolmer(model, pos_domain)
model.submodels["positive interface utilisation"] = pybamm.interface_utilisation.Constant(model, pos_domain)
model.submodels["positive interface current"] = pybamm.interface_current.Full(model, pos_domain)
model.submodels["positive surface potential difference"] = pybamm.surface_potential_difference.Full(model, pos_domain)
model.submodels["positive particle mechanics"] = pybamm.particle_mechanics.NoMechanics(model, pos_domain)

# Build the model
model.build_model()

# Create geometry
geometry = pybamm.Geometry("1D macro", "1D micro")

# Process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# Set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# Discretise the model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# Solve the model
t_eval = pybamm.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver()
solution = solver.solve(model, t_eval)

# Plot the solution dynamically
pybamm.dynamic_plot(solution)
```