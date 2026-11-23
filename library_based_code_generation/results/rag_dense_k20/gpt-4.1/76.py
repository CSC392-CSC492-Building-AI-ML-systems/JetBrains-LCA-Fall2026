```python
import pybamm

# Create a custom lithium-ion model
model = pybamm.BaseBatteryModel(name="my li-ion model")

# Parameters
param = pybamm.ParameterValues("Chen2020")

# External circuit
model.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(model.param)

# Current collector
model.submodels["current collector"] = pybamm.current_collector.Uniform(model.param)

# Thermal
model.submodels["thermal"] = pybamm.thermal.x_averaged.Lumped(model.param, "lithium-ion")

# Porosity
model.submodels["porosity"] = pybamm.porosity.Full(model.param, "lithium-ion")

# Electrolyte diffusion
model.submodels["electrolyte diffusion"] = pybamm.electrolyte.diffusion.Full(model.param, "lithium-ion")

# Electrolyte conductivity
model.submodels["electrolyte conductivity"] = pybamm.electrolyte.conductivity.Full(model.param, "lithium-ion")

# SEI
model.submodels["SEI"] = pybamm.sei.BaseModel(model.param, "lithium-ion")

# SEI on cracks
model.submodels["SEI on cracks"] = pybamm.sei.SEIOnCracks(model.param, "lithium-ion")

# Lithium plating
model.set_lithium_plating_submodel()

# Negative electrode submodels
for submodel_name, submodel_class in [
    ("active material", pybamm.active_material.Full),
    ("electrode potential", pybamm.electrode.potential.Full),
    ("particle", pybamm.particle.FDdiffusion),
    ("total particle concentration", pybamm.particle.TotalConcentration),
    ("open-circuit potential", pybamm.open_circuit_potential.Full),
    ("interface", pybamm.interface.Full),
    ("interface utilisation", pybamm.interface.InterfaceUtilisation),
    ("interface current", pybamm.interface.InterfaceCurrent),
    ("surface potential difference", pybamm.interface.SurfacePotentialDifference),
    ("particle mechanics", pybamm.particle_mechanics.BaseModel),
]:
    model.submodels[f"negative {submodel_name}"] = submodel_class(model.param, "Negative")

# Positive electrode submodels
for submodel_name, submodel_class in [
    ("active material", pybamm.active_material.Full),
    ("electrode potential", pybamm.electrode.potential.Full),
    ("particle", pybamm.particle.FDdiffusion),
    ("total particle concentration", pybamm.particle.TotalConcentration),
    ("open-circuit potential", pybamm.open_circuit_potential.Full),
    ("interface", pybamm.interface.Full),
    ("interface utilisation", pybamm.interface.InterfaceUtilisation),
    ("interface current", pybamm.interface.InterfaceCurrent),
    ("surface potential difference", pybamm.interface.SurfacePotentialDifference),
    ("particle mechanics", pybamm.particle_mechanics.BaseModel),
]:
    model.submodels[f"positive {submodel_name}"] = submodel_class(model.param, "Positive")

# Build the model
model.build_model()

# Create geometry
geometry = pybamm.geometry.lithium_ion_geometry()

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