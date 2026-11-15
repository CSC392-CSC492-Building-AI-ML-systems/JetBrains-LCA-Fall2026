```python
import pybamm

# Create a custom lithium-ion model
class MyLiIonModel(pybamm.lithium_ion.BaseModel):
    def __init__(self):
        super().__init__(name="my li-ion model")

        # External circuit
        self.set_external_circuit_submodel()

        # Current collector
        self.set_current_collector_submodel()

        # Thermal
        self.set_thermal_submodel()

        # Porosity
        self.set_porosity_submodel()

        # Electrolyte diffusion
        self.set_electrolyte_diffusion_submodel()

        # Electrolyte conductivity
        self.set_electrolyte_conductivity_submodel()

        # SEI
        self.set_sei_submodel()

        # SEI on cracks
        self.set_sei_on_cracks_submodel()

        # Lithium plating
        self.set_lithium_plating_submodel()

        # Negative electrode submodels
        for domain in ["negative", "positive"]:
            self.set_active_material_submodel(domain)
            self.set_electrode_potential_submodel(domain)
            self.set_particle_submodel(domain)
            self.set_total_particle_concentration_submodel(domain)
            self.set_open_circuit_potential_submodel(domain)
            self.set_interface_submodel(domain)
            self.set_interface_utilisation_submodel(domain)
            self.set_interface_current_submodel(domain)
            self.set_surface_potential_difference_submodel(domain)
            self.set_particle_mechanics_submodel(domain)

# Instantiate the model
model = MyLiIonModel()

# Build the model
model.build_model()

# Create geometry
geometry = pybamm.geometry.lithium_ion.DFN_geometry()

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
solver = pybamm.CasadiSolver()
solution = solver.solve(model, t_eval)

# Plot the solution dynamically
pybamm.dynamic_plot(solution)
```