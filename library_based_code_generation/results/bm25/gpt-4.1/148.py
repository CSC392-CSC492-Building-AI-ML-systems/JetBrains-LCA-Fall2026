```python
import os
from pyaedt import Desktop, downloads, Hfss, Circuit, Mechanical

# Set AEDT graphical mode
desktop_version = "2023.2"
non_graphical = False

# Launch AEDT Desktop
d = Desktop(desktop_version, non_graphical=non_graphical, new_desktop_session=True)

# Download and open a sample project
project_url = "https://github.com/ansys/pyaedt-assets/releases/download/v0.6.0/antenna_array.aedt"
project_path = downloads.download_file(project_url, "antenna_array.aedt")
project_dir = os.path.dirname(project_path)
project_name = os.path.basename(project_path)
d.open_project(project_path)

# Start HFSS
hfss = Hfss(projectname=project_name, specified_version=desktop_version, designname="HFSSDesign1", solution_type="DrivenModal")

# Start Circuit
circuit = Circuit(projectname=project_name, specified_version=desktop_version, designname="CircuitDesign1")

# Add HFSS dynamic link component to Circuit
hfss_dynamic_link = circuit.modeler.schematic.add_dynamic_link(hfss, solution_name="Setup1 : LastAdaptive", name="HFSS_Link")

# Set up dynamic link options
hfss_dynamic_link.set_dynamic_link_options(
    sweep_type="Discrete",
    interpolation=False,
    use_ports_only=True,
    use_causality=True,
    enforce_passivity=True
)

# Create ports and excitations in Circuit
port1 = circuit.modeler.schematic.create_port("Port1", pos_x=0, pos_y=0)
port2 = circuit.modeler.schematic.create_port("Port2", pos_x=2, pos_y=0)
circuit.modeler.schematic.connect_components(port1, hfss_dynamic_link.pins[0])
circuit.modeler.schematic.connect_components(port2, hfss_dynamic_link.pins[1])

# Create a setup and solve the circuit, push excitations to HFSS
setup_circuit = circuit.create_setup("CircuitSetup1")
setup_circuit.props["SweepDefinition"] = "LIN 1GHz 10GHz 1GHz"
setup_circuit.update()
circuit.analyze_setup("CircuitSetup1")
hfss.push_excitation_from_circuit()

# Start Mechanical and copy bodies from HFSS
mechanical = Mechanical(projectname=project_name, specified_version=desktop_version, designname="MechanicalDesign1")
mechanical.copy_bodies_from(hfss)

# Get losses from HFSS and assign convection to Mechanical
losses = hfss.get_loss("Setup1", "LastAdaptive")
mechanical.assign_convection_to_faces(faces=None, convection_value=10.0)
mechanical.assign_losses(losses)

# Plot the model
mechanical.plot(show=True)

# Solve and plot the thermal results
mechanical.analyze()
mechanical.plot_thermal_results(quantity="Temperature", show=True)

# Release AEDT
d.release_desktop(close_projects=True, close_desktop=True)
```