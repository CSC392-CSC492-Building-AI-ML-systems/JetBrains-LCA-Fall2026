```python
import os
import tempfile
from pyaedt import Desktop, downloads, Hfss, Circuit, Mechanical

# Set AEDT graphical mode
desktop_version = "2023.2"
non_graphical = False

# Download example project
project_url = "https://github.com/pyansys/pyaedt-assets/releases/download/v0.6.0/antenna_array.aedt"
project_path = os.path.join(tempfile.gettempdir(), "antenna_array.aedt")
if not os.path.exists(project_path):
    downloads.download_file(project_url, destination_folder=tempfile.gettempdir())

# Launch AEDT Desktop
d = Desktop(desktop_version, non_graphical=non_graphical, new_desktop_session=True)

# Open the downloaded project
d.open_project(project_path)
project_name = os.path.splitext(os.path.basename(project_path))[0]

# Start HFSS
hfss = Hfss(projectname=project_name, specified_version=desktop_version, designname="HFSSDesign1", solution_type="DrivenModal")

# Start Circuit and add HFSS dynamic link component
circuit = Circuit(projectname=project_name, specified_version=desktop_version, designname="CircuitDesign1")
hfss_comp = circuit.modeler.schematic.add_dynamic_link(hfss, setupname=None)

# Set up dynamic link options
hfss_comp.parameters["UsePortImpedance"] = True
hfss_comp.parameters["PreserveExcitations"] = True

# Create ports and excitations in Circuit
port1 = circuit.modeler.schematic.create_port("Port1", pos_x=0, pos_y=0)
port2 = circuit.modeler.schematic.create_port("Port2", pos_x=2, pos_y=0)
circuit.modeler.schematic.create_wire(port1, hfss_comp.pins[0])
circuit.modeler.schematic.create_wire(port2, hfss_comp.pins[1])

# Create a setup and solve the circuit
setup_circuit = circuit.create_setup("CircuitSetup1")
setup_circuit.props["Frequency"] = "10GHz"
circuit.analyze_setup("CircuitSetup1")

# Push excitations to the HFSS model
circuit.push_excitation_to_hfss(hfss)

# Start Mechanical and copy bodies from HFSS
mechanical = Mechanical(projectname=project_name, specified_version=desktop_version, designname="MechanicalDesign1")
mechanical.copy_solid_bodies_from(hfss)

# Get losses from HFSS and assign convection to Mechanical
losses = hfss.post.get_loss("Setup1", "LastAdaptive")
mechanical.assign_convection_to_faces(faces=None, convection_value=10.0)
mechanical.assign_surface_heat_flux_to_faces(faces=None, heat_flux=losses)

# Plot the model
mechanical.plot(show=True)

# Solve and plot the thermal results
mechanical.analyze_setup("Setup1")
mechanical.post.plot_contour(plot_type="Temperature", show=True)

# Release AEDT
d.release_desktop(close_projects=True, close_desktop=True)
```