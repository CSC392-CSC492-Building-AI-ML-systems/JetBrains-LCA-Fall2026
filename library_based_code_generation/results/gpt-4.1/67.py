```python
from pyaedt import Hfss3dLayout
from pyaedt import Desktop
import os

# Set non-graphical mode
non_graphical = True

# Launch AEDT and 2D Extractor
desktop_version = "2024.1"
project_name = "CPWG_2D_Extractor"
design_name = "CPWG_Design"

Desktop(desktop_version, non_graphical=non_graphical, new_desktop_session=True)
edb = Hfss3dLayout(projectname=project_name, designname=design_name, solution_type="2D Extractor")

# Define variables
edb["cpw_width"] = "0.5mm"
edb["cpw_gap"] = "0.25mm"
edb["cpw_length"] = "10mm"
edb["gnd_width"] = "2mm"
edb["dielectric_thickness"] = "0.8mm"
edb["conformal_thickness"] = "0.05mm"

# Create primitives
origin = [0, 0]
signal_trace = edb.modeler.create_trace(
    [origin, [edb["cpw_length"], 0]],
    width=edb["cpw_width"],
    layer_name="TOP"
)

# Create coplanar ground
left_gnd = edb.modeler.create_rectangle(
    "TOP",
    [origin[0], origin[1] - float(str(edb["cpw_gap"])) - float(str(edb["cpw_width"])) / 2 - float(str(edb["gnd_width"]))],
    float(str(edb["cpw_length"])),
    float(str(edb["gnd_width"]))
)
right_gnd = edb.modeler.create_rectangle(
    "TOP",
    [origin[0], origin[1] + float(str(edb["cpw_gap"])) + float(str(edb["cpw_width"])) / 2],
    float(str(edb["cpw_length"])),
    float(str(edb["gnd_width"]))
)

# Create reference ground plane
ref_gnd = edb.modeler.create_rectangle(
    "BOTTOM",
    [origin[0], origin[1] - 3 * float(str(edb["gnd_width"]))],
    float(str(edb["cpw_length"])),
    6 * float(str(edb["gnd_width"]))
)

# Create dielectric
edb.stackup.add_layer(
    "DIELECTRIC",
    thickness=edb["dielectric_thickness"],
    material="FR4_epoxy",
    fill_material="FR4_epoxy",
    is_dielectric=True,
    below_layer="TOP"
)

# Create conformal coating
edb.stackup.add_layer(
    "COATING",
    thickness=edb["conformal_thickness"],
    material="SolderMask",
    fill_material="SolderMask",
    is_dielectric=True,
    below_layer="DIELECTRIC"
)

# Assign conductor to the signal
edb.assign_material(signal_trace, "copper")

# Create reference ground
edb.assign_material(ref_gnd, "copper")
edb.assign_material(left_gnd, "copper")
edb.assign_material(right_gnd, "copper")

# Assign Huray model on the signal
edb.assign_huray_model(signal_trace, radius=0.5e-6, ratio=2.0)

# Create the setup
setup = edb.create_setup("CPWG_Setup")
setup.props["Frequency"] = "10GHz"
setup.update()

# Analyze
edb.analyze()

# Plot solution data
solution_data = edb.get_solution_data(expressions=["Z0"], setup_sweep_name="CPWG_Setup : LastAdaptive")
solution_data.plot("Z0")

# Save the project and close AEDT
edb.save_project()
edb.release_desktop(close_projects=True, close_desktop=True)
```