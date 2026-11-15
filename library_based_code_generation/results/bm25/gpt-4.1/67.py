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
app = Hfss3dLayout(projectname=project_name, designname=design_name, solution_type="2D Extractor")

# Define variables
app["cpw_width"] = "2mm"
app["cpw_gap"] = "0.5mm"
app["cpw_length"] = "20mm"
app["ground_width"] = "10mm"
app["dielectric_thickness"] = "0.8mm"
app["conformal_thickness"] = "0.05mm"
app["dielectric_material"] = "FR4_epoxy"
app["conformal_material"] = "ConformalCoating"
app["signal_thickness"] = "0.035mm"

# Create primitives
origin = [0, 0]
signal_trace = app.modeler.create_rectangle(
    position=origin,
    dimension_list=["cpw_length", "cpw_width"],
    name="Signal",
    matname="copper"
)

left_ground = app.modeler.create_rectangle(
    position=[0, f"-({app['ground_width']}/2+cpw_gap+cpw_width/2)"],
    dimension_list=["cpw_length", "cpw_gap"],
    name="LeftGround",
    matname="copper"
)

right_ground = app.modeler.create_rectangle(
    position=[0, f"({app['cpw_width']}/2+cpw_gap)"],
    dimension_list=["cpw_length", "cpw_gap"],
    name="RightGround",
    matname="copper"
)

# Create a reference ground plane
ref_ground = app.modeler.create_rectangle(
    position=[0, f"-({app['ground_width']}/2)"],
    dimension_list=["cpw_length", "ground_width"],
    name="RefGround",
    matname="copper"
)

# Create a dielectric
dielectric = app.modeler.create_rectangle(
    position=[0, f"-({app['ground_width']}/2)"],
    dimension_list=["cpw_length", "ground_width"],
    name="Dielectric",
    matname=app["dielectric_material"]
)
app.modeler.stackup.layers[2].thickness = app["dielectric_thickness"]

# Create a conformal coating
conformal = app.modeler._create_conformal(
    layer_name="TOP",
    thickness=app["conformal_thickness"],
    material=app["conformal_material"]
)

# Assign conductor to the signal
app.assign_single_conductor("Signal", signal_type="Signal")

# Create a reference ground
app.assign_single_conductor("RefGround", signal_type="ReferenceGround")

# Assign the Huray model on the signal
app.assign_huray_surface("Signal", radius=0.5e-6, ratio=2.0, surface_name="Signal")

# Create the setup
setup = app.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.update()

# Analyze it
app.analyze_setup("Setup1")

# Plot solution data
solution_data = app.post.get_solution_data(expressions=["Z0"], setup_sweep_name="Setup1 : LastAdaptive")
solution_data.plot("Z0")

# Save the project and close AEDT
app.save_project()
app.release_desktop(close_projects=True, close_desktop=True)
```