```python
from pyaedt import Hfss
from pyaedt import Desktop
import numpy as np

# Set non-graphical mode
Desktop(specified_version="2024.1", non_graphical=True, new_desktop_session=True)

# Launch AEDT in specified version and solution type
hfss = Hfss(projectname="Flex_Cable_CPWG", designname="CPWG_Flex", solution_type="DrivenModal", specified_version="2024.1")

# Set properties
hfss.material_override = False
hfss.modeler.model_units = "mm"
hfss.modeler.use_causal_materials = True
hfss.modeler.create_open_region(Frequency="10GHz", boundary="Radiation")
hfss.mesh.assign_initial_mesh_from_slider(level=4)

# Define CPWG variables
total_length = 100  # mm
theta = 45  # degrees
radius = 20  # mm
width = 2  # mm (signal line width)
height = 0.035  # mm (signal thickness)
spacing = 0.5  # mm (gap between signal and ground)
ground_width = 5  # mm (each ground width)
ground_thickness = 0.035  # mm
dielectric_height = 0.2  # mm
dielectric_material = "Polyimide"

# Function to create a bent polyline (arc + straight)
def create_bent_line(start, direction, length, theta_deg, radius, width, name):
    # Arc segment
    arc_center = [start[0], start[1] + radius, start[2]]
    arc_start = start
    arc_end = [
        start[0] + radius * np.sin(np.radians(theta_deg)),
        start[1] + radius * (1 - np.cos(np.radians(theta_deg))),
        start[2]
    ]
    arc_points = [arc_start, arc_center, arc_end]
    arc = hfss.modeler.create_polyline(arc_points, xsection_type="Rectangle", xsection_width=width, xsection_height=height, name=name+"_arc")
    # Straight segment
    straight_start = arc_end
    straight_end = [
        straight_start[0] + length * np.cos(np.radians(theta_deg)),
        straight_start[1] + length * np.sin(np.radians(theta_deg)),
        straight_start[2]
    ]
    straight = hfss.modeler.create_polyline([straight_start, straight_end], xsection_type="Rectangle", xsection_width=width, xsection_height=height, name=name+"_straight")
    return [arc, straight]

# Draw signal line (center)
signal_start = [0, 0, dielectric_height + height/2]
signal = create_bent_line(signal_start, [1, 0, 0], total_length - radius * np.radians(theta), theta, radius, width, "Signal")

# Draw ground lines (left and right)
gnd_offset = width/2 + spacing + ground_width/2
gnd1_start = [0, -gnd_offset, dielectric_height + ground_thickness/2]
gnd2_start = [0, gnd_offset, dielectric_height + ground_thickness/2]
gnd1 = create_bent_line(gnd1_start, [1, 0, 0], total_length - radius * np.radians(theta), theta, radius, ground_width, "GND1")
gnd2 = create_bent_line(gnd2_start, [1, 0, 0], total_length - radius * np.radians(theta), theta, radius, ground_width, "GND2")

# Draw dielectric
dielectric_length = total_length + 5
dielectric_width = 2 * (gnd_offset + ground_width/2) + 1
dielectric = hfss.modeler.create_box(
    [-2, -dielectric_width/2, 0],
    [dielectric_length, dielectric_width, dielectric_height],
    name="Dielectric"
)
hfss.modeler.assign_material(dielectric, dielectric_material)

# Create bottom metals (signal and grounds mirrored below)
signal_bottom = hfss.modeler.create_box(
    [0, -width/2, 0],
    [total_length, width, height],
    name="Signal_Bottom"
)
hfss.modeler.assign_material(signal_bottom, "copper")
gnd1_bottom = hfss.modeler.create_box(
    [0, -gnd_offset - ground_width/2, 0],
    [total_length, ground_width, ground_thickness],
    name="GND1_Bottom"
)
hfss.modeler.assign_material(gnd1_bottom, "copper")
gnd2_bottom = hfss.modeler.create_box(
    [0, gnd_offset - ground_width/2, 0],
    [total_length, ground_width, ground_thickness],
    name="GND2_Bottom"
)
hfss.modeler.assign_material(gnd2_bottom, "copper")

# Create port interfaces (PEC enclosures)
pec1 = hfss.modeler.create_box(
    [-2.5, -dielectric_width/2, 0],
    [1, dielectric_width, dielectric_height + height],
    name="PEC1"
)
hfss.modeler.assign_material(pec1, "pec")
pec2 = hfss.modeler.create_box(
    [total_length + 1.5, -dielectric_width/2, 0],
    [1, dielectric_width, dielectric_height + height],
    name="PEC2"
)
hfss.modeler.assign_material(pec2, "pec")

# Assign Perfect E boundary to enclosures
hfss.assign_perfecte_to_sheets([pec1.faces[4], pec2.faces[5]])

# Create wave ports
port1 = hfss.create_wave_port_from_face(pec1.faces[4], 1)
port2 = hfss.create_wave_port_from_face(pec2.faces[5], 2)

# Create setup and sweep
setup = hfss.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.props["PortsOnly"] = False
setup.props["MaximumPasses"] = 15
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 1
setup.props["PercentRefinement"] = 30
setup.props["BasisOrder"] = 1
setup.update()

sweep = setup.add_sweep("Sweep1")
sweep.props["RangeType"] = "LinearStep"
sweep.props["RangeStart"] = "1GHz"
sweep.props["RangeEnd"] = "20GHz"
sweep.props["RangeStep"] = "0.5GHz"
sweep.props["Type"] = "Fast"
sweep.update()

# Plot the model
hfss.plot(show=True)

# Release AEDT
hfss.release_desktop(close_projects=True, close_desktop=True)
```