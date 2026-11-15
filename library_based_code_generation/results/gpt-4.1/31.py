```python
import pyaedt
from pyaedt import Hfss
from pyaedt.modeler.geometry_operators import GeometryOperators
from pyaedt.generic.constants import unit_converter
import math

# Set non-graphical mode
pyaedt.settings.non_graphical = True

# Launch AEDT
aedt_version = "2023.2"
project_name = "Flex_CPWG"
design_name = "CPWG_Flex"
solution_type = "DrivenModal"

hfss = Hfss(projectname=project_name, designname=design_name, specified_version=aedt_version, solution_type=solution_type, new_desktop_session=True)

# Set properties
hfss.materials.material_override = True
hfss.materials.automatic_use_causal_materials = True
hfss.modeler.model_units = "mm"
hfss.modeler.create_open_region(Frequency="10GHz", boundary="radiation")
hfss.mesh.assign_initial_mesh_settings(max_length=2, min_length=0.1)

# Define CPWG variables
total_length = 100  # mm
theta = 60  # degrees, bending angle
radius = 30  # mm, bending radius
width = 2  # mm, signal width
height = 0.035  # mm, metal thickness
spacing = 0.3  # mm, gap between signal and ground
gnd_width = 4  # mm, ground width
gnd_thickness = 0.035  # mm, ground thickness
dielectric_height = 0.2  # mm
dielectric_material = "Polyimide"

# Helper function to create a bent path
def create_bent_path(start, length, theta_deg, radius, direction="X"):
    theta_rad = math.radians(theta_deg)
    arc_length = radius * theta_rad
    straight_length = length - arc_length
    points = []
    # Start point
    points.append(start)
    # First straight segment
    if direction == "X":
        p1 = [start[0] + straight_length, start[1], start[2]]
    else:
        p1 = [start[0], start[1] + straight_length, start[2]]
    points.append(p1)
    # Arc segment
    center = [p1[0], p1[1] + radius, p1[2]]
    arc = hfss.modeler.create_arc(center, radius, 270, 270 + theta_deg, axis="Z", num_segments=20)
    return points, arc

# Draw signal line
signal_start = [0, 0, dielectric_height + height / 2]
signal_points, signal_arc = create_bent_path(signal_start, total_length, theta, radius, direction="X")
signal_line = hfss.modeler.create_polyline(position_list=signal_points, xsection_type="Rectangle", xsection_width=width, xsection_height=height, name="Signal_Line")
signal_arc_clone = signal_arc.clone()
signal_arc_clone.move([signal_points[-1][0] - signal_arc_clone.bounding_box[0][0], signal_points[-1][1] - signal_arc_clone.bounding_box[0][1], 0])
signal_line = hfss.modeler.unite([signal_line, signal_arc_clone])

# Draw ground lines (left and right)
gnd_offset = width / 2 + spacing + gnd_width / 2
gnd1_start = [0, gnd_offset, dielectric_height + gnd_thickness / 2]
gnd2_start = [0, -gnd_offset, dielectric_height + gnd_thickness / 2]
gnd1_points, gnd1_arc = create_bent_path(gnd1_start, total_length, theta, radius, direction="X")
gnd2_points, gnd2_arc = create_bent_path(gnd2_start, total_length, theta, radius, direction="X")
gnd1_line = hfss.modeler.create_polyline(position_list=gnd1_points, xsection_type="Rectangle", xsection_width=gnd_width, xsection_height=gnd_thickness, name="GND1_Line")
gnd2_line = hfss.modeler.create_polyline(position_list=gnd2_points, xsection_type="Rectangle", xsection_width=gnd_width, xsection_height=gnd_thickness, name="GND2_Line")
gnd1_arc_clone = gnd1_arc.clone()
gnd1_arc_clone.move([gnd1_points[-1][0] - gnd1_arc_clone.bounding_box[0][0], gnd1_points[-1][1] - gnd1_arc_clone.bounding_box[0][1], 0])
gnd2_arc_clone = gnd2_arc.clone()
gnd2_arc_clone.move([gnd2_points[-1][0] - gnd2_arc_clone.bounding_box[0][0], gnd2_points[-1][1] - gnd2_arc_clone.bounding_box[0][1], 0])
gnd1_line = hfss.modeler.unite([gnd1_line, gnd1_arc_clone])
gnd2_line = hfss.modeler.unite([gnd2_line, gnd2_arc_clone])

# Draw dielectric
dielectric_length = total_length + 10
dielectric_width = 2 * (gnd_offset + gnd_width / 2) + 2
dielectric_box = hfss.modeler.create_box([0, -dielectric_width / 2, 0], [dielectric_length, dielectric_width, dielectric_height], name="Dielectric")
hfss.modeler.assign_material(dielectric_box, dielectric_material)

# Draw bottom metals (ground plane)
bottom_gnd = hfss.modeler.create_box([0, -dielectric_width / 2, 0], [dielectric_length, dielectric_width, gnd_thickness], name="Bottom_GND")
hfss.modeler.assign_material(bottom_gnd, "copper")

# Create port interfaces (PEC enclosures)
pec_box1 = hfss.modeler.create_box([-2, -dielectric_width / 2, 0], [2, dielectric_width, dielectric_height + height], name="PEC1")
pec_box2 = hfss.modeler.create_box([dielectric_length, -dielectric_width / 2, 0], [2, dielectric_width, dielectric_height + height], name="PEC2")
hfss.modeler.assign_material(pec_box1, "pec")
hfss.modeler.assign_material(pec_box2, "pec")

# Assign Perfect E boundary
hfss.assign_perfecte([pec_box1, pec_box2])

# Create wave ports
port1_face = hfss.modeler.get_faceid_from_position([-1, 0, dielectric_height / 2])
port2_face = hfss.modeler.get_faceid_from_position([dielectric_length + 1, 0, dielectric_height / 2])
hfss.create_wave_port_from_face(port1_face, 1)
hfss.create_wave_port_from_face(port2_face, 2)

# Create setup and sweep
setup = hfss.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.props["MaximumPasses"] = 10
setup.props["MinimumPasses"] = 1
setup.props["MinimumConvergedPasses"] = 1
setup.props["PercentRefinement"] = 30
setup.props["BasisOrder"] = 1
setup.update()
hfss.create_linear_count_sweep(setupname="Setup1", unit="GHz", freqstart=1, freqstop=20, num_of_freq_points=201, sweep_type="Interpolating")

# Plot model
hfss.plot(show=True)

# Release AEDT
hfss.release_desktop(close_projects=True, close_desktop=True)
```