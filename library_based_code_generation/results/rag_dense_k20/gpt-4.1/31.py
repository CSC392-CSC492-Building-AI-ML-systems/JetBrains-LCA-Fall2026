```python
import pyaedt
from pyaedt import Hfss
from pyaedt.generic.constants import unit_converter
from pyaedt.modeler.geometry_operators import GeometryOperators
import math

# Set non-graphical mode
pyaedt.settings.non_graphical = True

# Launch AEDT
aedt_version = "2024.1"
project_name = "Flex_Cable_CPWG"
design_name = "CPWG_Flex"
solution_type = "DrivenModal"

hfss = Hfss(projectname=project_name, designname=design_name, solution_type=solution_type, specified_version=aedt_version)

# Set properties
hfss.material_override = False
hfss.use_causal_materials = True
hfss.modeler.model_units = "mm"
hfss.modeler.create_open_region(Frequency="10GHz", boundary_type="Radiation")
hfss.mesh.assign_initial_mesh_from_slider(level=4)

# CPWG Variables
total_length = 100  # mm
theta = 60  # degrees, bending angle
radius = 20  # mm, bending radius
width = 2  # mm, signal line width
height = 0.035  # mm, signal line thickness
spacing = 0.5  # mm, gap between signal and ground
ground_width = 4  # mm, ground width
ground_thickness = 0.035  # mm, ground thickness
dielectric_height = 0.2  # mm, dielectric thickness
dielectric_material = "Polyimide"

# Helper function to create a bent path
def create_bent_path(start, direction, length, bend_radius, bend_angle_deg):
    # Create a straight segment, then a bend (arc), then another straight segment
    points = []
    # First straight segment
    straight_len = (length - math.radians(bend_angle_deg) * bend_radius)
    if straight_len < 0:
        straight_len = 0
    p0 = start
    p1 = [p0[0] + direction[0] * straight_len, p0[1] + direction[1] * straight_len, p0[2]]
    points.append(p0)
    points.append(p1)
    # Arc segment
    center = [p1[0], p1[1] + bend_radius, p1[2]]
    arc_points = []
    num_arc_points = 20
    for i in range(num_arc_points + 1):
        angle = math.radians(bend_angle_deg) * i / num_arc_points
        x = p1[0] + bend_radius * math.sin(angle)
        y = p1[1] + bend_radius * (1 - math.cos(angle))
        arc_points.append([x, y, p1[2]])
    points.extend(arc_points)
    # Final straight segment
    end_direction = [math.cos(math.radians(bend_angle_deg)), math.sin(math.radians(bend_angle_deg))]
    p2 = [arc_points[-1][0] + end_direction[0] * straight_len, arc_points[-1][1] + end_direction[1] * straight_len, arc_points[-1][2]]
    points.append(p2)
    return points

# Draw signal line
signal_start = [0, 0, dielectric_height + height / 2]
signal_direction = [1, 0]
signal_path = create_bent_path(signal_start, signal_direction, total_length, radius, theta)
signal_polyline = hfss.modeler.create_polyline(signal_path, xsection_type="Rectangle", xsection_width=width, xsection_height=height, name="Signal_Line", matname="copper")

# Draw ground lines (left and right)
ground_offset = width / 2 + spacing + ground_width / 2
ground1_start = [0, -ground_offset, dielectric_height + ground_thickness / 2]
ground2_start = [0, ground_offset, dielectric_height + ground_thickness / 2]
ground_direction = [1, 0]
ground1_path = create_bent_path(ground1_start, ground_direction, total_length, radius, theta)
ground2_path = create_bent_path(ground2_start, ground_direction, total_length, radius, theta)
ground1_polyline = hfss.modeler.create_polyline(ground1_path, xsection_type="Rectangle", xsection_width=ground_width, xsection_height=ground_thickness, name="Ground1", matname="copper")
ground2_polyline = hfss.modeler.create_polyline(ground2_path, xsection_type="Rectangle", xsection_width=ground_width, xsection_height=ground_thickness, name="Ground2", matname="copper")

# Draw dielectric
dielectric_length = total_length + 10
dielectric_width = 2 * (ground_offset + ground_width / 2) + 2
dielectric_box = hfss.modeler.create_box(
    [-5, -dielectric_width / 2, 0],
    [dielectric_length, dielectric_width, dielectric_height],
    name="Dielectric",
    matname=dielectric_material
)

# Draw bottom metals (ground plane)
bottom_ground = hfss.modeler.create_box(
    [-5, -dielectric_width / 2, 0],
    [dielectric_length, dielectric_width, ground_thickness],
    name="Bottom_Ground",
    matname="copper"
)

# Create port interfaces (PEC enclosures)
pec_box1 = hfss.modeler.create_box(
    [-5, -dielectric_width / 2, 0],
    [0.5, dielectric_width, dielectric_height + height],
    name="PEC1",
    matname="pec"
)
pec_box2 = hfss.modeler.create_box(
    [total_length + 4.5, -dielectric_width / 2, 0],
    [0.5, dielectric_width, dielectric_height + height],
    name="PEC2",
    matname="pec"
)

# Assign Perfect E boundary to PEC enclosures
hfss.assign_perfecte_to_sheets(["PEC1", "PEC2"])

# Create wave ports
port1 = hfss.create_wave_port_from_sheet("PEC1", axisdir=1, name="Port1")
port2 = hfss.create_wave_port_from_sheet("PEC2", axisdir=1, name="Port2")

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

# Plot model
hfss.plot(show=True)

# Release AEDT
hfss.release_desktop(close_projects=True, close_desktop=True)
```