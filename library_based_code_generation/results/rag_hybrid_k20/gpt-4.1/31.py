```python
from pyaedt import Hfss, Desktop
from pyaedt.modeler.geometry_operators import GeometryOperators
import math

# Set non-graphical mode
non_graphical = True

# Launch AEDT in specified version and solution type
aedt_version = "2024.1"
project_name = "Flex_CPWG"
design_name = "CPWG_Flex"
solution_type = "DrivenModal"

desktop = Desktop(specified_version=aedt_version, non_graphical=non_graphical, new_desktop_session=True)
hfss = Hfss(projectname=project_name, designname=design_name, solution_type=solution_type, specified_version=aedt_version)

# Set properties
hfss.material_override = False
hfss.use_causal_materials = True
hfss.modeler.model_units = "mm"
hfss.modeler.create_open_region(Frequency="10GHz", boundary="Radiation")
hfss.mesh.assign_initial_mesh_from_slider(level=4)

# Define CPWG variables
total_length = 100  # mm
theta = 60  # degrees
radius = 30  # mm (bend radius)
signal_width = 2  # mm
signal_height = 0.035  # mm (copper thickness)
spacing = 0.3  # mm (gap between signal and ground)
ground_width = 4  # mm (each ground width)
ground_thickness = 0.035  # mm
dielectric_height = 0.2  # mm
dielectric_material = "Polyimide"

# Function to create a bent polyline (arc + straight)
def create_bent_line(start, direction, length_straight, bend_radius, bend_angle_deg, width, height, name):
    # Create straight segment
    end_straight = [start[0] + length_straight * direction[0],
                    start[1] + length_straight * direction[1],
                    start[2]]
    polyline_points = [start, end_straight]
    # Create arc segment
    angle_rad = math.radians(bend_angle_deg)
    arc_center = [end_straight[0], end_straight[1] + bend_radius, end_straight[2]]
    arc_start = end_straight
    arc_end = [arc_center[0] + bend_radius * math.sin(angle_rad),
               arc_center[1] - bend_radius * (1 - math.cos(angle_rad)),
               arc_center[2]]
    arc_points = [arc_start, arc_end]
    # Combine points
    all_points = polyline_points + arc_points
    polyline = hfss.modeler.create_polyline(
        all_points,
        cover_surface=True,
        xsection_type="Rectangle",
        xsection_width=width,
        xsection_height=height,
        name=name
    )
    return polyline

# Draw signal line (center)
signal_start = [0, 0, dielectric_height]
signal_direction = [1, 0]
signal_straight = total_length / 2 - radius * math.radians(theta) / 2
signal = create_bent_line(signal_start, signal_direction, signal_straight, radius, theta, signal_width, signal_height, "Signal")

# Draw ground lines (left and right)
ground_offset = signal_width / 2 + spacing + ground_width / 2
ground_left_start = [0, -ground_offset, dielectric_height]
ground_right_start = [0, ground_offset, dielectric_height]
ground_direction = [1, 0]
ground_left = create_bent_line(ground_left_start, ground_direction, signal_straight, radius, theta, ground_width, ground_thickness, "GND_Left")
ground_right = create_bent_line(ground_right_start, ground_direction, signal_straight, radius, theta, ground_width, ground_thickness, "GND_Right")

# Draw dielectric
dielectric_length = total_length
dielectric_width = 2 * (ground_offset + ground_width / 2) + 1
dielectric = hfss.modeler.create_box(
    [-1, -dielectric_width / 2, 0],
    [dielectric_length + 2, dielectric_width, dielectric_height],
    name="Dielectric",
    matname=dielectric_material
)

# Create bottom metals (ground plane under signal and grounds)
bottom_plane = hfss.modeler.create_box(
    [-1, -dielectric_width / 2, 0],
    [dielectric_length + 2, dielectric_width, ground_thickness],
    name="Bottom_Ground",
    matname="copper"
)

# Create port interfaces (PEC enclosures)
pec_box1 = hfss.modeler.create_box(
    [-2, -dielectric_width / 2 - 1, 0],
    [1, dielectric_width + 2, dielectric_height + 0.5],
    name="PEC1",
    matname="pec"
)
pec_box2 = hfss.modeler.create_box(
    [dielectric_length + 1, -dielectric_width / 2 - 1, 0],
    [1, dielectric_width + 2, dielectric_height + 0.5],
    name="PEC2",
    matname="pec"
)

# Assign Perfect E boundary to PEC boxes
hfss.assign_perfecte_to_sheets([pec_box1.faces[4], pec_box2.faces[5]])

# Create wave ports
port1 = hfss.create_wave_port_from_face(pec_box1.faces[4], integration_line=[signal_start, [signal_start[0], signal_start[1], signal_start[2] + 1]], name="Port1")
port2 = hfss.create_wave_port_from_face(pec_box2.faces[5], integration_line=[[dielectric_length, 0, dielectric_height], [dielectric_length, 0, dielectric_height + 1]], name="Port2")

# Create setup and sweep
setup = hfss.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.props["MaximumPasses"] = 20
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 2
setup.props["PercentRefinement"] = 30
setup.props["BasisOrder"] = 1
setup.update()

sweep = setup.add_sweep(name="Sweep1")
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