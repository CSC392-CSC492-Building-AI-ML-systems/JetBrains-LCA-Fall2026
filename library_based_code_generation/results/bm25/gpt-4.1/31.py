```python
import pyaedt
from pyaedt import Hfss
from pyaedt.modeler.geometry_operators import GeometryOperators
from pyaedt.modeler.cad.object3d import Object3d
from pyaedt.generic.constants import AEDT_UNITS

# Set non-graphical mode
pyaedt.settings.non_graphical = True

# Launch AEDT
aedt_version = "2023.2"
project_name = "Flex_CPWG"
design_name = "CPWG_Flex"
solution_type = "DrivenModal"

hfss = Hfss(projectname=project_name, designname=design_name, solution_type=solution_type, specified_version=aedt_version)

# Set properties
hfss.materials.material_override = True
hfss.materials.enforce_dc_and_causality = True
hfss.modeler.create_open_region(Frequency="10GHz")
hfss.model_units = "mm"
hfss.mesh.assign_initial_mesh()

# Define CPWG variables
total_length = 100  # mm
theta = 45  # degrees
radius = 20  # mm
signal_width = 1  # mm
signal_height = 0.035  # mm
spacing = 0.2  # mm
ground_width = 3  # mm
ground_thickness = 0.035  # mm
dielectric_height = 0.2  # mm
dielectric_material = "Polyimide"

# Function to create a bent polyline
def create_bent_polyline(start_point, length, theta, radius, width, name, mat, height=0.035):
    # Create a straight segment
    straight_length = length / 2 - (radius * theta * 3.14159 / 180) / 2
    p1 = start_point
    p2 = [p1[0] + straight_length, p1[1], p1[2]]
    # Create arc segment
    arc_center = [p2[0], p2[1] + radius, p2[2]]
    arc_start = p2
    arc_end = [arc_center[0] + radius * GeometryOperators.cosd(theta), arc_center[1] - radius * (1 - GeometryOperators.sind(theta)), p2[2]]
    # Create second straight segment
    p3 = [arc_end[0] + straight_length, arc_end[1], arc_end[2]]
    points = [p1, p2, arc_end, p3]
    polyline = hfss.modeler.create_polyline(
        points,
        xsection_type="Rectangle",
        xsection_width=width,
        xsection_height=height,
        name=name,
        matname=mat
    )
    return polyline

# Draw signal line
signal_start = [0, 0, dielectric_height + signal_height / 2]
signal_line = create_bent_polyline(signal_start, total_length, theta, radius, signal_width, "Signal", "copper", signal_height)

# Draw ground lines (left and right)
ground_offset = signal_width / 2 + spacing + ground_width / 2
ground_left_start = [0, -ground_offset, dielectric_height + ground_thickness / 2]
ground_right_start = [0, ground_offset, dielectric_height + ground_thickness / 2]
ground_left = create_bent_polyline(ground_left_start, total_length, theta, radius, ground_width, "GND_Left", "copper", ground_thickness)
ground_right = create_bent_polyline(ground_right_start, total_length, theta, radius, ground_width, "GND_Right", "copper", ground_thickness)

# Draw dielectric
dielectric_length = total_length + 10
dielectric_width = 2 * (ground_offset + ground_width / 2) + 2
dielectric_box = hfss.modeler.create_box(
    [-5, -dielectric_width / 2, 0],
    [dielectric_length, dielectric_width, dielectric_height],
    name="Dielectric",
    matname=dielectric_material
)

# Draw bottom metals (signal and grounds mirrored below)
bottom_signal = hfss.modeler.duplicate_and_mirror(signal_line, mirror_plane="XZ", duplicate=True)
bottom_left = hfss.modeler.duplicate_and_mirror(ground_left, mirror_plane="XZ", duplicate=True)
bottom_right = hfss.modeler.duplicate_and_mirror(ground_right, mirror_plane="XZ", duplicate=True)

# Create port interfaces (PEC enclosures)
port_face1 = hfss.modeler.create_rectangle(
    cs_plane="YZ",
    position=[0, 0, dielectric_height / 2],
    dimension_list=[dielectric_width, dielectric_height],
    name="PortFace1",
    matname="pec"
)
port_face2 = hfss.modeler.create_rectangle(
    cs_plane="YZ",
    position=[total_length, 0, dielectric_height / 2],
    dimension_list=[dielectric_width, dielectric_height],
    name="PortFace2",
    matname="pec"
)

# Assign Perfect E boundary to port faces
hfss.assign_perfecte_to_sheets(["PortFace1", "PortFace2"])

# Create wave ports
hfss.create_wave_port_from_sheet("PortFace1", 1)
hfss.create_wave_port_from_sheet("PortFace2", 2)

# Setup and sweep
setup = hfss.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.props["MaximumPasses"] = 10
setup.props["MinimumPasses"] = 1
setup.props["MinimumConvergedPasses"] = 1
setup.props["PercentRefinement"] = 30
setup.props["BasisOrder"] = 1
setup.props["MinimumFrequency"] = "1GHz"
setup.props["MaximumDeltaS"] = 0.02
setup.update()

hfss.create_linear_count_sweep(
    setupname="Setup1",
    unit="GHz",
    freqstart=1,
    freqstop=20,
    num_of_freq_points=201,
    sweepname="Sweep1"
)

# Plot the model
hfss.plot(show=True)

# Release AEDT
hfss.release_desktop(close_projects=True, close_desktop=True)
```