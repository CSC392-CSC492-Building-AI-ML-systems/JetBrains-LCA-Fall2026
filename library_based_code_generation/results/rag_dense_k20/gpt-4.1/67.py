```python
from pyaedt import Hfss2dExtractor, Desktop
import matplotlib.pyplot as plt

# Set non-graphical mode
non_graphical = True

# Launch AEDT and 2D Extractor
desktop = Desktop(specified_version="2024.1", non_graphical=non_graphical, new_desktop_session=True)
proj_name = "CPWG_2D_Extractor"
design_name = "CPWG_Design"
aedtapp = Hfss2dExtractor(projectname=proj_name, designname=design_name, specified_version="2024.1", new_desktop_session=False)

# Define variables
aedtapp["wg_width"] = "2mm"
aedtapp["wg_gap"] = "0.2mm"
aedtapp["wg_length"] = "10mm"
aedtapp["gnd_width"] = "8mm"
aedtapp["dielectric_thickness"] = "0.8mm"
aedtapp["coating_thickness"] = "0.05mm"

# Create primitives
# Signal trace
signal = aedtapp.modeler.create_rectangle(
    position=[0, 0],
    dimension_list=["wg_length", "wg_width"],
    name="Signal",
    matname="copper"
)

# Coplanar ground (two rectangles on each side of the signal)
gnd_left = aedtapp.modeler.create_rectangle(
    position=[0, f"-wg_gap-wg_width"],
    dimension_list=["wg_length", "wg_gap"],
    name="GND_Left",
    matname="copper"
)
gnd_right = aedtapp.modeler.create_rectangle(
    position=[0, "wg_width"],
    dimension_list=["wg_length", "wg_gap"],
    name="GND_Right",
    matname="copper"
)

# Reference ground plane (below signal and grounds)
ref_gnd = aedtapp.modeler.create_rectangle(
    position=[0, f"-gnd_width/2"],
    dimension_list=["wg_length", "gnd_width"],
    name="Ref_GND",
    matname="copper"
)

# Dielectric
dielectric = aedtapp.modeler.create_rectangle(
    position=[0, f"-gnd_width/2"],
    dimension_list=["wg_length", "gnd_width"],
    name="Dielectric",
    matname="FR4_epoxy"
)
aedtapp.modeler.thicken_sheet("Dielectric", "dielectric_thickness", False)

# Conformal coating
coating = aedtapp.modeler.create_rectangle(
    position=[0, f"-gnd_width/2"],
    dimension_list=["wg_length", "gnd_width"],
    name="Coating",
    matname="SolderMask"
)
aedtapp.modeler.thicken_sheet("Coating", "coating_thickness", True)

# Assign conductor to signal
aedtapp.assign_single_conductor("Signal", return_path_list=["GND_Left", "GND_Right", "Ref_GND"])

# Assign Huray model to signal
aedtapp.assign_huray_surface("Signal", radius=0.5e-6, ratio=2.0, model_name="Huray")

# Create setup
setup = aedtapp.create_setup("Setup1")
setup.props["Frequency"] = "10GHz"
setup.update()

# Analyze
aedtapp.analyze_setup("Setup1")

# Plot solution data
solution_data = aedtapp.get_solution_data(expressions=["Z0"], setup_sweep_name="Setup1 : LastAdaptive")
plt.plot(solution_data.sweeps["Freq"], solution_data.data_real("Z0"))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Z0 (Ohm)")
plt.title("Characteristic Impedance vs Frequency")
plt.grid(True)
plt.show()

# Save project and close AEDT
aedtapp.save_project()
aedtapp.release_desktop(close_projects=True, close_desktop=True)
```