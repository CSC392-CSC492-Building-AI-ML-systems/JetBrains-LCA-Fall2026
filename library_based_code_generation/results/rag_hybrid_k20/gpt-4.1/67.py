```python
from pyaedt import Hfss2dExtractor, Desktop
import matplotlib.pyplot as plt

# Set non-graphical mode
non_graphical = True

# Launch AEDT and 2D Extractor
desktop = Desktop(specified_version="2024.1", non_graphical=non_graphical, new_desktop_session=True)
extractor = Hfss2dExtractor(projectname="CPWG_Design", designname="CPWG_2D")

# Define variables
extractor["wg_width"] = "2mm"
extractor["gap"] = "0.2mm"
extractor["gnd_width"] = "5mm"
extractor["sub_height"] = "1.6mm"
extractor["sub_length"] = "20mm"
extractor["cond_thick"] = "0.035mm"
extractor["coating_thick"] = "0.01mm"
extractor["coating_eps"] = "3"
extractor["sub_eps"] = "4.4"

# Create primitives
# Signal trace
signal = extractor.modeler.create_rectangle(
    position=[0, 0],
    dimension_list=["wg_width", "sub_length"],
    name="Signal",
    matname="copper"
)

# Coplanar ground (left)
gnd_left = extractor.modeler.create_rectangle(
    position=["-gap-gnd_width", 0],
    dimension_list=["gnd_width", "sub_length"],
    name="GND_Left",
    matname="copper"
)

# Coplanar ground (right)
gnd_right = extractor.modeler.create_rectangle(
    position=["wg_width+gap", 0],
    dimension_list=["gnd_width", "sub_length"],
    name="GND_Right",
    matname="copper"
)

# Reference ground plane (bottom)
ref_gnd = extractor.modeler.create_rectangle(
    position=["-gap-gnd_width", "-sub_height"],
    dimension_list=["2*gnd_width+wg_width+2*gap", "sub_length"],
    name="Ref_GND",
    matname="copper"
)

# Dielectric substrate
substrate = extractor.modeler.create_rectangle(
    position=["-gap-gnd_width", "-sub_height"],
    dimension_list=["2*gnd_width+wg_width+2*gap", "sub_length"],
    name="Substrate",
    matname="FR4_epoxy"
)
extractor.modeler.assign_material("Substrate", "FR4_epoxy")
extractor["FR4_epoxy"] = {"permittivity": "sub_eps", "thickness": "sub_height"}

# Conformal coating
coating = extractor.modeler.create_rectangle(
    position=["-gap-gnd_width", "0"],
    dimension_list=["2*gnd_width+wg_width+2*gap", "sub_length"],
    name="Coating",
    matname="coating_mat"
)
extractor.modeler.assign_material("Coating", "coating_mat")
extractor["coating_mat"] = {"permittivity": "coating_eps", "thickness": "coating_thick"}

# Assign conductor to signal
extractor.assign_single_conductor(signal.name, thickness="cond_thick")

# Assign reference ground
extractor.assign_reference_ground(ref_gnd.name)

# Assign Huray model to signal
extractor.assign_huray_surface(signal.name, radius=0.5e-6, ratio=2.0)

# Create setup
setup = extractor.create_setup(name="CPWG_Setup")
setup.props["Frequency"] = "10GHz"
setup.update()

# Analyze
extractor.analyze()

# Plot solution data
solution_data = extractor.get_solution_data(
    expressions=["Z0"],
    setup_sweep_name="CPWG_Setup : LastAdaptive"
)
plt.plot(solution_data.sweeps["Freq"], solution_data.data_real("Z0"))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Characteristic Impedance (Ohm)")
plt.title("CPWG Characteristic Impedance vs Frequency")
plt.grid(True)
plt.show()

# Save project and close AEDT
extractor.save_project()
extractor.release_desktop(close_projects=True, close_desktop=True)
```