```python
from pyaedt import Maxwell2d, Desktop
from pyaedt.modeler.geometry_operators import GeometryOperators
import pyvista as pv
import os

# Launch AEDT in non-graphical mode
desktop = Desktop(specified_version="2024.1", non_graphical=True, new_desktop_session=True)

# Create Maxwell 2D project and design
maxwell2d = Maxwell2d(projectname="Maxwell2D_Transient_Project", designname="TransientDesign", solution_type="Transient")

# Save the project
project_path = os.path.join(os.getcwd(), "Maxwell2D_Transient_Project.aedt")
maxwell2d.save_project(project_path)

# Create a rectangle
rect1 = maxwell2d.modeler.create_rectangle(cs_plane="XY", position=[0, 0, 0], dimension_list=[10, 5], name="Rect1", matname="Copper")

# Duplicate the rectangle
rect2 = maxwell2d.modeler.duplicate_and_move(rect1, [0, 7, 0])
maxwell2d.modeler.rename_object(rect2[0], "Rect2")

# Create air region
air_region = maxwell2d.modeler.create_air_region(x_pos=20, y_pos=20, z_pos=0, x_neg=20, y_neg=20, z_neg=0, 
                                                is2d=True, air_material="air", name="AirRegion")

# Assign windings to the rectangles
maxwell2d.assign_winding(rect1, winding_type="Current", current_value=10, name="Winding1")
maxwell2d.assign_winding("Rect2", winding_type="Current", current_value=-10, name="Winding2")

# Assign balloon boundary to air region
maxwell2d.assign_balloon("AirRegion")

# Plot the model
maxwell2d.plot(show=False, export_path="maxwell2d_model.png")

# Create a transient setup
setup = maxwell2d.create_setup("TransientSetup")
setup.props["StopTime"] = "10ms"
setup.props["TimeStep"] = "0.1ms"
setup.update()

# Create a rectangular plot (field plot)
plot = maxwell2d.post.create_2d_plot(
    quantity_name="B",
    plot_folder="Field_Plots",
    plot_name="B_Field",
    intrinsic_dict={"Time": "5ms"},
    plot_type="Rectangular"
)

# Solve the model
maxwell2d.analyze_setup("TransientSetup")

# Create output and plot it using PyVista
solution = maxwell2d.post.get_solution_data(
    expression="B",
    intrinsic_dict={"Time": "5ms"},
    context="Rect1"
)
x = solution.primary_sweep_values
y = solution.data_real()
plotter = pv.Plotter()
plotter.add_mesh(pv.Line([[i, 0, 0] for i in x]), scalars=y, line_width=5)
plotter.show(screenshot="pyvista_plot.png")

# Generate the same plot outside AEDT
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlabel("Position")
plt.ylabel("B Field (T)")
plt.title("B Field along Rect1 at 5ms")
plt.savefig("matplotlib_plot.png")
plt.close()

# Close AEDT
maxwell2d.release_desktop(close_projects=True, close_desktop=True)
```