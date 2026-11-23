```python
from pyaedt import Maxwell2d
from pyaedt import Desktop
from pyaedt.modeler.geometry_operators import GeometryOperators
import os

# Set non-graphical mode and launch AEDT
desktop = Desktop(specified_version="2024.1", non_graphical=True, new_desktop_session=True)

# Launch Maxwell 2D and create a new project
maxwell2d = Maxwell2d(projectname="Maxwell2D_Transient_Project", designname="TransientDesign", solution_type="Transient")

# Create a rectangle (sheet)
rect1 = maxwell2d.modeler.create_rectangle(position=[0, 0], dimension_list=[10, 5], name="Rect1", matname="copper")

# Duplicate the rectangle and move it
rect2 = maxwell2d.modeler.duplicate_and_move(rect1, [0, 10])
rect2 = rect2[0]
maxwell2d.modeler.rename_object(rect2, "Rect2")

# Create an air region
air_region = maxwell2d.modeler.create_air_region(x_pos=20, y_pos=20, z_pos=0, x_neg=20, y_neg=20, z_neg=0, 
                                                is2d=True, air_material="air", name="AirRegion")

# Assign windings to the sheets
winding1 = maxwell2d.assign_winding(rect1, winding_type="Current", current_value=1, name="Winding1")
winding2 = maxwell2d.assign_winding(rect2, winding_type="Current", current_value=-1, name="Winding2")

# Assign balloon boundary to air region
maxwell2d.assign_balloon(air_region)

# Plot the model
maxwell2d.plot(show=False, export_path="model_plot.png")

# Create a transient setup
setup = maxwell2d.create_setup("TransientSetup")
setup.props["StopTime"] = "10ms"
setup.props["TimeStep"] = "0.1ms"
setup.update()

# Create a rectangular plot (field plot)
plot_name = maxwell2d.post.create_fieldplot_surface([rect1, rect2], "Mag_B", plot_type="Rectangular", name="B_FieldPlot")

# Save the project
maxwell2d.save_project()

# Solve the model
maxwell2d.analyze_setup("TransientSetup")

# Create output and plot it using PyVista
fields = maxwell2d.post.get_solution_data(expressions="Mag_B", setup_sweep_name="TransientSetup : LastAdaptive")
fields.plot_contour(show=False, export_path="field_contour.png")

# Generate the same plot outside AEDT using PyVista
import pyvista as pv
import numpy as np

# Extract mesh and field data
mesh = maxwell2d.post.get_mesh_plot(plot_name)
if mesh:
    grid = mesh.meshgrid
    scalars = mesh.data
    p = pv.Plotter()
    p.add_mesh(grid, scalars=scalars, cmap="viridis")
    p.show(screenshot="pyvista_field_contour.png")

# Close AEDT
maxwell2d.release_desktop(close_projects=True, close_desktop=True)
```