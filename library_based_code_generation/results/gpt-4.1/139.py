```python
import os
import tempfile
import matplotlib.pyplot as plt
from ansys.dpf import core as dpf
from ansys.dpf.core import examples
from ansys.dpf.core.plotter import DpfPlotter

# Download a multi-stage cyclic result file
# For demonstration, we use an example file from DPF if available
# Replace with your own file if needed
result_path = examples.download_multi_stage_cyclic_result()

# Create a model from the result file
model = dpf.Model(result_path)
print("Model state:")
print(model)

# Verify that the model is a multi-stage model
result_info = model.metadata.result_info
print("\nResult Info:")
print(result_info)
if not result_info.has_multi_stage_cyclic_support:
    raise RuntimeError("The model is not a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_support = model.metadata.cyclic_support
num_stages = cyclic_support.num_stages
print(f"\nNumber of stages: {num_stages}")

for stage_id in range(num_stages):
    num_sectors = cyclic_support.num_sectors(stage_id)
    print(f"Stage {stage_id}: Number of sectors = {num_sectors}")

# Number of nodes in the first stage's base sector
base_sector_nodes = cyclic_support.base_sector_nodes(0)
print(f"\nNumber of nodes in the first stage's base sector: {len(base_sector_nodes)}")

# Expand displacement results on chosen sectors
# Create a displacement cyclic operator
disp_op = dpf.operators.result.displacement_cyclic()

# Connect the model's data sources
disp_op.inputs.data_sources.connect(model.metadata.data_sources)

# Select sectors to expand on the first stage (e.g., sectors 0 and 1)
sectors_to_expand_stage0 = [0, 1]
disp_op.inputs.sectors_to_expand.connect({0: sectors_to_expand_stage0})

# Select sectors to expand stage by stage (e.g., sectors 0,1 for stage 0 and 0 for stage 1)
sectors_to_expand = {0: [0, 1], 1: [0]}
disp_op.inputs.sectors_to_expand.connect(sectors_to_expand)

# Expand the displacements
expanded_disp = disp_op.outputs.fields_container()

# Get total deformation (norm)
norm_op = dpf.operators.math.norm_fc(fields_container=expanded_disp)
total_deformation = norm_op.outputs.field()

# Get the expanded mesh
mesh_exp_op = dpf.operators.mesh.expand_cyclic_mesh()
mesh_exp_op.inputs.data_sources.connect(model.metadata.data_sources)
mesh_exp_op.inputs.sectors_to_expand.connect(sectors_to_expand)
expanded_mesh = mesh_exp_op.outputs.mesh()

# Plot the expanded result on the expanded mesh
plotter = DpfPlotter()
plotter.add_field(total_deformation, mesh=expanded_mesh)
plotter.show_figure(show_axes=True, cbar=True, title="Expanded Total Deformation")

# Expand only some sectors for the mesh (e.g., only sector 0 for stage 0)
partial_sectors = {0: [0]}
mesh_exp_op.inputs.sectors_to_expand.connect(partial_sectors)
partial_expanded_mesh = mesh_exp_op.outputs.mesh()

# Plot the expanded result on the partially expanded mesh
plotter = DpfPlotter()
plotter.add_field(total_deformation, mesh=partial_expanded_mesh)
plotter.show_figure(show_axes=True, cbar=True, title="Partial Expansion: Total Deformation")

# Check results precisely
# Print the time frequency support to see the harmonic index
time_freq_support = model.metadata.time_freq_support
print("\nTime Frequency Support:")
print(time_freq_support)

# Verify that the displacement values are the same on all nodes (for a given time/harmonic)
disp_field = expanded_disp[0]
disp_data = disp_field.data
all_equal = (disp_data == disp_data[0]).all()
print(f"\nAre all displacement values the same on all nodes? {all_equal}")
```