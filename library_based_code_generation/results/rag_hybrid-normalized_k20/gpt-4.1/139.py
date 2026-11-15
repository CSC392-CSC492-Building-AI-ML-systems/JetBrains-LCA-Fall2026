```python
import os
from ansys.dpf import core as dpf
from ansys.dpf.core import examples
from ansys.dpf.core.plotter import DpfPlotter

# Download a multi-stage cyclic result
multi_stage_cyclic_path = examples.download_multi_stage_cyclic_result()

# Create a model from the result
model = dpf.Model(multi_stage_cyclic_path)
print("Model state:")
print(model)

# Verify that the model is a multi-stage model
result_info = model.metadata.result_info
cyclic_symmetry_type = dpf.result_info_get_cyclic_symmetry_type(result_info)
print(f"Cyclic symmetry type: {cyclic_symmetry_type}")
if cyclic_symmetry_type != "MultiStage":
    raise RuntimeError("The model is not a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_support = dpf.result_info_get_cyclic_support(result_info)
num_stages = cyclic_support.num_stages
print(f"Number of stages: {num_stages}")

for stage in range(num_stages):
    num_sectors = dpf.cyclic_support_get_num_sectors(cyclic_support, stage)
    print(f"Stage {stage}: Number of sectors = {num_sectors}")

# Number of nodes in the first stage's base sector
base_sector_scoping = dpf.cyclic_support_get_sectors_scoping(cyclic_support, stage=0, sector=0)
num_nodes_base_sector = len(base_sector_scoping.ids)
print(f"Number of nodes in the first stage's base sector: {num_nodes_base_sector}")

# Expand displacement results on chosen sectors
# Create a displacement cyclic operator
disp_op = dpf.cyclic_expanded_displacement(model)

# Select sectors to expand on the first stage (e.g., sectors 0 and 1)
sectors_to_expand_stage0 = [0, 1]
sector_scoping_stage0 = dpf.result_definition_get_cyclic_sectors_scoping(model.metadata.result_info, stage=0, sectors=sectors_to_expand_stage0)
dpf.result_definition_set_cyclic_sectors_scoping(model.metadata.result_info, stage=0, scoping=sector_scoping_stage0)

# Select sectors to expand stage by stage (e.g., sector 0 for stage 1)
sectors_to_expand_stage1 = [0]
sector_scoping_stage1 = dpf.result_definition_get_cyclic_sectors_scoping(model.metadata.result_info, stage=1, sectors=sectors_to_expand_stage1)
dpf.result_definition_set_cyclic_sectors_scoping(model.metadata.result_info, stage=1, scoping=sector_scoping_stage1)

# Expand the displacements
expanded_disp = disp_op.outputs.fields_container()

# Get total deformation
total_deformation = expanded_disp[0].norm()

# Get the expanded mesh
mesh_expansion_op = dpf.cyclic_mesh_expansion(model)
expanded_mesh = mesh_expansion_op.outputs.mesh()

# Plot the expanded result on the expanded mesh
plotter = DpfPlotter()
plotter.add_field(total_deformation, mesh=expanded_mesh)
plotter.show_figure(auto_close=True)

# Expand only some sectors for the mesh (e.g., only sector 0 for both stages)
for stage in range(num_stages):
    sector_scoping = dpf.result_definition_get_cyclic_sectors_scoping(model.metadata.result_info, stage=stage, sectors=[0])
    dpf.result_definition_set_cyclic_sectors_scoping(model.metadata.result_info, stage=stage, scoping=sector_scoping)

# Re-expand mesh and displacement
expanded_mesh_partial = mesh_expansion_op.outputs.mesh()
expanded_disp_partial = disp_op.outputs.fields_container()
total_deformation_partial = expanded_disp_partial[0].norm()

# Plot the expanded result on the partially expanded mesh
plotter_partial = DpfPlotter()
plotter_partial.add_field(total_deformation_partial, mesh=expanded_mesh_partial)
plotter_partial.show_figure(auto_close=True)

# Check results precisely
# Print the time frequency support to see the harmonic index
time_freq_support = expanded_disp[0].time_freq_support
print("Time frequency support:")
print(time_freq_support)

# Verify that the displacement values are the same on all nodes
disp_values = expanded_disp[0].data
if (disp_values == disp_values[0]).all():
    print("Displacement values are the same on all nodes.")
else:
    print("Displacement values differ across nodes.")
```