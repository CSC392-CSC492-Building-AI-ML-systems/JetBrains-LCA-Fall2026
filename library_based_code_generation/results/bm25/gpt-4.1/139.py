```python
import os
import tempfile
from ansys.dpf import core as dpf
from ansys.dpf.core import cyclic_support
from ansys.dpf.core import operators as ops

# Download a multi-stage cyclic result
result_path = dpf.downloads.download_multi_stage_cyclic_result()

# Create a model from the result
model = dpf.Model(result_path)
print("Model state:")
print(model)

# Verify that the model is a multi-stage model by checking the result info
result_info = model.metadata.result_info
if result_info.cyclic_multistage:
    print("This is a multi-stage cyclic model.")
else:
    print("This is NOT a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_sup = model.metadata.cyclic_support
num_stages = cyclic_sup.num_stages
print(f"Number of stages: {num_stages}")

for stage in range(num_stages):
    num_sectors = cyclic_sup.num_sectors(stage)
    print(f"Stage {stage}: Number of sectors = {num_sectors}")

# Number of nodes in the first stage's base sector
base_sector_scoping = cyclic_sup.sectors_scoping(stage=0, sector=0)
mesh = model.metadata.mesh
nodes_in_base_sector = mesh.nodes.scoping.get_ids(base_sector_scoping)
print(f"Number of nodes in first stage's base sector: {len(nodes_in_base_sector)}")

# Expand displacement results on chosen sectors
# Create a displacement cyclic operator
disp_cyclic_op = ops.result.cyclic_expanded_displacement(data_sources=model)

# Select the sectors to expand on the first stage
sectors_to_expand_stage0 = [0, 1]  # Example: expand on sectors 0 and 1 of stage 0
disp_cyclic_op.inputs.sectors_to_expand.connect({0: sectors_to_expand_stage0})

# Select the sectors to expand stage by stage
sectors_to_expand_stage1 = [0, 2]  # Example: expand on sectors 0 and 2 of stage 1
if num_stages > 1:
    disp_cyclic_op.inputs.sectors_to_expand.connect({1: sectors_to_expand_stage1})

# Expand the displacements and get total deformation
expanded_disp = disp_cyclic_op.outputs.fields_container()
total_deformation = ops.math.norm_fc(fields_container=expanded_disp).outputs.field()

# Get the expanded mesh
expanded_mesh = disp_cyclic_op.outputs.meshed_region()

# Plot the expanded result on the expanded mesh
print("Plotting expanded total deformation on expanded mesh...")
total_deformation.plot(expanded_mesh)

# Demonstrate how to expand only some sectors for the mesh
# For example, expand only sector 0 of stage 0 and sector 1 of stage 1
partial_sectors = {0: [0], 1: [1]} if num_stages > 1 else {0: [0]}
disp_cyclic_op_partial = ops.result.cyclic_expanded_displacement(data_sources=model)
disp_cyclic_op_partial.inputs.sectors_to_expand.connect(partial_sectors)
expanded_disp_partial = disp_cyclic_op_partial.outputs.fields_container()
total_deformation_partial = ops.math.norm_fc(fields_container=expanded_disp_partial).outputs.field()
expanded_mesh_partial = disp_cyclic_op_partial.outputs.meshed_region()

print("Plotting expanded total deformation on partially expanded mesh...")
total_deformation_partial.plot(expanded_mesh_partial)

# Check results precisely
# Print the time frequency support to see the harmonic index
tf_support = model.metadata.time_freq_support
print("Time frequency support:")
print(tf_support)

# Verify that the displacement values are the same on all nodes
disp_field = expanded_disp[0]
disp_data = disp_field.data
if (disp_data == disp_data[0]).all():
    print("Displacement values are the same on all nodes.")
else:
    print("Displacement values differ across nodes.")
```