```python
import matplotlib.pyplot as plt
from ansys.dpf import core as dpf
from ansys.dpf.core import cyclic_support
from ansys.dpf.core import operators as ops
from ansys.dpf.core.plotter import DpfPlotter

# Download a multi-stage cyclic result
path = dpf.downloads.download_multi_stage_cyclic_result()

# Create a model from the result
model = dpf.Model(path)
print("Model state:")
print(model)

# Verify that the model is a multi-stage model
result_info = model.metadata.result_info
cyclic_type = result_info.cyclic_symmetry_type
print(f"Cyclic symmetry type: {cyclic_type}")
if cyclic_type != dpf.enums.CyclicSymmetryType.MultiStage:
    raise RuntimeError("The model is not a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_sup = result_info.cyclic_support
num_stages = cyclic_sup.num_stages
print(f"Number of stages: {num_stages}")

for stage in range(num_stages):
    num_sectors = cyclic_sup.get_num_sectors(stage)
    print(f"Stage {stage}: Number of sectors = {num_sectors}")

# Number of nodes in the first stage's base sector
base_sector_scoping = cyclic_sup.get_sectors_scoping(stage=0, sectors=[0])
num_nodes_base_sector = len(base_sector_scoping.ids)
print(f"Number of nodes in first stage's base sector: {num_nodes_base_sector}")

# Expand displacement results on chosen sectors
# Create a displacement cyclic operator
disp_op = ops.result.cyclic_expanded_displacement(data_sources=model)

# Select sectors to expand on the first stage
sectors_to_expand_stage0 = [0, 1]  # Example: expand on sectors 0 and 1
disp_op.inputs.sectors_scoping.connect(
    cyclic_sup.get_sectors_scoping(stage=0, sectors=sectors_to_expand_stage0)
)

# Select sectors to expand stage by stage
sectors_to_expand = []
for stage in range(num_stages):
    sectors = [0, 1]  # Example: expand on sectors 0 and 1 for each stage
    scoping = cyclic_sup.get_sectors_scoping(stage=stage, sectors=sectors)
    sectors_to_expand.append(scoping)

# Expand the displacements and get total deformation
expanded_disp = disp_op.outputs.fields_container()
total_deformation = ops.math.norm_fc(fields_container=expanded_disp).outputs.field()

# Get the expanded mesh
mesh_expansion_op = ops.mesh.cyclic_mesh_expansion(data_sources=model)
expanded_mesh = mesh_expansion_op.outputs.mesh()

# Plot the expanded result on the expanded mesh
plotter = DpfPlotter()
plotter.add_field(total_deformation, expanded_mesh)
plotter.show_figure(show_axes=True, cbar=True, title="Total Deformation on Expanded Mesh")

# Expand only some sectors for the mesh and plot
partial_sectors = [0]  # Only expand sector 0 for each stage
partial_scoping = []
for stage in range(num_stages):
    scoping = cyclic_sup.get_sectors_scoping(stage=stage, sectors=partial_sectors)
    partial_scoping.append(scoping)

mesh_expansion_op.inputs.sectors_scoping.connect(partial_scoping[0])
partial_expanded_mesh = mesh_expansion_op.outputs.mesh()
plotter = DpfPlotter()
plotter.add_field(total_deformation, partial_expanded_mesh)
plotter.show_figure(show_axes=True, cbar=True, title="Total Deformation on Partial Expanded Mesh")

# Check results precisely
# Print the time frequency support to see the harmonic index
tf_support = expanded_disp[0].time_freq_support
print("Time frequency support:")
print(tf_support)

# Verify that the displacement values are the same on all nodes
disp_values = expanded_disp[0].data
if (disp_values == disp_values[0]).all():
    print("Displacement values are the same on all nodes.")
else:
    print("Displacement values differ across nodes.")
```