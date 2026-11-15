```python
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

# Verify that the model is a multi-stage model by checking the result info
result_info = model.metadata.result_info
cyclic_type = result_info.cyclic_symmetry_type
print(f"Cyclic symmetry type: {cyclic_type}")
if cyclic_type != dpf.enums.CyclicSymmetryType.MultiStage:
    raise RuntimeError("The model is not a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_sup = model.metadata.cyclic_support
num_stages = cyclic_sup.num_stages
print(f"Number of stages: {num_stages}")

for stage in range(num_stages):
    num_sectors = cyclic_sup.num_sectors(stage)
    print(f"Stage {stage}: Number of sectors = {num_sectors}")

# Number of nodes in the first stage's base sector
base_sector_scoping = cyclic_sup.sectors_scoping(stage=0, sectors=[0])
base_sector_nodes = base_sector_scoping.ids
print(f"Number of nodes in first stage's base sector: {len(base_sector_nodes)}")

# Expand displacement results on chosen sectors
# Create a displacement cyclic operator
disp_op = ops.result.displacement_cyclic_expanded(model=model)

# Select sectors to expand on the first stage (e.g., sectors 0 and 1)
sectors_to_expand_stage0 = [0, 1]
disp_op.inputs.sectors_scoping.connect(
    cyclic_sup.sectors_scoping(stage=0, sectors=sectors_to_expand_stage0)
)

# Select sectors to expand stage by stage (e.g., sectors 0,1 for stage 0 and 0 for stage 1)
sectors_to_expand = []
for stage in range(num_stages):
    if stage == 0:
        sectors_to_expand.append([0, 1])
    else:
        sectors_to_expand.append([0])
disp_op.inputs.sectors_scoping_per_stage.connect(sectors_to_expand)

# Expand the displacements and get total deformation
expanded_disp = disp_op.outputs.fields_container()
total_deformation = ops.math.norm_fc(expanded_disp).outputs.fields_container()

# Get the expanded mesh
mesh_expansion_op = ops.mesh.cyclic_mesh_expansion(model=model)
mesh_expansion_op.inputs.sectors_scoping_per_stage.connect(sectors_to_expand)
expanded_mesh = mesh_expansion_op.outputs.mesh()

# Plot the expanded result on the expanded mesh
plotter = DpfPlotter()
plotter.add_field(total_deformation[0], expanded_mesh)
plotter.show_figure(show_axes=True, cbar=True, window_title="Expanded Total Deformation")

# Demonstrate how to expand only some sectors for the mesh (e.g., only sector 0 for all stages)
some_sectors = [[0] for _ in range(num_stages)]
mesh_expansion_op2 = ops.mesh.cyclic_mesh_expansion(model=model)
mesh_expansion_op2.inputs.sectors_scoping_per_stage.connect(some_sectors)
expanded_mesh_some = mesh_expansion_op2.outputs.mesh()

disp_op2 = ops.result.displacement_cyclic_expanded(model=model)
disp_op2.inputs.sectors_scoping_per_stage.connect(some_sectors)
expanded_disp_some = disp_op2.outputs.fields_container()
total_deformation_some = ops.math.norm_fc(expanded_disp_some).outputs.fields_container()

plotter2 = DpfPlotter()
plotter2.add_field(total_deformation_some[0], expanded_mesh_some)
plotter2.show_figure(show_axes=True, cbar=True, window_title="Partial Sector Expansion")

# Check results precisely
# Print the time frequency support to see the harmonic index
tf_support = model.metadata.time_freq_support
print("Time frequency support:")
print(tf_support)

# Verify that the displacement values are the same on all nodes (for a given sector)
field = expanded_disp[0]
values = field.data
if (values == values[0]).all():
    print("Displacement values are the same on all nodes.")
else:
    print("Displacement values differ across nodes.")
```