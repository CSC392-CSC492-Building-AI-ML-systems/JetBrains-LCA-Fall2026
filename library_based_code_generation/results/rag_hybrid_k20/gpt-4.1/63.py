```python
import matplotlib.pyplot as plt
from ansys.dpf import core as dpf
from ansys.dpf.core import operators as ops

# Path to result file
result_file = "path/to/your/result_file.rst"

# Load model
model = dpf.Model(result_file)

# Get all node IDs
mesh = model.metadata.meshed_region
all_node_ids = mesh.nodes.scoping.ids

# Find node with minimum number of surrounding elements
min_elem_count = float('inf')
min_node_id = None
node_to_elems = {}
for node_id in all_node_ids:
    elems = mesh.elements_scoping_by_node(node_id)
    node_to_elems[node_id] = elems
    if len(elems) < min_elem_count:
        min_elem_count = len(elems)
        min_node_id = node_id

# Get minimum list of elements by node
min_elem_list = node_to_elems[min_node_id]

# Read elemental volumes
vol_op = ops.mesh.elemental_volume()
vol_op.inputs.mesh.connect(mesh)
elemental_volumes = vol_op.outputs.fields_container()[0]
elem_ids = elemental_volumes.scoping.ids
elem_id_to_vol = {eid: elemental_volumes.get_entity_data_by_id(eid)[0] for eid in elem_ids}

# Get volumes for min_elem_list
min_elem_volumes = [elem_id_to_vol[eid] for eid in min_elem_list]
total_volume = sum(min_elem_volumes)

# Compute equivalent elemental stress
stress_op = ops.result.stress()
stress_op.inputs.data_sources.connect(model.metadata.data_sources)
stress_fc = stress_op.outputs.fields_container()

# Compute equivalent (von Mises) stress per element
seqv_op = ops.math.norm()
seqv_op.inputs.fields_container.connect(stress_fc)
seqv_fc = seqv_op.outputs.fields_container()[0]

# Average equivalent stress on elements in min_elem_list
elem_id_to_seqv = {eid: seqv_fc.get_entity_data_by_id(eid)[0] for eid in seqv_fc.scoping.ids}
seqv_on_min_elems = [elem_id_to_seqv[eid] for eid in min_elem_list]

# Dot product seqv * volume, sum, divide by total volume
weighted_sum = sum([s * v for s, v in zip(seqv_on_min_elems, min_elem_volumes)])
average_elemental_stress = weighted_sum / total_volume

print("Average Elemental Stress on Minimum Volume:", average_elemental_stress)

# Plot equivalent elemental stress
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(seqv_fc.scoping.ids, [seqv_fc.get_entity_data_by_id(eid)[0] for eid in seqv_fc.scoping.ids])
plt.title("Equivalent Elemental Stress")
plt.xlabel("Element ID")
plt.ylabel("Seqv Stress")

# Plot volume averaged elemental equivalent stress (single value)
plt.subplot(1, 2, 2)
plt.bar(['Volume Averaged'], [average_elemental_stress])
plt.title("Volume Averaged Elemental Equivalent Stress")
plt.ylabel("Averaged Seqv Stress")
plt.tight_layout()
plt.show()

# Use operator with same algorithm
avg_stress_op = ops.math.weighted_average()
avg_stress_op.inputs.fields_container.connect(seqv_fc)
vol_field = dpf.FieldsContainer()
vol_field.add_field(elemental_volumes)
avg_stress_op.inputs.weights.connect(vol_field)
avg_stress = avg_stress_op.outputs.field()[0].data[0]
print("Operator Volume Averaged Elemental Equivalent Stress:", avg_stress)
```