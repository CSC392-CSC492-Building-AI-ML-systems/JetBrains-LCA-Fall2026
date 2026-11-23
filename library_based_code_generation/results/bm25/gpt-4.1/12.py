```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh

# Define nodes (vertices) of a tetrahedron
nodes = np.array([
    [0.0, 0.0, 0.0],  # Node 0
    [1.0, 0.0, 0.0],  # Node 1
    [0.0, 1.0, 0.0],  # Node 2
    [0.0, 0.0, 1.0],  # Node 3
])

# Define cells (tetrahedra) by indices of their nodes
cells = np.array([
    [0, 1, 2, 3],  # Cell 0
])

# Create the mesh
mesh = TetrahedronMesh(nodes, cells)

# Number of entities
num_nodes = mesh.number_of_nodes()
num_edges = mesh.number_of_edges()
num_faces = mesh.number_of_faces()
num_cells = mesh.number_of_cells()

# Store entities
node_coords = mesh.node
edge_nodes = mesh.entity('edge')
face_nodes = mesh.entity('face')
cell_nodes = mesh.entity('cell')

# Barycenters
node_barycenters = node_coords
edge_barycenters = mesh.entity_barycenter('edge')
face_barycenters = mesh.entity_barycenter('face')
cell_barycenters = mesh.entity_barycenter('cell')

# Measures
node_measures = np.zeros(num_nodes)
edge_measures = mesh.entity_measure('edge')
face_measures = mesh.entity_measure('face')
cell_measures = mesh.entity_measure('cell')

# Relationships
cell2cell = mesh.ds.cell_to_cell()
cell2face = mesh.ds.cell_to_face()
cell2edge = mesh.ds.cell_to_edge()
cell2node = mesh.ds.cell_to_node()
face2cell = mesh.ds.face_to_cell()
face2edge = mesh.ds.face_to_edge()
face2node = mesh.ds.face_to_node()
edge2cell = mesh.ds.edge_to_cell()
edge2face = mesh.ds.edge_to_face()
edge2node = mesh.ds.edge_to_node()
node2cell = mesh.ds.node_to_cell()
node2edge = mesh.ds.node_to_edge()
node2face = mesh.ds.node_to_face()

# Boundary flags
node_boundary_flag = mesh.ds.boundary_node_flag()
edge_boundary_flag = mesh.ds.boundary_edge_flag()
face_boundary_flag = mesh.ds.boundary_face_flag()
cell_boundary_flag = mesh.ds.boundary_cell_flag()

# Indices of boundary entities
boundary_node_indices = np.where(node_boundary_flag)[0]
boundary_edge_indices = np.where(edge_boundary_flag)[0]
boundary_face_indices = np.where(face_boundary_flag)[0]
boundary_cell_indices = np.where(cell_boundary_flag)[0]

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mesh.add_plot(ax=ax, showindex=True, color='w', linewidth=1)

# Plot node indices
for i, xyz in enumerate(node_coords):
    ax.text(*xyz, f'N{i}', color='red', fontsize=12)

# Plot edge indices
edge_centers = edge_barycenters
for i, xyz in enumerate(edge_centers):
    ax.text(*xyz, f'E{i}', color='blue', fontsize=12)

# Plot cell indices
cell_centers = cell_barycenters
for i, xyz in enumerate(cell_centers):
    ax.text(*xyz, f'C{i}', color='green', fontsize=12)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tetrahedron Mesh with Indices')
plt.tight_layout()
plt.show()
```