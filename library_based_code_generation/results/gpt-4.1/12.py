```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import TetrahedronMesh

# Define nodes (vertices) of the tetrahedron
nodes = np.array([
    [0.0, 0.0, 0.0],  # Node 0
    [1.0, 0.0, 0.0],  # Node 1
    [0.0, 1.0, 0.0],  # Node 2
    [0.0, 0.0, 1.0],  # Node 3
])

# Define cells (tetrahedra) by indices of nodes
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
entity_nodes = mesh.node
entity_edges = mesh.entity('edge')
entity_faces = mesh.entity('face')
entity_cells = mesh.cell

# Barycenters
barycenter_nodes = entity_nodes
barycenter_edges = mesh.entity_barycenter('edge')
barycenter_faces = mesh.entity_barycenter('face')
barycenter_cells = mesh.entity_barycenter('cell')

# Measures
measure_nodes = np.zeros(num_nodes)
measure_edges = mesh.entity_measure('edge')
measure_faces = mesh.entity_measure('face')
measure_cells = mesh.entity_measure('cell')

# Relationships
cell2cell = mesh.ds.cell2cell
cell2face = mesh.ds.cell2face
cell2edge = mesh.ds.cell2edge
cell2node = mesh.ds.cell2node

face2cell = mesh.ds.face2cell
face2edge = mesh.ds.face2edge
face2node = mesh.ds.face2node

edge2cell = mesh.ds.edge2cell
edge2face = mesh.ds.edge2face
edge2node = mesh.ds.edge2node

node2cell = mesh.ds.node2cell
node2face = mesh.ds.node2face
node2edge = mesh.ds.node2edge

# Boundary flags
node_bd_flag = mesh.ds.node_bd_flag
edge_bd_flag = mesh.ds.edge_bd_flag
face_bd_flag = mesh.ds.face_bd_flag
cell_bd_flag = mesh.ds.cell_bd_flag

# Indices of boundary entities
boundary_node_indices = np.where(node_bd_flag == 1)[0]
boundary_edge_indices = np.where(edge_bd_flag == 1)[0]
boundary_face_indices = np.where(face_bd_flag == 1)[0]
boundary_cell_indices = np.where(cell_bd_flag == 1)[0]

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in entity_edges:
    pts = entity_nodes[edge]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='k', linewidth=1)

# Plot nodes and their indices
ax.scatter(entity_nodes[:, 0], entity_nodes[:, 1], entity_nodes[:, 2], color='r', s=50)
for i, (x, y, z) in enumerate(entity_nodes):
    ax.text(x, y, z, f'N{i}', color='r', fontsize=12)

# Plot edge indices at barycenters
for i, (x, y, z) in enumerate(barycenter_edges):
    ax.text(x, y, z, f'E{i}', color='b', fontsize=10)

# Plot cell indices at barycenters
for i, (x, y, z) in enumerate(barycenter_cells):
    ax.text(x, y, z, f'C{i}', color='g', fontsize=14)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tetrahedron Mesh with Node, Edge, and Cell Indices')
plt.tight_layout()
plt.show()
```