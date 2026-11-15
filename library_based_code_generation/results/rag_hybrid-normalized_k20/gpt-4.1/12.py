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
node_coords = mesh.node  # shape: (num_nodes, 3)
edge_nodes = mesh.edges  # shape: (num_edges, 2)
face_nodes = mesh.faces  # shape: (num_faces, 3)
cell_nodes = mesh.cell  # shape: (num_cells, 4)

# Barycenters
node_barycenters = node_coords
edge_barycenters = mesh.entity_barycenter('edge')
face_barycenters = mesh.entity_barycenter('face')
cell_barycenters = mesh.entity_barycenter('cell')

# Measures
node_measures = np.zeros(num_nodes)  # nodes are points, measure is zero
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
node2face = mesh.ds.node_to_face()
node2edge = mesh.ds.node_to_edge()

# Boundary flags
node_bd_flag = mesh.ds.boundary_flag('node')
edge_bd_flag = mesh.ds.boundary_flag('edge')
face_bd_flag = mesh.ds.boundary_flag('face')
cell_bd_flag = mesh.ds.boundary_flag('cell')

# Indices of boundary entities
bd_node_idx = np.where(node_bd_flag)[0]
bd_edge_idx = np.where(edge_bd_flag)[0]
bd_face_idx = np.where(face_bd_flag)[0]
bd_cell_idx = np.where(cell_bd_flag)[0]

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in edge_nodes:
    pts = node_coords[edge]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='k', linewidth=1)

# Plot nodes and their indices
ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color='r', s=50)
for i, (x, y, z) in enumerate(node_coords):
    ax.text(x, y, z, f'N{i}', color='r', fontsize=12)

# Plot edge indices at barycenters
for i, (x, y, z) in enumerate(edge_barycenters):
    ax.text(x, y, z, f'E{i}', color='b', fontsize=12)

# Plot cell indices at barycenters
for i, (x, y, z) in enumerate(cell_barycenters):
    ax.text(x, y, z, f'C{i}', color='g', fontsize=14)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tetrahedron Mesh with Node, Edge, and Cell Indices')
plt.tight_layout()
plt.show()
```