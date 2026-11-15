```python
from nodegraphqt import NodeGraph, NodeGraphWidget, GroupNode

def zoom_in(graph: NodeGraph, factor: float = 1.2):
    zoom = graph.get_zoom()
    graph.set_zoom(zoom * factor)

def zoom_out(graph: NodeGraph, factor: float = 1.2):
    zoom = graph.get_zoom()
    graph.set_zoom(zoom / factor)

def reset_zoom(graph: NodeGraph):
    graph.reset_zoom()

def set_layout_direction(graph: NodeGraph, direction: str):
    if direction.lower() == 'horizontal':
        graph.set_layout_direction('horizontal')
    elif direction.lower() == 'vertical':
        graph.set_layout_direction('vertical')

def open_session(graph: NodeGraph, file_path: str):
    graph.load_session(file_path)

def import_session(graph: NodeGraph, file_path: str):
    graph.import_session(file_path)

def save_session(graph: NodeGraph, file_path: str):
    graph.save_session(file_path)

def clear_session(graph: NodeGraph):
    graph.clear_session()

def clear_undo_history(graph: NodeGraph):
    graph.undo_stack().clear()

def copy_nodes(graph: NodeGraph):
    graph.copy_nodes()

def cut_nodes(graph: NodeGraph):
    graph.cut_nodes()

def paste_nodes(graph: NodeGraph):
    graph.paste_nodes()

def delete_nodes(graph: NodeGraph):
    graph.delete_nodes()

def extract_nodes(graph: NodeGraph):
    graph.extract_nodes()

def clear_connections(graph: NodeGraph):
    nodes = graph.selected_nodes()
    for node in nodes:
        for port in node.input_ports():
            port.clear_connections()
        for port in node.output_ports():
            port.clear_connections()

def select_all(graph: NodeGraph):
    graph.select_all()

def clear_selection(graph: NodeGraph):
    graph.clear_selection()

def invert_selection(graph: NodeGraph):
    graph.invert_selection()

def disable_nodes(graph: NodeGraph):
    graph.disable_nodes(graph.selected_nodes())

def duplicate_nodes(graph: NodeGraph):
    graph.duplicate_nodes()

def expand_group_nodes(graph: NodeGraph):
    for node in graph.selected_nodes():
        if isinstance(node, GroupNode):
            graph.expand_group_node(node)

def fit_zoom_to_selected(graph: NodeGraph):
    nodes = graph.selected_nodes()
    if nodes:
        graph.zoom_to_nodes(nodes)

def show_undo_list(graph: NodeGraph):
    graph.show_undo_view()

def set_pipe_style(graph: NodeGraph, style: str):
    if style.lower() in ['curved', 'straight', 'angled']:
        graph.set_pipe_style(style.lower())

def set_background_grid(graph: NodeGraph, grid_type: str):
    if grid_type.lower() in ['none', 'dots', 'lines']:
        graph.set_grid_mode(grid_type.lower())

def auto_layout_nodes(graph: NodeGraph, direction: str = 'downstream'):
    nodes = graph.selected_nodes()
    if direction.lower() == 'downstream':
        graph.auto_layout_nodes(nodes, downstream=True)
    elif direction.lower() == 'upstream':
        graph.auto_layout_nodes(nodes, downstream=False)

def toggle_node_search(graph: NodeGraph):
    graph.toggle_node_search()
```