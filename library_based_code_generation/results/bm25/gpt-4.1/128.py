```python
from nodegraphqt import NodeGraph, Node, BaseNode
from nodegraphqt.constants import LayoutDirection, PipeLayout, GridMode

def zoom_in(graph: NodeGraph, factor: float = 1.2):
    zoom = graph.get_zoom()
    graph.set_zoom(zoom * factor)

def zoom_out(graph: NodeGraph, factor: float = 1.2):
    zoom = graph.get_zoom()
    graph.set_zoom(zoom / factor)

def reset_zoom(graph: NodeGraph):
    graph.reset_zoom()

def set_layout_direction_horizontal(graph: NodeGraph):
    graph.set_layout_direction(LayoutDirection.HORIZONTAL)

def set_layout_direction_vertical(graph: NodeGraph):
    graph.set_layout_direction(LayoutDirection.VERTICAL)

def open_session(graph: NodeGraph, file_path: str):
    graph.load_session(file_path)

def import_session(graph: NodeGraph, file_path: str):
    graph.import_session(file_path)

def save_session(graph: NodeGraph, file_path: str):
    graph.save_session(file_path)

def clear_session(graph: NodeGraph):
    graph.clear_session()

def clear_undo_history(graph: NodeGraph):
    graph.clear_undo_stack()

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

def clear_connections_of_nodes(graph: NodeGraph):
    for node in graph.selected_nodes():
        node.clear_connections()

def select_all_nodes(graph: NodeGraph):
    for node in graph.all_nodes():
        node.set_selected(True)

def clear_selection(graph: NodeGraph):
    graph.clear_selection()

def invert_selection(graph: NodeGraph):
    all_nodes = set(graph.all_nodes())
    selected = set(graph.selected_nodes())
    for node in all_nodes:
        node.set_selected(node not in selected)

def disable_nodes(graph: NodeGraph):
    for node in graph.selected_nodes():
        node.set_disabled(True)

def duplicate_nodes(graph: NodeGraph):
    graph.duplicate_nodes()

def expand_group_nodes(graph: NodeGraph):
    for node in graph.selected_nodes():
        if hasattr(node, 'expand_group'):
            node.expand_group()

def fit_zoom_to_selected(graph: NodeGraph):
    graph.fit_to_selection()

def show_undo_list(graph: NodeGraph):
    graph.show_undo_view()

def set_pipe_style_curved(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayout.CURVED)

def set_pipe_style_straight(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayout.STRAIGHT)

def set_pipe_style_angled(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayout.ANGLED)

def set_background_grid_none(graph: NodeGraph):
    graph.set_grid_mode(GridMode.NONE)

def set_background_grid_dots(graph: NodeGraph):
    graph.set_grid_mode(GridMode.DOTS)

def set_background_grid_lines(graph: NodeGraph):
    graph.set_grid_mode(GridMode.LINES)

def auto_layout_nodes_downstream(graph: NodeGraph):
    graph.auto_layout_nodes(direction='downstream')

def auto_layout_nodes_upstream(graph: NodeGraph):
    graph.auto_layout_nodes(direction='upstream')

def toggle_node_search(graph: NodeGraph):
    graph.toggle_node_search()
```