```python
from nodegraphqt import NodeGraph, NodeGraphWidget, NodeGraphCommand, GraphAction
from nodegraphqt.constants import PipeLayoutEnum, GridModeEnum

def zoom_in(graph: NodeGraph):
    graph.zoom_in()

def zoom_out(graph: NodeGraph):
    graph.zoom_out()

def reset_zoom(graph: NodeGraph):
    graph.reset_zoom()

def set_layout_direction_horizontal(graph: NodeGraph):
    graph._draw_node_horizontal()

def set_layout_direction_vertical(graph: NodeGraph):
    graph._draw_node_vertical()

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
    cut_nodes(graph)

def paste_nodes(graph: NodeGraph):
    graph.paste_nodes()

def delete_nodes(graph: NodeGraph):
    graph.delete_nodes()

def extract_nodes(graph: NodeGraph):
    graph.extract_nodes()

def clear_connections(graph: NodeGraph):
    graph.clear_connections()

def select_all_nodes(graph: NodeGraph):
    graph.select_all()

def clear_selection(graph: NodeGraph):
    graph.clear_selection()

def invert_selection(graph: NodeGraph):
    graph.invert_selection()

def disable_nodes(graph: NodeGraph):
    disable_nodes(graph)

def duplicate_nodes(graph: NodeGraph):
    graph.duplicate_nodes()

def expand_group_node(graph: NodeGraph):
    expand_group_node(graph)

def fit_zoom_to_selected_nodes(graph: NodeGraph):
    graph.zoom_to_nodes(graph.selected_nodes())

def show_undo_list(graph: NodeGraph):
    graph.undo_view().show()

def set_pipe_style_curved(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayoutEnum.CURVED)

def set_pipe_style_straight(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayoutEnum.STRAIGHT)

def set_pipe_style_angled(graph: NodeGraph):
    graph.set_pipe_layout(PipeLayoutEnum.ANGLED)

def set_background_grid_none(graph: NodeGraph):
    graph.set_grid_mode(GridModeEnum.NONE)

def set_background_grid_dots(graph: NodeGraph):
    graph.set_grid_mode(GridModeEnum.DOTS)

def set_background_grid_lines(graph: NodeGraph):
    graph.set_grid_mode(GridModeEnum.LINES)

def auto_layout_nodes_downstream(graph: NodeGraph):
    auto_layout_nodes(graph, direction='downstream')

def auto_layout_nodes_upstream(graph: NodeGraph):
    auto_layout_nodes(graph, direction='upstream')

def toggle_node_search(graph: NodeGraph):
    toggle_node_search(graph)
```