```python
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.core.window import Window
from kivy.properties import ListProperty
from kivy.clock import Clock

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    selected_color = ListProperty([0.2, 0.6, 1, 1])
    normal_color = ListProperty([1, 1, 1, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._number_buffer = ''
        self._number_buffer_ev = None
        self.multiselect = True
        self.touch_deselect_last = False
        self.bind(selected_nodes=self.on_selection_change)
        Window.bind(on_key_down=self._on_key_down, on_key_up=self._on_key_up)

    def add_widget(self, widget, index=0, canvas=None):
        super().add_widget(widget, index=index, canvas=canvas)
        widget.background_color = self.normal_color
        widget.bind(on_touch_down=self.on_node_touch_down)
        widget.bind(on_touch_up=self.on_node_touch_up)

    def select_node(self, node):
        super().select_node(node)
        node.background_color = self.selected_color

    def deselect_node(self, node):
        super().deselect_node(node)
        node.background_color = self.normal_color

    def on_selection_change(self, *args):
        print("Selected nodes:", [node.text for node in self.selected_nodes])

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        if codepoint and codepoint.isdigit():
            self._number_buffer += codepoint
            if self._number_buffer_ev:
                self._number_buffer_ev.cancel()
            self._number_buffer_ev = Clock.schedule_once(self._on_number_buffer_timeout, 0.7)
            return True
        elif key == 273:  # up
            self.move_selection('up')
            return True
        elif key == 274:  # down
            self.move_selection('down')
            return True
        elif key == 275:  # right
            self.move_selection('right')
            return True
        elif key == 276:  # left
            self.move_selection('left')
            return True
        elif key == 32:  # space
            self.toggle_selection()
            return True
        return False

    def _on_key_up(self, window, key, scancode):
        return False

    def _on_number_buffer_timeout(self, dt):
        if self._number_buffer:
            self.navigate_to_node_by_number(self._number_buffer)
            self._number_buffer = ''

    def navigate_to_node_by_number(self, number_str):
        for node in self.children:
            if getattr(node, 'text', None) == number_str:
                self.select_with_touch(node, None)
                self.show_node(node)
                break

    def move_selection(self, direction):
        if not self.selected_nodes:
            if self.children:
                self.select_node(self.children[-1])
            return
        current = self.selected_nodes[-1]
        idx = self.children.index(current)
        cols = self.cols
        rows = self.rows if self.rows else (len(self.children) // self.cols)
        if direction == 'up':
            new_idx = idx + cols
        elif direction == 'down':
            new_idx = idx - cols
        elif direction == 'left':
            new_idx = idx + 1
        elif direction == 'right':
            new_idx = idx - 1
        else:
            return
        if 0 <= new_idx < len(self.children):
            self.deselect_node(current)
            self.select_node(self.children[new_idx])
            self.show_node(self.children[new_idx])

    def toggle_selection(self):
        if not self.selected_nodes:
            return
        node = self.selected_nodes[-1]
        if node in self.selected_nodes:
            self.deselect_node(node)
        else:
            self.select_node(node)

    def show_node(self, node):
        # Placeholder for scrolling into view if needed
        pass

    def on_node_touch_down(self, node, touch):
        if node.collide_point(*touch.pos):
            self.select_with_touch(node, touch)
            return True
        return False

    def on_node_touch_up(self, node, touch):
        return False

class TestApp(App):
    def build(self):
        grid = SelectableGrid(cols=5, spacing=5, padding=5, size_hint=(1, 1))
        for i in range(1, 41):
            btn = Button(text=str(i), background_normal='', background_color=[1, 1, 1, 1])
            grid.add_widget(btn)
        return grid

if __name__ == '__main__':
    TestApp().run()
```