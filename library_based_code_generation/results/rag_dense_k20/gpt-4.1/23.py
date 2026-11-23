```python
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.core.window import Window
from kivy.properties import ListProperty

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    selected_nodes = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._typed_number = ''
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        if self._keyboard:
            self._keyboard.bind(on_key_down=self._on_key_down, on_key_up=self._on_key_up)
        self.bind(selected_nodes=self.on_selection_change)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard.unbind(on_key_up=self._on_key_up)
        self._keyboard = None

    def _on_key_down(self, keyboard, keycode, text, modifiers):
        key, key_str = keycode
        if key_str.isdigit():
            self._typed_number += key_str
            self.navigate_to_node_by_number(self._typed_number)
        elif key_str == 'enter':
            self.select_node_by_number(self._typed_number)
            self._typed_number = ''
        elif key_str == 'escape':
            self._typed_number = ''
        elif key_str == 'up':
            self.move_selection('up')
        elif key_str == 'down':
            self.move_selection('down')
        elif key_str == 'left':
            self.move_selection('left')
        elif key_str == 'right':
            self.move_selection('right')
        return True

    def _on_key_up(self, keyboard, keycode):
        return True

    def navigate_to_node_by_number(self, number_str):
        for child in self.children:
            if child.text == number_str:
                self.clear_selection()
                self.select_node(child)
                break

    def select_node_by_number(self, number_str):
        for child in self.children:
            if child.text == number_str:
                self.select_node(child)
                break

    def select_node(self, node):
        if node not in self.selected_nodes:
            self.selected_nodes.append(node)
            self.apply_selection(node, True)

    def deselect_node(self, node):
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            self.apply_selection(node, False)

    def clear_selection(self):
        for node in self.selected_nodes[:]:
            self.apply_selection(node, False)
        self.selected_nodes = []

    def apply_selection(self, node, selected):
        if selected:
            node.background_color = (0.2, 0.6, 1, 1)
        else:
            node.background_color = (1, 1, 1, 1)

    def on_selection_change(self, instance, value):
        print("Selected nodes:", [node.text for node in self.selected_nodes])

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        for child in self.children:
            if child.collide_point(*touch.pos):
                if child in self.selected_nodes:
                    self.deselect_node(child)
                else:
                    self.select_node(child)
                return True
        return super().on_touch_down(touch)

    def move_selection(self, direction):
        if not self.selected_nodes:
            return
        current = self.selected_nodes[-1]
        idx = self.children.index(current)
        cols = self.cols
        rows = self.rows
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
            self.clear_selection()
            self.select_node(self.children[new_idx])

class TestApp(App):
    def build(self):
        grid = SelectableGrid(cols=8, rows=5, spacing=5, padding=5, size_hint=(1, 1))
        for i in range(1, 41):
            btn = Button(text=str(i), background_color=(1, 1, 1, 1))
            btn.bind(on_touch_down=grid.on_touch_down)
            grid.add_widget(btn)
        return grid

if __name__ == '__main__':
    TestApp().run()
```