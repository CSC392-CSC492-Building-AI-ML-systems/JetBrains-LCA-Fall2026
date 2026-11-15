```python
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior
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
        key = keycode[1]
        if key.isdigit():
            self._typed_number += key
            self.navigate_to_node_by_number(self._typed_number)
        elif key == 'enter':
            self.select_with_key_down()
        elif key == 'up':
            self.select_with_key_up()
        elif key == 'down':
            self.select_with_key_down()
        elif key == 'escape':
            self._typed_number = ''
        return True

    def _on_key_up(self, keyboard, keycode):
        return True

    def navigate_to_node_by_number(self, number_str):
        for child in self.children:
            if getattr(child, 'text', None) == number_str:
                self.clear_selection()
                self.select_node(child)
                self._typed_number = ''
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
        for node in list(self.selected_nodes):
            self.deselect_node(node)

    def apply_selection(self, node, selected):
        if selected:
            node.background_color = (0.2, 0.6, 1, 1)
        else:
            node.background_color = (1, 1, 1, 1)

    def select_with_key_down(self):
        idx = self.get_index_of_node(self.selected_nodes[-1]) if self.selected_nodes else -1
        if idx + 1 < len(self.children):
            self.clear_selection()
            node = self.children[idx + 1]
            self.select_node(node)

    def select_with_key_up(self):
        idx = self.get_index_of_node(self.selected_nodes[-1]) if self.selected_nodes else len(self.children)
        if idx - 1 >= 0:
            self.clear_selection()
            node = self.children[idx - 1]
            self.select_node(node)

    def get_index_of_node(self, node):
        try:
            return self.children.index(node)
        except ValueError:
            return -1

    def on_selection_change(self, *args):
        print("Selected nodes:", [getattr(node, 'text', '') for node in self.selected_nodes])

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        for child in self.children:
            if child.collide_point(*child.to_widget(*touch.pos)):
                if child in self.selected_nodes:
                    self.deselect_node(child)
                else:
                    self.clear_selection()
                    self.select_node(child)
                return True
        return super().on_touch_down(touch)

class TestApp(App):
    def build(self):
        grid = SelectableGrid(cols=5, spacing=5, padding=5, size_hint=(1, 1))
        for i in range(1, 41):
            btn = Button(text=str(i), background_color=(1, 1, 1, 1))
            btn.bind(on_touch_down=grid.on_touch_down)
            grid.add_widget(btn)
        return grid

if __name__ == '__main__':
    TestApp().run()
```