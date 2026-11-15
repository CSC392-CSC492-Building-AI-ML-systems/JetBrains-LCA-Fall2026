```python
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior
from kivy.properties import ListProperty
from kivy.core.window import Window

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    selected_nodes = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._typed_number = ""
        self._typed_number_time = 0
        Window.bind(on_key_down=self._on_key_down, on_key_up=self._on_key_up)
        self.bind(selected_nodes=self.on_selection_change)

    def on_selection_change(self, *args):
        print("Selected nodes:", [btn.text for btn in self.selected_nodes])

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        if codepoint and codepoint.isdigit():
            from time import time
            now = time()
            if now - self._typed_number_time > 1:
                self._typed_number = ""
            self._typed_number_time = now
            self._typed_number += codepoint
            self.navigate_to_node_by_number(self._typed_number)
        else:
            self.select_with_key_down(key, modifiers)

    def _on_key_up(self, window, key, scancode):
        self.select_with_key_up(key)

    def navigate_to_node_by_number(self, number_str):
        for child in self.children:
            if child.text == number_str:
                self.clear_selection()
                self.select_node(child)
                break

    def select_node(self, node):
        if node not in self.selected_nodes:
            self.selected_nodes.append(node)
            node.background_color = (0.2, 0.6, 1, 1)

    def deselect_node(self, node):
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            node.background_color = (1, 1, 1, 1)

    def clear_selection(self):
        for node in self.selected_nodes[:]:
            self.deselect_node(node)

    def select_with_touch(self, node, touch=None):
        if node.collide_point(*touch.pos):
            if node in self.selected_nodes:
                self.deselect_node(node)
            else:
                self.select_node(node)

    def on_touch_down(self, touch):
        for child in self.children:
            if child.collide_point(*touch.pos):
                self.select_with_touch(child, touch)
                return True
        return super().on_touch_down(touch)

class TestApp(App):
    def build(self):
        grid = SelectableGrid(cols=5, spacing=5, padding=5, size_hint=(1, 1))
        for i in range(1, 41):
            btn = Button(text=str(i), background_color=(1, 1, 1, 1))
            btn.bind(on_touch_down=lambda instance, touch, btn=btn: grid.on_touch_down(touch))
            grid.add_widget(btn)
        return grid

if __name__ == '__main__':
    TestApp().run()
```