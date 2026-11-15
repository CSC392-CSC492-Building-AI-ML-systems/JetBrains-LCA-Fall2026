```python
import urwid
import sys
import os

class LazyFileListWalker(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self._lines = []
        self._widgets = {}
        self._load_file()
        self.focus = 0

    def _load_file(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                self._lines = [line.rstrip('\n') for line in f]
        else:
            self._lines = ['']
        self._widgets = {}

    def get_focus(self):
        return self._get_widget(self.focus), self.focus

    def set_focus(self, position):
        if 0 <= position < len(self._lines):
            self.focus = position
            self._modified()

    def get_next(self, position):
        if position + 1 < len(self._lines):
            return self._get_widget(position + 1), position + 1
        return None, None

    def get_prev(self, position):
        if position > 0:
            return self._get_widget(position - 1), position - 1
        return None, None

    def _get_widget(self, position):
        if position not in self._widgets:
            edit = urwid.Edit(multiline=False)
            edit.set_edit_text(self._lines[position])
            edit.set_edit_pos(len(self._lines[position]))
            urwid.connect_signal(edit, 'change', self._on_edit_change, position)
            self._widgets[position] = edit
        return self._widgets[position]

    def _on_edit_change(self, edit, new_text, position):
        self._lines[position] = new_text

    def insert_line(self, position, text=''):
        self._lines.insert(position, text)
        self._widgets = {k if k < position else k+1: v for k, v in self._widgets.items()}
        self._widgets.pop(position, None)
        self._modified()

    def delete_line(self, position):
        if len(self._lines) > 1:
            self._lines.pop(position)
            self._widgets = {k if k < position else k-1: v for k, v in self._widgets.items() if k != position}
            self._modified()

    def combine_lines(self, position):
        if position + 1 < len(self._lines):
            self._lines[position] += self._lines[position + 1]
            self.delete_line(position + 1)
            self._widgets.pop(position, None)
            self._modified()

    def split_line(self, position, split_pos):
        line = self._lines[position]
        self._lines[position] = line[:split_pos]
        self.insert_line(position + 1, line[split_pos:])
        self._widgets.pop(position, None)
        self._modified()

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for line in self._lines:
                f.write(line + '\n')

class TextEditor(urwid.WidgetWrap):
    def __init__(self, filename):
        self.filename = filename
        self.walker = LazyFileListWalker(filename)
        self.listbox = urwid.ListBox(self.walker)
        self.footer = urwid.Text(
            u"Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move"
        )
        frame = urwid.Frame(self.listbox, footer=self.footer)
        super().__init__(frame)

    def keypress(self, size, key):
        if key == 'ctrl s':
            self.walker.save()
            self.footer.set_text("Saved! Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move")
            return
        if key == 'ctrl q':
            raise urwid.ExitMainLoop()
        focus_widget, pos = self.walker.get_focus()
        if not isinstance(focus_widget, urwid.Edit):
            return super().keypress(size, key)
        edit = focus_widget
        edit_text = edit.get_edit_text()
        edit_pos = edit.edit_pos

        if key == 'enter':
            self.walker.split_line(pos, edit_pos)
            self.walker.set_focus(pos + 1)
            return
        if key == 'backspace':
            if edit_pos == 0:
                if pos > 0:
                    prev_text = self.walker._lines[pos - 1]
                    self.walker.set_focus(pos - 1)
                    self.walker._lines[pos - 1] += edit_text
                    self.walker.delete_line(pos)
                    prev_widget = self.walker._get_widget(pos - 1)
                    prev_widget.set_edit_text(self.walker._lines[pos - 1])
                    prev_widget.set_edit_pos(len(prev_text))
            else:
                return edit.keypress(size, key)
            return
        if key == 'delete':
            if edit_pos == len(edit_text):
                if pos + 1 < len(self.walker._lines):
                    self.walker._lines[pos] += self.walker._lines[pos + 1]
                    self.walker.delete_line(pos + 1)
                    edit.set_edit_text(self.walker._lines[pos])
                    edit.set_edit_pos(edit_pos)
            else:
                return edit.keypress(size, key)
            return
        if key == 'left':
            if edit_pos == 0 and pos > 0:
                self.walker.set_focus(pos - 1)
                prev_widget = self.walker._get_widget(pos - 1)
                prev_widget.set_edit_pos(len(self.walker._lines[pos - 1]))
            else:
                return edit.keypress(size, key)
            return
        if key == 'right':
            if edit_pos == len(edit_text) and pos + 1 < len(self.walker._lines):
                self.walker.set_focus(pos + 1)
                next_widget = self.walker._get_widget(pos + 1)
                next_widget.set_edit_pos(0)
            else:
                return edit.keypress(size, key)
            return
        return edit.keypress(size, key)

def main(filename):
    editor = TextEditor(filename)
    urwid.MainLoop(editor, palette=[('reversed', 'standout', '')]).run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python text_editor.py <filename>")
        sys.exit(1)
    main(sys.argv[1])
```