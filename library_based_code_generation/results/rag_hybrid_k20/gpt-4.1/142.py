```python
import urwid
import sys
import os

class LazyFileListWalker(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self.lines = []
        self._load_file()
        self.focus = 0

    def _load_file(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.lines = [line.rstrip('\n') for line in f]
        else:
            self.lines = ['']
        self._widgets = [urwid.Edit(edit_text=line) for line in self.lines]

    def _get_widget(self, pos):
        if pos < 0 or pos >= len(self._widgets):
            return None, None
        return self._widgets[pos], pos

    def get_focus(self):
        return self._get_widget(self.focus)

    def set_focus(self, focus):
        if 0 <= focus < len(self._widgets):
            self.focus = focus
            self._modified()

    def get_next(self, pos):
        return self._get_widget(pos + 1)

    def get_prev(self, pos):
        return self._get_widget(pos - 1)

    def insert_line(self, pos, text=''):
        self.lines.insert(pos, text)
        self._widgets.insert(pos, urwid.Edit(edit_text=text))
        self._modified()

    def remove_line(self, pos):
        if len(self.lines) > 1:
            del self.lines[pos]
            del self._widgets[pos]
            self._modified()

    def get_all_text(self):
        return [w.edit_text for w in self._widgets]

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for line in self.get_all_text():
                f.write(line + '\n')

class TextEditor(urwid.WidgetWrap):
    def __init__(self, filename):
        self.filename = filename
        self.walker = LazyFileListWalker(filename)
        self.listbox = urwid.ListBox(self.walker)
        self.footer = urwid.Text(
            u"Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move"
        )
        self.frame = urwid.Frame(self.listbox, footer=self.footer)
        super().__init__(self.frame)

    def keypress(self, size, key):
        if key == 'ctrl s':
            self.walker.save()
            self.footer.set_text("Saved! Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move")
            return
        elif key == 'ctrl q':
            raise urwid.ExitMainLoop()
        elif key == 'enter':
            self._split_line()
            return
        elif key == 'backspace':
            self._backspace()
            return
        elif key == 'delete':
            self._delete()
            return
        elif key in ('left', 'right'):
            self._move_cursor(key)
            return
        else:
            return super().keypress(size, key)

    def _get_focus_widget(self):
        widget, pos = self.walker.get_focus()
        return widget, pos

    def _split_line(self):
        widget, pos = self._get_focus_widget()
        edit_text = widget.edit_text
        edit_pos = widget.edit_pos
        before = edit_text[:edit_pos]
        after = edit_text[edit_pos:]
        widget.set_edit_text(before)
        widget.edit_pos = len(before)
        self.walker.insert_line(pos + 1, after)
        self.walker.set_focus(pos + 1)

    def _backspace(self):
        widget, pos = self._get_focus_widget()
        if widget.edit_pos > 0:
            widget.keypress((0,), 'backspace')
        elif pos > 0:
            prev_widget, prev_pos = self.walker._get_widget(pos - 1)
            prev_text = prev_widget.edit_text
            prev_len = len(prev_text)
            prev_widget.set_edit_text(prev_text + widget.edit_text)
            prev_widget.edit_pos = prev_len
            self.walker.remove_line(pos)
            self.walker.set_focus(prev_pos)

    def _delete(self):
        widget, pos = self._get_focus_widget()
        if widget.edit_pos < len(widget.edit_text):
            widget.keypress((0,), 'delete')
        elif pos < len(self.walker._widgets) - 1:
            next_widget, next_pos = self.walker._get_widget(pos + 1)
            widget.set_edit_text(widget.edit_text + next_widget.edit_text)
            widget.edit_pos = len(widget.edit_text)
            self.walker.remove_line(pos + 1)
            self.walker.set_focus(pos)

    def _move_cursor(self, direction):
        widget, pos = self._get_focus_widget()
        if direction == 'left':
            if widget.edit_pos > 0:
                widget.edit_pos -= 1
            elif pos > 0:
                prev_widget, prev_pos = self.walker._get_widget(pos - 1)
                self.walker.set_focus(prev_pos)
                prev_widget.edit_pos = len(prev_widget.edit_text)
        elif direction == 'right':
            if widget.edit_pos < len(widget.edit_text):
                widget.edit_pos += 1
            elif pos < len(self.walker._widgets) - 1:
                next_widget, next_pos = self.walker._get_widget(pos + 1)
                self.walker.set_focus(next_pos)
                next_widget.edit_pos = 0

def main(filename):
    editor = TextEditor(filename)
    urwid.MainLoop(editor, palette=[('reversed', 'standout', '')]).run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python text_editor.py <filename>")
        sys.exit(1)
    main(sys.argv[1])
```