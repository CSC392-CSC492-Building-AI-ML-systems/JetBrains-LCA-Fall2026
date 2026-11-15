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
        self.widgets = [urwid.Edit(edit_text=line) for line in self.lines]

    def get_focus(self):
        return self._get_at_pos(self.focus)

    def set_focus(self, position):
        self.focus = position
        self._modified()

    def get_next(self, position):
        return self._get_at_pos(position + 1)

    def get_prev(self, position):
        return self._get_at_pos(position - 1)

    def _get_at_pos(self, pos):
        if 0 <= pos < len(self.widgets):
            return self.widgets[pos], pos
        return None, None

    def insert_line(self, pos, text=''):
        self.lines.insert(pos, text)
        self.widgets.insert(pos, urwid.Edit(edit_text=text))
        self._modified()

    def remove_line(self, pos):
        if len(self.lines) > 1:
            del self.lines[pos]
            del self.widgets[pos]
            self._modified()

    def set_line(self, pos, text):
        self.lines[pos] = text
        self.widgets[pos].set_edit_text(text)
        self._modified()

    def get_line(self, pos):
        return self.lines[pos]

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for line in self.lines:
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
        elif key == 'left':
            self._move_left()
            return
        elif key == 'right':
            self._move_right()
            return
        else:
            return super().keypress(size, key)

    def _get_focus_widget(self):
        widget, pos = self.walker.get_focus()
        return widget, pos

    def _split_line(self):
        widget, pos = self._get_focus_widget()
        text = widget.edit_text
        cursor = widget.edit_pos
        before = text[:cursor]
        after = text[cursor:]
        self.walker.set_line(pos, before)
        self.walker.insert_line(pos + 1, after)
        self.walker.set_focus(pos + 1)
        self.walker.widgets[pos + 1].edit_pos = 0

    def _backspace(self):
        widget, pos = self._get_focus_widget()
        if widget.edit_pos == 0:
            if pos > 0:
                prev_widget = self.walker.widgets[pos - 1]
                prev_text = prev_widget.edit_text
                curr_text = widget.edit_text
                new_text = prev_text + curr_text
                self.walker.set_line(pos - 1, new_text)
                self.walker.remove_line(pos)
                self.walker.set_focus(pos - 1)
                prev_widget.edit_pos = len(prev_text)
        else:
            text = widget.edit_text
            cursor = widget.edit_pos
            new_text = text[:cursor - 1] + text[cursor:]
            self.walker.set_line(pos, new_text)
            widget.edit_pos = cursor - 1

    def _delete(self):
        widget, pos = self._get_focus_widget()
        text = widget.edit_text
        cursor = widget.edit_pos
        if cursor == len(text):
            if pos < len(self.walker.widgets) - 1:
                next_widget = self.walker.widgets[pos + 1]
                next_text = next_widget.edit_text
                new_text = text + next_text
                self.walker.set_line(pos, new_text)
                self.walker.remove_line(pos + 1)
                widget.edit_pos = cursor
        else:
            new_text = text[:cursor] + text[cursor + 1:]
            self.walker.set_line(pos, new_text)
            widget.edit_pos = cursor

    def _move_left(self):
        widget, pos = self._get_focus_widget()
        if widget.edit_pos > 0:
            widget.edit_pos -= 1
        elif pos > 0:
            prev_widget = self.walker.widgets[pos - 1]
            self.walker.set_focus(pos - 1)
            prev_widget.edit_pos = len(prev_widget.edit_text)

    def _move_right(self):
        widget, pos = self._get_focus_widget()
        if widget.edit_pos < len(widget.edit_text):
            widget.edit_pos += 1
        elif pos < len(self.walker.widgets) - 1:
            next_widget = self.walker.widgets[pos + 1]
            self.walker.set_focus(pos + 1)
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