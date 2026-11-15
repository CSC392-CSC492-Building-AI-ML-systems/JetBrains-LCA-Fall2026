```python
import urwid
import sys
import os

class LazyTextWalker(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self.lines = []
        self.focus = 0
        self._file_loaded = False
        self._load_file()

    def _load_file(self):
        if not self._file_loaded:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.lines = [line.rstrip('\n') for line in f]
            else:
                self.lines = ['']
            self._file_loaded = True

    def get_focus(self):
        return self._get_at_pos(self.focus)

    def set_focus(self, focus):
        self.focus = focus
        self._modified()

    def get_next(self, pos):
        return self._get_at_pos(pos + 1)

    def get_prev(self, pos):
        return self._get_at_pos(pos - 1)

    def _get_at_pos(self, pos):
        if 0 <= pos < len(self.lines):
            return (urwid.AttrMap(urwid.Text(self.lines[pos], wrap='any'), None), pos)
        return (None, None)

    def insert_line(self, pos, text=''):
        self.lines.insert(pos, text)
        self._modified()

    def delete_line(self, pos):
        if 0 <= pos < len(self.lines):
            del self.lines[pos]
            if not self.lines:
                self.lines.append('')
            self._modified()

    def set_line(self, pos, text):
        if 0 <= pos < len(self.lines):
            self.lines[pos] = text
            self._modified()

    def get_line(self, pos):
        if 0 <= pos < len(self.lines):
            return self.lines[pos]
        return ''

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for line in self.lines:
                f.write(line + '\n')

class TextEditor(urwid.WidgetWrap):
    def __init__(self, filename):
        self.filename = filename
        self.walker = LazyTextWalker(filename)
        self.listbox = urwid.ListBox(self.walker)
        self.footer = urwid.Text(
            u"Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move Cursor"
        )
        self.frame = urwid.Frame(self.listbox, footer=urwid.AttrMap(self.footer, 'footer'))
        self.cursor_col = 0
        self._update_focus_widget()
        super().__init__(self.frame)

    def _update_focus_widget(self):
        focus_widget, focus_pos = self.walker.get_focus()
        if focus_widget is not None:
            line = self.walker.get_line(focus_pos)
            self._set_edit_widget(line, focus_pos)
        else:
            self.walker.insert_line(0, '')
            self._set_edit_widget('', 0)

    def _set_edit_widget(self, text, pos):
        edit = urwid.Edit('', edit_text=text, wrap='any')
        edit.set_edit_pos(min(self.cursor_col, len(text)))
        self.listbox.body[self.walker.focus] = urwid.AttrMap(edit, None)
        self._invalidate()

    def keypress(self, size, key):
        focus_widget, focus_pos = self.walker.get_focus()
        if isinstance(focus_widget.base_widget, urwid.Edit):
            edit = focus_widget.base_widget
            line = edit.edit_text
            pos = edit.edit_pos

            if key == 'ctrl s':
                self.walker.set_line(focus_pos, line)
                self.walker.save()
                self.footer.set_text("Saved.")
                return

            if key == 'ctrl q':
                raise urwid.ExitMainLoop()

            if key == 'enter':
                self.walker.set_line(focus_pos, line[:pos])
                self.walker.insert_line(focus_pos + 1, line[pos:])
                self.walker.set_focus(focus_pos + 1)
                self.cursor_col = 0
                self._update_focus_widget()
                return

            if key == 'backspace':
                if pos > 0:
                    new_line = line[:pos-1] + line[pos:]
                    self.walker.set_line(focus_pos, new_line)
                    self.cursor_col = pos - 1
                elif focus_pos > 0:
                    prev_line = self.walker.get_line(focus_pos - 1)
                    self.walker.set_line(focus_pos - 1, prev_line + line)
                    self.walker.delete_line(focus_pos)
                    self.walker.set_focus(focus_pos - 1)
                    self.cursor_col = len(prev_line)
                self._update_focus_widget()
                return

            if key == 'delete':
                if pos < len(line):
                    new_line = line[:pos] + line[pos+1:]
                    self.walker.set_line(focus_pos, new_line)
                    self.cursor_col = pos
                elif focus_pos < len(self.walker.lines) - 1:
                    next_line = self.walker.get_line(focus_pos + 1)
                    self.walker.set_line(focus_pos, line + next_line)
                    self.walker.delete_line(focus_pos + 1)
                    self.cursor_col = pos
                self._update_focus_widget()
                return

            if key == 'left':
                if pos > 0:
                    self.cursor_col = pos - 1
                elif focus_pos > 0:
                    prev_line = self.walker.get_line(focus_pos - 1)
                    self.walker.set_focus(focus_pos - 1)
                    self.cursor_col = len(prev_line)
                self._update_focus_widget()
                return

            if key == 'right':
                if pos < len(line):
                    self.cursor_col = pos + 1
                elif focus_pos < len(self.walker.lines) - 1:
                    self.walker.set_focus(focus_pos + 1)
                    self.cursor_col = 0
                self._update_focus_widget()
                return

            # Default: let Edit handle the keypress
            edit.keypress(size, key)
            self.walker.set_line(focus_pos, edit.edit_text)
            self.cursor_col = edit.edit_pos
            self._update_focus_widget()
        else:
            return super().keypress(size, key)

def main(filename):
    palette = [
        ('footer', 'black', 'light gray'),
    ]
    editor = TextEditor(filename)
    urwid.MainLoop(editor, palette=palette).run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python text_editor.py <filename>")
        sys.exit(1)
    main(sys.argv[1])
```