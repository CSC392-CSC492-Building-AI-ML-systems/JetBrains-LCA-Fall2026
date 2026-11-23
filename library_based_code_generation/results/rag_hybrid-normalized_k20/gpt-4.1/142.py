```python
import urwid
import sys
import os

class LazyFileListWalker(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self.lines = []
        self.loaded = False
        self.focus = 0

    def load_file(self):
        if not self.loaded:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.lines = [urwid.Edit('', line.rstrip('\n')) for line in f]
            else:
                self.lines = [urwid.Edit('')]
            if not self.lines:
                self.lines = [urwid.Edit('')]
            self.loaded = True

    def get_focus(self):
        self.load_file()
        if not self.lines:
            return None, None
        return self.lines[self.focus], self.focus

    def set_focus(self, position):
        self.load_file()
        if 0 <= position < len(self.lines):
            self.focus = position
            self._modified()

    def get_next(self, position):
        self.load_file()
        if position + 1 < len(self.lines):
            return self.lines[position + 1], position + 1
        return None, None

    def get_prev(self, position):
        self.load_file()
        if position > 0:
            return self.lines[position - 1], position - 1
        return None, None

    def insert_line(self, position, text=''):
        self.lines.insert(position, urwid.Edit('', text))
        self._modified()

    def remove_line(self, position):
        if len(self.lines) > 1:
            del self.lines[position]
            self._modified()

    def get_all_text(self):
        return [w.edit_text for w in self.lines]

    def set_line_text(self, position, text):
        self.lines[position].set_edit_text(text)
        self._modified()

class TextEditor:
    def __init__(self, filename):
        self.filename = filename
        self.walker = LazyFileListWalker(filename)
        self.listbox = urwid.ListBox(self.walker)
        self.footer = urwid.Text(
            u"Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move | ↑/↓: Navigate"
        )
        self.frame = urwid.Frame(self.listbox, footer=self.footer)
        self.loop = urwid.MainLoop(
            self.frame,
            unhandled_input=self.unhandled_input,
            handle_mouse=False
        )

    def unhandled_input(self, key):
        if key in ('ctrl s',):
            self.save_file()
        elif key in ('ctrl q',):
            raise urwid.ExitMainLoop()

    def keypress(self, size, key):
        focus_widget, focus_pos = self.walker.get_focus()
        if focus_widget is None:
            return

        if key == 'enter':
            text = focus_widget.edit_text
            cursor = focus_widget.edit_pos
            before = text[:cursor]
            after = text[cursor:]
            self.walker.set_line_text(focus_pos, before)
            self.walker.insert_line(focus_pos + 1, after)
            self.walker.set_focus(focus_pos + 1)
            self.walker.lines[focus_pos + 1].edit_pos = 0
        elif key == 'backspace':
            if focus_widget.edit_pos == 0:
                if focus_pos > 0:
                    prev_widget = self.walker.lines[focus_pos - 1]
                    prev_text = prev_widget.edit_text
                    prev_len = len(prev_text)
                    self.walker.set_line_text(focus_pos - 1, prev_text + focus_widget.edit_text)
                    self.walker.remove_line(focus_pos)
                    self.walker.set_focus(focus_pos - 1)
                    prev_widget.edit_pos = prev_len
            else:
                focus_widget.keypress(size, key)
        elif key == 'delete':
            text = focus_widget.edit_text
            cursor = focus_widget.edit_pos
            if cursor == len(text):
                if focus_pos + 1 < len(self.walker.lines):
                    next_widget = self.walker.lines[focus_pos + 1]
                    self.walker.set_line_text(focus_pos, text + next_widget.edit_text)
                    self.walker.remove_line(focus_pos + 1)
            else:
                focus_widget.keypress(size, key)
        elif key == 'left':
            if focus_widget.edit_pos == 0 and focus_pos > 0:
                self.walker.set_focus(focus_pos - 1)
                prev_widget = self.walker.lines[focus_pos - 1]
                prev_widget.edit_pos = len(prev_widget.edit_text)
            else:
                focus_widget.keypress(size, key)
        elif key == 'right':
            if focus_widget.edit_pos == len(focus_widget.edit_text) and focus_pos + 1 < len(self.walker.lines):
                self.walker.set_focus(focus_pos + 1)
                next_widget = self.walker.lines[focus_pos + 1]
                next_widget.edit_pos = 0
            else:
                focus_widget.keypress(size, key)
        elif key in ('up', 'down'):
            self.listbox.keypress(size, key)
        else:
            focus_widget.keypress(size, key)

    def save_file(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for line in self.walker.get_all_text():
                f.write(line + '\n')
        self.footer.set_text(u"Saved! Ctrl+S: Save | Ctrl+Q: Quit | Enter: New Line | Backspace/Delete: Remove | ←/→: Move | ↑/↓: Navigate")

    def run(self):
        self.loop.widget = urwid.AttrMap(self.frame, None)
        self.loop.screen.set_terminal_properties(colors=256)
        self.loop.input_filter = self.input_filter
        self.loop.run()

    def input_filter(self, keys, raw):
        for key in keys:
            self.keypress((80,), key)
        return []

def main():
    if len(sys.argv) != 2:
        print("Usage: python text_editor.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    editor = TextEditor(filename)
    editor.run()

if __name__ == '__main__':
    main()
```