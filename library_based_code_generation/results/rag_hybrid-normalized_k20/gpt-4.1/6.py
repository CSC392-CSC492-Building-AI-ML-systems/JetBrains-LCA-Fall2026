```python
import urwid
import sys

# Custom characters for LCD (simulate with unicode)
CHECKBOX_UNCHECKED = u'\u25A1'  # □
CHECKBOX_CHECKED = u'\u25A0'    # ■
RADIO_UNSELECTED = u'\u25CB'    # ○
RADIO_SELECTED = u'\u25C9'      # ◉
PROGRESS_CHARS = [u'\u2591', u'\u2592', u'\u2593', u'\u2588']  # ░ ▒ ▓ █
SLIDER_LEFT = u'\u25C0'         # ◀
SLIDER_RIGHT = u'\u25B6'        # ▶
SLIDER_KNOB = u'\u25CF'         # ●
MENU_ARROW = u'\u25B6'          # ▶

# Dummy LCD API
class DummyLCD:
    def __init__(self, port):
        self.port = port
        self.brightness = 50
        self.contrast = 50
        self.led_colors = [(0,0,0)]*4
        self.cursor_style = 0

    def set_brightness(self, value):
        self.brightness = value

    def set_contrast(self, value):
        self.contrast = value

    def set_led(self, idx, color):
        self.led_colors[idx] = color

    def set_cursor_style(self, style):
        self.cursor_style = style

# Connect to LCD
lcd_port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
lcd = DummyLCD(lcd_port)

palette = [
    ('reversed', 'standout', ''),
    ('slider', 'light cyan', ''),
    ('slider_knob', 'dark red', ''),
    ('menu_arrow', 'yellow', ''),
    ('led_red', 'dark red', ''),
    ('led_green', 'dark green', ''),
    ('led_blue', 'dark blue', ''),
    ('led_yellow', 'yellow', ''),
    ('focus', 'black', 'light gray'),
]

def menu_button(caption, callback, user_data=None):
    arrow = urwid.Text(('menu_arrow', MENU_ARROW), align='right')
    button = urwid.Button(caption)
    urwid.connect_signal(button, 'click', callback, user_data)
    w = urwid.AttrMap(urwid.Columns([button, (3, arrow)]), None, focus_map='focus')
    return w

def back_button(callback):
    button = urwid.Button("Back")
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='focus')

class CustomCheckBox(urwid.WidgetWrap):
    def __init__(self, label, state=False, on_state_change=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='focus')

    def _get_widget(self):
        char = CHECKBOX_CHECKED if self.state else CHECKBOX_UNCHECKED
        return urwid.Columns([(2, urwid.Text(char)), urwid.Text(self.label)])

    def keypress(self, size, key):
        if key in ('enter', ' '):
            self.state = not self.state
            if self.on_state_change:
                self.on_state_change(self.state)
            self._w.original_widget = self._get_widget()
        return key

class CustomRadioButton(urwid.WidgetWrap):
    group = []
    def __init__(self, label, state=False, on_state_change=None, group=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        self.group = group if group is not None else CustomRadioButton.group
        self.group.append(self)
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='focus')

    def _get_widget(self):
        char = RADIO_SELECTED if self.state else RADIO_UNSELECTED
        return urwid.Columns([(2, urwid.Text(char)), urwid.Text(self.label)])

    def keypress(self, size, key):
        if key in ('enter', ' '):
            for btn in self.group:
                btn.state = False
                btn._w.original_widget = btn._get_widget()
            self.state = True
            if self.on_state_change:
                self.on_state_change(self)
            self._w.original_widget = self._get_widget()
        return key

class CustomProgressBar(urwid.WidgetWrap):
    def __init__(self, current, total, width=16):
        self.current = current
        self.total = total
        self.width = width
        self._w = self._get_widget()

    def _get_widget(self):
        frac = float(self.current) / self.total if self.total else 0
        n_full = int(frac * self.width)
        n_partial = int((frac * self.width - n_full) * (len(PROGRESS_CHARS)-1))
        bar = PROGRESS_CHARS[-1]*n_full
        if n_full < self.width:
            bar += PROGRESS_CHARS[n_partial]
            bar += PROGRESS_CHARS[0]*(self.width-n_full-1)
        return urwid.Text(bar)

    def set_progress(self, value):
        self.current = value
        self._w = self._get_widget()

    def render(self, size, focus=False):
        return self._w.render(size, focus)

class CustomSlider(urwid.WidgetWrap):
    def __init__(self, min_value, max_value, value, width=16, on_change=None):
        self.min = min_value
        self.max = max_value
        self.value = value
        self.width = width
        self.on_change = on_change
        self._w = self._get_widget()

    def _get_widget(self):
        pos = int((self.value - self.min) / float(self.max - self.min) * (self.width-2))
        bar = SLIDER_LEFT
        for i in range(self.width-2):
            if i == pos:
                bar += ('slider_knob', SLIDER_KNOB)
            else:
                bar += ('slider', u'\u2500')  # ─
        bar += SLIDER_RIGHT
        return urwid.AttrMap(urwid.Text(bar), None)

    def keypress(self, size, key):
        if key in ('left', 'h'):
            if self.value > self.min:
                self.value -= 1
                if self.on_change:
                    self.on_change(self.value)
                self._w = self._get_widget()
        elif key in ('right', 'l'):
            if self.value < self.max:
                self.value += 1
                if self.on_change:
                    self.on_change(self.value)
                self._w = self._get_widget()
        return key

    def render(self, size, focus=False):
        return self._w.render(size, focus)

# Menu callbacks
def main_menu():
    body = [
        urwid.Text("Crystalfontz 635 LCD Demo", align='center'),
        urwid.Divider(),
        menu_button("Display Settings", open_display_settings),
        menu_button("Cursor Settings", open_cursor_settings),
        menu_button("LED Settings", open_led_settings),
        menu_button("About this Demo", open_about),
    ]
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def open_display_settings(button=None):
    brightness_slider = CustomSlider(0, 100, lcd.brightness, width=20,
        on_change=lambda v: lcd.set_brightness(v))
    contrast_slider = CustomSlider(0, 100, lcd.contrast, width=20,
        on_change=lambda v: lcd.set_contrast(v))
    items = [
        urwid.Text("Display Settings", align='center'),
        urwid.Divider(),
        urwid.Text("Brightness:"),
        brightness_slider,
        urwid.Divider(),
        urwid.Text("Contrast:"),
        contrast_slider,
        urwid.Divider(),
        back_button(go_back),
    ]
    push_menu(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

def open_cursor_settings(button=None):
    styles = ["None", "Underline", "Block", "Blink"]
    group = []
    items = [urwid.Text("Cursor Settings", align='center'), urwid.Divider()]
    for i, style in enumerate(styles):
        btn = CustomRadioButton(style, state=(i==lcd.cursor_style),
            on_state_change=lambda btn, idx=i: lcd.set_cursor_style(idx), group=group)
        items.append(btn)
    items.append(urwid.Divider())
    items.append(back_button(go_back))
    push_menu(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

def open_led_settings(button=None):
    led_labels = ["LED 1", "LED 2", "LED 3", "LED 4"]
    led_colors = ["Red", "Green", "Blue", "Yellow"]
    led_palette = ['led_red', 'led_green', 'led_blue', 'led_yellow']
    items = [urwid.Text("LED Settings", align='center'), urwid.Divider()]
    for i, label in enumerate(led_labels):
        def make_led_slider(idx):
            def on_change(val):
                color = [0,0,0]
                color[idx%3] = val
                lcd.set_led(idx, tuple(color))
            return CustomSlider(0, 255, lcd.led_colors[i][i%3], width=20, on_change=on_change