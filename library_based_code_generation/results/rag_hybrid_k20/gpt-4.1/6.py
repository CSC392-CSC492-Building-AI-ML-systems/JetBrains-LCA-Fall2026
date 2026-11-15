```python
import urwid
import sys

# Custom characters for LCD (simulate with unicode for demo)
CHECKED = u"\u25A3"   # Custom checkbox (checked)
UNCHECKED = u"\u25A1" # Custom checkbox (unchecked)
RADIO_ON = u"\u25C9"  # Custom radio button (selected)
RADIO_OFF = u"\u25EF" # Custom radio button (unselected)
PROGRESS_CHARS = [u"\u2591", u"\u2592", u"\u2593", u"\u2588"]  # Progress bar levels
SLIDER_LEFT = u"\u25C0"
SLIDER_RIGHT = u"\u25B6"
SLIDER_KNOB = u"\u25CF"
SLIDER_BG = u"\u2500"
MENU_ARROW = u"\u25B6"

# Dummy LCD API
class DummyLCD:
    def __init__(self, port):
        self.port = port
        self.brightness = 50
        self.contrast = 50
        self.led_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        self.cursor_style = 0

    def set_brightness(self, value):
        self.brightness = value

    def set_contrast(self, value):
        self.contrast = value

    def set_led_color(self, idx, color):
        self.led_colors[idx] = color

    def set_cursor_style(self, style):
        self.cursor_style = style

# Connect to LCD
lcd_port = sys.argv[1] if len(sys.argv) > 1 else "COM1"
lcd = DummyLCD(lcd_port)

# Custom widgets
class CustomCheckBox(urwid.WidgetWrap):
    def __init__(self, label, state=False, on_state_change=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        self._w = urwid.Text(self._get_text())
        super().__init__(self._w)

    def _get_text(self):
        return ('default', (CHECKED if self.state else UNCHECKED) + " " + self.label)

    def keypress(self, size, key):
        if key in ('enter', ' '):
            self.state = not self.state
            self._w.set_text(self._get_text())
            if self.on_state_change:
                self.on_state_change(self.state)
        return key

class CustomRadioButton(urwid.WidgetWrap):
    group = []
    def __init__(self, label, state=False, group=None, on_state_change=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        if group is not None:
            self.group = group
            self.group.append(self)
        else:
            self.group = [self]
        self._w = urwid.Text(self._get_text())
        super().__init__(self._w)

    def _get_text(self):
        return ('default', (RADIO_ON if self.state else RADIO_OFF) + " " + self.label)

    def keypress(self, size, key):
        if key in ('enter', ' '):
            for btn in self.group:
                btn.state = False
                btn._w.set_text(btn._get_text())
            self.state = True
            self._w.set_text(self._get_text())
            if self.on_state_change:
                self.on_state_change(self.label)
        return key

class CustomProgressBar(urwid.WidgetWrap):
    def __init__(self, current, total, width=16):
        self.current = current
        self.total = total
        self.width = width
        self._w = urwid.Text(self._get_text())
        super().__init__(self._w)

    def set_progress(self, value):
        self.current = value
        self._w.set_text(self._get_text())

    def _get_text(self):
        frac = float(self.current) / self.total if self.total else 0
        full_blocks = int(frac * self.width)
        partial = int((frac * self.width - full_blocks) * (len(PROGRESS_CHARS)-1))
        bar = PROGRESS_CHARS[-1]*full_blocks
        if full_blocks < self.width:
            bar += PROGRESS_CHARS[partial]
            bar += PROGRESS_CHARS[0]*(self.width-full_blocks-1)
        return bar

class CustomSlider(urwid.WidgetWrap):
    def __init__(self, min_value, max_value, value, width=16, on_change=None):
        self.min = min_value
        self.max = max_value
        self.value = value
        self.width = width
        self.on_change = on_change
        self._w = urwid.Text(self._get_text())
        super().__init__(self._w)

    def _get_text(self):
        pos = int((self.value - self.min) / (self.max - self.min) * (self.width-2))
        slider = SLIDER_LEFT
        for i in range(self.width-2):
            if i == pos:
                slider += SLIDER_KNOB
            else:
                slider += SLIDER_BG
        slider += SLIDER_RIGHT
        return slider

    def keypress(self, size, key):
        if key in ('left', 'h'):
            if self.value > self.min:
                self.value -= 1
                self._w.set_text(self._get_text())
                if self.on_change:
                    self.on_change(self.value)
        elif key in ('right', 'l'):
            if self.value < self.max:
                self.value += 1
                self._w.set_text(self._get_text())
                if self.on_change:
                    self.on_change(self.value)
        return key

# Menu system
def menu_button(caption, callback):
    button = urwid.Button([caption, " ", MENU_ARROW])
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='reversed')

def back_button(callback):
    button = urwid.Button(u"Back")
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='reversed')

def display_settings_menu(parent):
    brightness_slider = CustomSlider(0, 100, lcd.brightness, width=16, on_change=lambda v: lcd.set_brightness(v))
    contrast_slider = CustomSlider(0, 100, lcd.contrast, width=16, on_change=lambda v: lcd.set_contrast(v))
    items = [
        urwid.Text("Display Settings"),
        urwid.Divider(),
        urwid.Text("Brightness:"),
        brightness_slider,
        urwid.Text("Contrast:"),
        contrast_slider,
        urwid.Divider(),
        back_button(lambda btn: open_menu(parent))
    ]
    return urwid.ListBox(urwid.SimpleFocusListWalker(items))

def cursor_settings_menu(parent):
    styles = ["None", "Underline", "Block", "Blink"]
    group = []
    def on_style_change(style):
        lcd.set_cursor_style(styles.index(style))
    items = [urwid.Text("Cursor Settings"), urwid.Divider()]
    for style in styles:
        btn = CustomRadioButton(style, state=(styles.index(style)==lcd.cursor_style), group=group, on_state_change=on_style_change)
        items.append(btn)
    items.append(urwid.Divider())
    items.append(back_button(lambda btn: open_menu(parent)))
    return urwid.ListBox(urwid.SimpleFocusListWalker(items))

def led_settings_menu(parent):
    led_labels = ["LED 1", "LED 2", "LED 3", "LED 4"]
    sliders = []
    def make_slider(idx, color_idx):
        def on_change(val):
            color = list(lcd.led_colors[idx])
            color[color_idx] = val
            lcd.set_led_color(idx, tuple(color))
        return CustomSlider(0, 255, lcd.led_colors[idx][color_idx], width=16, on_change=on_change)
    items = [urwid.Text("LED Settings"), urwid.Divider()]
    for idx, label in enumerate(led_labels):
        items.append(urwid.Text(label))
        items.append(urwid.Text("Red:"))
        items.append(make_slider(idx, 0))
        items.append(urwid.Text("Green:"))
        items.append(make_slider(idx, 1))
        items.append(urwid.Text("Blue:"))
        items.append(make_slider(idx, 2))
        items.append(urwid.Divider())
    items.append(back_button(lambda btn: open_menu(parent)))
    return urwid.ListBox(urwid.SimpleFocusListWalker(items))

def about_menu(parent):
    text = (
        "Crystalfontz 635 LCD Demo\n"
        "This demo shows a simulated user interface for the Crystalfontz 635 LCD display.\n"
        "Features:\n"
        "- Adjust display brightness and contrast\n"
        "- Change cursor style\n"
        "- Set LED colors\n"
        "- Custom widgets: check box, radio button, progress bar, slider\n"
        "- Menu navigation with back/cancel support\n"
        "LCD Port: %s" % lcd.port
    )
    items = [
        urwid.Text("About this Demo"),
        urwid.Divider(),
        urwid.Text(text),
        urwid.Divider(),
        back_button(lambda btn: open_menu(parent))
    ]
    return urwid.ListBox(urwid.SimpleFocusListWalker(items))

def main_menu():
    items = [
        urwid.Text("Crystalfontz 635 LCD Demo"),
        urwid.Divider