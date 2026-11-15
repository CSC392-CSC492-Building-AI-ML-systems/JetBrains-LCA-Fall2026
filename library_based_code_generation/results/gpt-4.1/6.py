```python
import urwid
import sys

# Custom characters for LCD (simulate with Unicode for demo)
CHECKED_CHAR = u'\u25A3'   # ⎣ (simulate checked box)
UNCHECKED_CHAR = u'\u25A1' # □ (simulate unchecked box)
RADIO_ON_CHAR = u'\u25C9'  # ◉ (simulate radio on)
RADIO_OFF_CHAR = u'\u25CB' # ○ (simulate radio off)
PROGRESS_CHARS = [u'\u2591', u'\u2592', u'\u2593', u'\u2588']  # ░ ▒ ▓ █
SLIDER_LEFT = u'\u25C0'    # ◀
SLIDER_RIGHT = u'\u25B6'   # ▶
SLIDER_KNOB = u'\u25CF'    # ●
MENU_ARROW = u'\u25B6'     # ▶

# Simulate LCD connection (replace with actual connection code)
def connect_lcd(port):
    # Placeholder for actual LCD connection
    return True

if len(sys.argv) < 2:
    print("Usage: python demo.py <lcd_port>")
    sys.exit(1)

lcd_port = sys.argv[1]
lcd_connected = connect_lcd(lcd_port)

# --- Custom Widgets ---

class CustomCheckBox(urwid.WidgetWrap):
    def __init__(self, label, state=False, on_state_change=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='reversed')
        super().__init__(self._w)

    def _get_widget(self):
        char = CHECKED_CHAR if self.state else UNCHECKED_CHAR
        return urwid.Text([char + ' ' + self.label])

    def keypress(self, size, key):
        if key in ('enter', ' '):
            self.state = not self.state
            self._w.original_widget = self._get_widget()
            if self.on_state_change:
                self.on_state_change(self.state)
        return key

class CustomRadioButton(urwid.WidgetWrap):
    group = []
    def __init__(self, label, state=False, on_state_change=None, group=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        if group is None:
            self.group = []
        else:
            self.group = group
        self.group.append(self)
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='reversed')
        super().__init__(self._w)

    def _get_widget(self):
        char = RADIO_ON_CHAR if self.state else RADIO_OFF_CHAR
        return urwid.Text([char + ' ' + self.label])

    def keypress(self, size, key):
        if key in ('enter', ' '):
            for btn in self.group:
                btn.state = False
                btn._w.original_widget = btn._get_widget()
            self.state = True
            self._w.original_widget = self._get_widget()
            if self.on_state_change:
                self.on_state_change(self.state)
        return key

class CustomProgressBar(urwid.WidgetWrap):
    def __init__(self, current, total, width=16):
        self.current = current
        self.total = total
        self.width = width
        self._w = urwid.Text(self._get_bar())
        super().__init__(self._w)

    def _get_bar(self):
        percent = float(self.current) / self.total
        full_blocks = int(percent * self.width)
        partial_block = int((percent * self.width - full_blocks) * (len(PROGRESS_CHARS)-1))
        bar = PROGRESS_CHARS[-1] * full_blocks
        if full_blocks < self.width:
            bar += PROGRESS_CHARS[partial_block]
            bar += PROGRESS_CHARS[0] * (self.width - full_blocks - 1)
        return bar

    def set_progress(self, value):
        self.current = max(0, min(self.total, value))
        self._w.set_text(self._get_bar())

class CustomSlider(urwid.WidgetWrap):
    def __init__(self, min_value, max_value, value, width=16, on_change=None):
        self.min = min_value
        self.max = max_value
        self.value = value
        self.width = width
        self.on_change = on_change
        self._w = urwid.Text(self._get_slider())
        super().__init__(self._w)

    def _get_slider(self):
        pos = int((self.value - self.min) / (self.max - self.min) * (self.width-2))
        slider = SLIDER_LEFT
        for i in range(self.width-2):
            if i == pos:
                slider += SLIDER_KNOB
            else:
                slider += u'\u2500'  # ─
        slider += SLIDER_RIGHT
        return slider

    def keypress(self, size, key):
        if key in ('left', 'h'):
            if self.value > self.min:
                self.value -= 1
                self._w.set_text(self._get_slider())
                if self.on_change:
                    self.on_change(self.value)
        elif key in ('right', 'l'):
            if self.value < self.max:
                self.value += 1
                self._w.set_text(self._get_slider())
                if self.on_change:
                    self.on_change(self.value)
        return key

# --- Menu System ---

class MenuButton(urwid.Button):
    def __init__(self, caption, callback):
        super().__init__("")
        urwid.connect_signal(self, 'click', callback)
        self._w = urwid.AttrMap(urwid.SelectableIcon([caption, ' ', MENU_ARROW], 0), None, focus_map='reversed')

def menu(title, choices):
    body = [urwid.Text(title), urwid.Divider()]
    for c in choices:
        body.append(MenuButton(c[0], c[1]))
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def back_button(callback):
    return urwid.Button("Back", on_press=callback)

# --- Display Settings ---

class DisplaySettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.brightness = 8
        self.contrast = 8
        self.brightness_slider = CustomSlider(0, 15, self.brightness, width=16, on_change=self.on_brightness_change)
        self.contrast_slider = CustomSlider(0, 15, self.contrast, width=16, on_change=self.on_contrast_change)
        items = [
            urwid.Text("Display Settings"),
            urwid.Divider(),
            urwid.Text("Brightness:"),
            self.brightness_slider,
            urwid.Divider(),
            urwid.Text("Contrast:"),
            self.contrast_slider,
            urwid.Divider(),
            back_button(on_back)
        ]
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_brightness_change(self, value):
        self.brightness = value
        # Send brightness to LCD here

    def on_contrast_change(self, value):
        self.contrast = value
        # Send contrast to LCD here

# --- Cursor Settings ---

class CursorSettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.styles = ["None", "Underline", "Block", "Blink"]
        self.radio_group = []
        items = [urwid.Text("Cursor Settings"), urwid.Divider()]
        for style in self.styles:
            btn = CustomRadioButton(style, state=(style=="None"), group=self.radio_group, on_state_change=self.on_style_change)
            items.append(btn)
        items.append(urwid.Divider())
        items.append(back_button(on_back))
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_style_change(self, state):
        # Send cursor style to LCD here
        pass

# --- LED Settings ---

class LEDSettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.leds = [ [255,0,0], [0,255,0], [0,0,255], [255,255,0] ]  # 4 LEDs, RGB
        self.sliders = []
        items = [urwid.Text("LED Settings"), urwid.Divider()]
        for i, led in enumerate(self.leds):
            items.append(urwid.Text(f"LED {i+1}:"))
            for j, color in enumerate(['R','G','B']):
                slider = CustomSlider(0,255,led[j],width=16, on_change=lambda val, i=i, j=j: self.on_led_change(i,j,val))
                self.sliders.append(slider)
                items.append(urwid.Text(f"  {color}:"))
                items.append(slider)
            items.append(urwid.Divider())
        items.append(back_button(on_back))
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_led_change(self, led_idx, color_idx, value):
        self.leds[led_idx][color_idx] = value
        # Send LED color to LCD here

# --- About Section ---

class AboutDemo(urwid.WidgetWrap):
    def __init__(self, on_back):
        text = (
            "Crystalfontz 635 LCD Demo\n"
            "This