```python
import sys
import urwid
from crystalfontz635 import CF635Screen, set_lcd_contrast

# Custom characters for check box, radio button, progress bar, slider, and arrow
CHECKED_CHAR = u'\u25A3'   # ⎣ (as example for checked box)
UNCHECKED_CHAR = u'\u25A1' # □
RADIO_ON_CHAR = u'\u25C9'  # ◉
RADIO_OFF_CHAR = u'\u25CB' # ○
PROGRESS_CHARS = [u'\u2591', u'\u2592', u'\u2593', u'\u2588']  # ░ ▒ ▓ █
SLIDER_LEFT = u'\u25C0'    # ◀
SLIDER_RIGHT = u'\u25B6'   # ▶
SLIDER_KNOB = u'\u25CF'    # ●
MENU_ARROW = u'\u25B6'     # ▶

# LCD connection
if len(sys.argv) < 2:
    print("Usage: python demo.py <lcd_device>")
    sys.exit(1)
lcd_device = sys.argv[1]
lcd = CF635Screen(lcd_device)

# Helper widgets
class CustomCheckBox(urwid.WidgetWrap):
    def __init__(self, label, state=False, on_state_change=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='reversed')

    def _get_widget(self):
        char = CHECKED_CHAR if self.state else UNCHECKED_CHAR
        return urwid.Text([char, ' ', self.label])

    def toggle(self):
        self.state = not self.state
        if self.on_state_change:
            self.on_state_change(self.state)
        self._w.original_widget = self._get_widget()

    def selectable(self):
        return True

    def keypress(self, size, key):
        if key in ('enter', ' '):
            self.toggle()
            return None
        return key

class CustomRadioButton(urwid.WidgetWrap):
    group = []
    def __init__(self, label, state=False, on_state_change=None, group=None):
        self.state = state
        self.label = label
        self.on_state_change = on_state_change
        if group is not None:
            self.group = group
            self.group.append(self)
        else:
            self.group = [self]
        self._w = urwid.AttrMap(self._get_widget(), None, focus_map='reversed')

    def _get_widget(self):
        char = RADIO_ON_CHAR if self.state else RADIO_OFF_CHAR
        return urwid.Text([char, ' ', self.label])

    def set_state(self, state):
        if state:
            for btn in self.group:
                if btn is not self and btn.state:
                    btn.state = False
                    btn._w.original_widget = btn._get_widget()
        self.state = state
        if self.on_state_change:
            self.on_state_change(self.state)
        self._w.original_widget = self._get_widget()

    def selectable(self):
        return True

    def keypress(self, size, key):
        if key in ('enter', ' '):
            self.set_state(True)
            return None
        return key

class CustomProgressBar(urwid.WidgetWrap):
    def __init__(self, current, maximum, width=16):
        self.current = current
        self.maximum = maximum
        self.width = width
        self._w = urwid.Text(self._get_bar())

    def _get_bar(self):
        percent = float(self.current) / self.maximum if self.maximum else 0
        full_blocks = int(percent * self.width)
        partial_block = int((percent * self.width - full_blocks) * (len(PROGRESS_CHARS)-1))
        bar = PROGRESS_CHARS[-1] * full_blocks
        if full_blocks < self.width:
            bar += PROGRESS_CHARS[partial_block]
            bar += PROGRESS_CHARS[0] * (self.width - full_blocks - 1)
        return bar

    def set_progress(self, value):
        self.current = max(0, min(self.maximum, value))
        self._w.set_text(self._get_bar())

    def selectable(self):
        return False

class CustomSlider(urwid.WidgetWrap):
    def __init__(self, min_value, max_value, value, width=16, on_change=None):
        self.min = min_value
        self.max = max_value
        self.value = value
        self.width = width
        self.on_change = on_change
        self._w = urwid.Text(self._get_slider())

    def _get_slider(self):
        pos = int((self.value - self.min) / float(self.max - self.min) * (self.width-2))
        slider = SLIDER_LEFT
        for i in range(self.width-2):
            if i == pos:
                slider += SLIDER_KNOB
            else:
                slider += u'\u2500'  # ─
        slider += SLIDER_RIGHT
        return slider

    def set_value(self, value):
        self.value = max(self.min, min(self.max, value))
        self._w.set_text(self._get_slider())
        if self.on_change:
            self.on_change(self.value)

    def selectable(self):
        return True

    def keypress(self, size, key):
        if key in ('left', 'h'):
            self.set_value(self.value - 1)
            return None
        elif key in ('right', 'l'):
            self.set_value(self.value + 1)
            return None
        return key

# Menu system
class MenuButton(urwid.Button):
    def __init__(self, caption, callback):
        super().__init__("")
        urwid.connect_signal(self, 'click', callback)
        self._w = urwid.AttrMap(urwid.SelectableIcon([MENU_ARROW, ' ', caption], 0), None, focus_map='reversed')

def menu(title, choices):
    body = [urwid.Text(title), urwid.Divider()]
    for c, cb in choices:
        button = MenuButton(c, cb)
        body.append(button)
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def back_button(callback):
    return urwid.Button("Back", on_press=callback)

# Display Settings
class DisplaySettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.brightness = 50
        self.contrast = 50
        self.brightness_slider = CustomSlider(0, 100, self.brightness, width=16, on_change=self.on_brightness_change)
        self.contrast_slider = CustomSlider(0, 100, self.contrast, width=16, on_change=self.on_contrast_change)
        items = [
            urwid.Text("Display Settings"),
            urwid.Divider(),
            urwid.Text("Brightness:"),
            self.brightness_slider,
            urwid.Text("Contrast:"),
            self.contrast_slider,
            urwid.Divider(),
            back_button(lambda btn: on_back())
        ]
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_brightness_change(self, value):
        self.brightness = value
        # lcd.set_brightness(value)  # If API available

    def on_contrast_change(self, value):
        self.contrast = value
        set_lcd_contrast(lcd, value)

# Cursor Settings
class CursorSettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.styles = ["None", "Underline", "Block", "Blink"]
        self.group = []
        items = [urwid.Text("Cursor Settings"), urwid.Divider()]
        for style in self.styles:
            btn = CustomRadioButton(style, state=(style=="None"), group=self.group, on_state_change=self.on_style_change)
            items.append(btn)
        items.append(urwid.Divider())
        items.append(back_button(lambda btn: on_back()))
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_style_change(self, state):
        # lcd.set_cursor_style(...)  # If API available
        pass

# LED Settings
class LEDSettings(urwid.WidgetWrap):
    def __init__(self, on_back):
        self.leds = [ [255,0,0], [0,255,0], [0,0,255], [255,255,0] ]
        self.sliders = []
        items = [urwid.Text("LED Settings"), urwid.Divider()]
        for i, color in enumerate(self.leds):
            items.append(urwid.Text(f"LED {i+1}:"))
            for j, cname in enumerate(['R','G','B']):
                slider = CustomSlider(0,255,color[j],width=16,on_change=lambda v,i=i,j=j:self.on_led_change(i,j,v))
                self.sliders.append(slider)
                items.append(urwid.Text(f"  {cname}:"))
                items.append(slider)
        items.append(urwid.Divider())
        items.append(back_button(lambda btn: on_back()))
        super().__init__(urwid.ListBox(urwid.SimpleFocusListWalker(items)))

    def on_led_change(self, led_idx, color_idx, value):
        self.leds[led_idx][color_idx] = value
        # lcd.set_led(led_idx, *self.leds[led_idx])  # If API available

