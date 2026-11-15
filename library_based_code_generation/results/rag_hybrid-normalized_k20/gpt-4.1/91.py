```python
import sys
import time
import threading

from shiny import App, ui, render, reactive, req
import matplotlib.pyplot as plt
import pandas as pd

# Detect Pyodide environment
def is_pyodide():
    try:
        import js  # pyodide-specific
        return True
    except ImportError:
        return False

# Fake psutil for Pyodide
class FakePsutil:
    import random
    @staticmethod
    def cpu_percent(interval=None):
        return FakePsutil.random.uniform(0, 100)

if is_pyodide():
    psutil = FakePsutil
else:
    import psutil

# Available colormaps
COLORMAPS = sorted([
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Greens', 'Blues', 'Reds', 'Purples', 'Oranges',
    'gray', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'
])

# Hide axis ticks utility
def hide_axis_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

app_ui = ui.page_fluid(
    ui.panel_title("CPU Usage Monitor"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("colormap", "Colormap", COLORMAPS, selected="viridis"),
            ui.input_numeric("samples", "Samples per graph", 30, min=5, max=200),
            ui.input_numeric("rows", "Rows in table", 10, min=1, max=200),
            ui.input_action_button("clear", "Clear History"),
            ui.input_checkbox("freeze", "Freeze Output", False),
        ),
        ui.panel_main(
            ui.output_plot("cpu_plot", height="350px"),
            ui.output_data_frame("cpu_table")
        )
    )
)

def server(input, output, session):
    # Reactive values for CPU usage history
    cpu_history = reactive.Value([])

    # Background thread for polling CPU usage (only in standard Python)
    def poll_cpu():
        while True:
            if not input.freeze():
                hist = cpu_history.get()
                hist = hist[-input.samples()+1:] if len(hist) >= input.samples() else hist
                hist.append({
                    "Timestamp": pd.Timestamp.now(),
                    "CPU (%)": psutil.cpu_percent(interval=None)
                })
                cpu_history.set(hist)
            time.sleep(1)

    if not is_pyodide():
        threading.Thread(target=poll_cpu, daemon=True).start()

    # In Pyodide, use a reactive timer
    @reactive.effect
    def _pyodide_poll():
        if is_pyodide():
            if not input.freeze():
                hist = cpu_history.get()
                hist = hist[-input.samples()+1:] if len(hist) >= input.samples() else hist
                hist.append({
                    "Timestamp": pd.Timestamp.now(),
                    "CPU (%)": psutil.cpu_percent(interval=None)
                })
                cpu_history.set(hist)
            reactive.invalidate_later(1)

    @reactive.Effect
    @reactive.event(input.clear)
    def _clear_history():
        cpu_history.set([])

    @output
    @render.plot
    def cpu_plot():
        hist = cpu_history.get()
        if not hist:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
            hide_axis_ticks(ax)
            return fig
        df = pd.DataFrame(hist)[-input.samples():]
        fig, ax = plt.subplots()
        cmap = plt.get_cmap(input.colormap())
        ax.plot(df["Timestamp"], df["CPU (%)"], color=cmap(0.6), marker='o')
        ax.set_title("CPU Usage (%)")
        ax.set_xlabel("Time")
        ax.set_ylabel("CPU (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        hide_axis_ticks(ax)
        return fig

    @output
    @render.data_frame
    def cpu_table():
        hist = cpu_history.get()
        if not hist:
            return pd.DataFrame(columns=["Timestamp", "CPU (%)"])
        df = pd.DataFrame(hist)[-input.rows():]
        return df.reset_index(drop=True)

app = App(app_ui, server)
```