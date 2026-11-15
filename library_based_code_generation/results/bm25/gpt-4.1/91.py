```python
import sys
import time
import threading

from shiny import App, ui, render, reactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Detect Pyodide environment
def is_pyodide():
    try:
        import js  # pyodide-specific
        return True
    except ImportError:
        return False

# Fake psutil for Pyodide
class FakePsutil:
    def cpu_percent(self, interval=None):
        # Return a random float between 0 and 100
        return np.random.uniform(0, 100)

if is_pyodide():
    psutil = FakePsutil()
else:
    import psutil

# Available colormaps
COLORMAPS = sorted([
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greens", "Blues", "Reds", "Purples", "Oranges",
    "gray", "hot", "cool", "spring", "summer", "autumn", "winter"
])

# Hide axis ticks utility
def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# UI
app_ui = ui.page_fluid(
    ui.h2("CPU Usage Monitor"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("colormap", "Colormap", COLORMAPS, selected="viridis"),
            ui.input_numeric("num_samples", "Samples per graph", 30, min=5, max=200),
            ui.input_numeric("num_rows", "Rows in table", 10, min=1, max=200),
            ui.input_checkbox("freeze", "Freeze output", False),
            ui.input_action_button("clear", "Clear History"),
            ui.hr(),
            ui.p("Monitor your CPU usage in real time. Works in both standard Python and Pyodide environments."),
        ),
        ui.panel_main(
            ui.output_plot("cpu_plot"),
            ui.output_data_frame("cpu_table"),
        )
    )
)

# Server logic
def server(input, output, session):
    # Reactive values for CPU usage history
    cpu_history = reactive.Value([])

    # Background thread for polling CPU usage (only in standard Python)
    def poll_cpu():
        while True:
            if not input.freeze():
                history = cpu_history()
                history.append({
                    "timestamp": pd.Timestamp.now(),
                    "cpu_percent": psutil.cpu_percent(interval=None)
                })
                # Keep only the last N samples
                max_samples = input.num_samples()
                if len(history) > max_samples:
                    history = history[-max_samples:]
                cpu_history.set(history)
            time.sleep(1)

    if not is_pyodide():
        threading.Thread(target=poll_cpu, daemon=True).start()

    # In Pyodide, use a reactive timer
    @reactive.effect
    @reactive.event(lambda: None if not is_pyodide() else reactive.timer(1000))
    def _pyodide_poll():
        if is_pyodide() and not input.freeze():
            history = cpu_history()
            history.append({
                "timestamp": pd.Timestamp.now(),
                "cpu_percent": psutil.cpu_percent(interval=None)
            })
            max_samples = input.num_samples()
            if len(history) > max_samples:
                history = history[-max_samples:]
            cpu_history.set(history)

    # Clear history on button click
    @reactive.effect
    @reactive.event(input.clear)
    def _clear_history():
        cpu_history.set([])

    # CPU usage DataFrame
    @reactive.calc
    def cpu_df():
        data = cpu_history()
        if not data:
            return pd.DataFrame(columns=["timestamp", "cpu_percent"])
        df = pd.DataFrame(data)
        return df

    # Plot output
    @output
    @render.plot
    def cpu_plot():
        df = cpu_df()
        if df.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
            hide_ticks(ax)
            return fig
        fig, ax = plt.subplots(figsize=(6, 3))
        cmap = plt.get_cmap(input.colormap())
        normed = np.array(df["cpu_percent"]) / 100
        colors = cmap(normed)
        ax.plot(df["timestamp"], df["cpu_percent"], color="black", marker="o")
        ax.scatter(df["timestamp"], df["cpu_percent"], color=colors, s=60)
        ax.set_ylim(0, 100)
        ax.set_ylabel("CPU Usage (%)")
        ax.set_xlabel("Time")
        ax.set_title("CPU Usage Over Time")
        hide_ticks(ax)
        fig.autofmt_xdate()
        return fig

    # Table output
    @output
    @render.data_frame
    def cpu_table():
        df = cpu_df()
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "cpu_percent"])
        nrows = input.num_rows()
        return df.tail(nrows).reset_index(drop=True)

app = App(app_ui, server)
```