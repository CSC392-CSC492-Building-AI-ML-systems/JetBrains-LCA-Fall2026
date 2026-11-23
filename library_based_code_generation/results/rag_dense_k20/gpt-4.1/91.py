```python
import sys
import time
import threading

from shiny import App, ui, render, reactive, Inputs, Outputs, Session
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

def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

app_ui = ui.page_fluid(
    ui.h2("CPU Usage Monitor"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("colormap", "Colormap", COLORMAPS, selected="viridis"),
            ui.input_numeric("num_samples", "Samples per graph", 30, min=5, max=200),
            ui.input_numeric("num_rows", "Rows in table", 10, min=1, max=200),
            ui.input_checkbox("freeze", "Freeze output", False),
            ui.input_action_button("clear", "Clear history"),
        ),
        ui.panel_main(
            ui.output_plot("cpu_plot", height="350px"),
            ui.output_data_frame("cpu_table")
        )
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    # Store CPU usage history
    cpu_history = reactive.Value([])

    # Background thread for polling CPU usage
    stop_event = threading.Event()

    def poll_cpu():
        while not stop_event.is_set():
            if not input.freeze():
                history = cpu_history()
                if len(history) >= input.num_samples():
                    history = history[-(input.num_samples()-1):]
                usage = psutil.cpu_percent(interval=None)
                history.append({"Time": pd.Timestamp.now(), "CPU %": usage})
                cpu_history.set(history)
            time.sleep(1)

    if not is_pyodide():
        thread = threading.Thread(target=poll_cpu, daemon=True)
        thread.start()

    @reactive.Effect
    def _pyodide_poll():
        if is_pyodide():
            if not input.freeze():
                history = cpu_history()
                if len(history) >= input.num_samples():
                    history = history[-(input.num_samples()-1):]
                usage = psutil.cpu_percent(interval=None)
                history.append({"Time": pd.Timestamp.now(), "CPU %": usage})
                cpu_history.set(history)
            session.invalidate_later(1)

    @reactive.Effect
    @reactive.event(input.clear)
    def _clear_history():
        cpu_history.set([])

    @output
    @render.plot
    def cpu_plot():
        history = cpu_history()
        if not history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
            hide_ticks(ax)
            return fig
        df = pd.DataFrame(history)
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df["CPU %"], color="black", marker="o", linestyle="-", label="CPU %")
        sc = ax.scatter(df["Time"], df["CPU %"], c=df["CPU %"], cmap=input.colormap(), s=60)
        ax.set_ylabel("CPU Usage (%)")
        ax.set_xlabel("Time")
        ax.set_ylim(0, 100)
        fig.autofmt_xdate()
        plt.colorbar(sc, ax=ax, label="CPU %")
        hide_ticks(ax)
        return fig

    @output
    @render.data_frame
    def cpu_table():
        history = cpu_history()
        if not history:
            return pd.DataFrame(columns=["Time", "CPU %"])
        df = pd.DataFrame(history)
        return df.tail(input.num_rows())

app = App(app_ui, server)
```