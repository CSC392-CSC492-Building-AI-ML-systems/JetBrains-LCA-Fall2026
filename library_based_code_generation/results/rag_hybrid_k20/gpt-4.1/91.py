```python
import sys
import time
import threading

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Detect Pyodide environment
def is_pyodide():
    try:
        import js  # pyodide provides 'js' module
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

# Colormap options
COLORMAPS = sorted([
    m for m in plt.colormaps()
    if not m.endswith("_r") and not m.startswith("tab")
])

# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("colormap", "Colormap", COLORMAPS, selected="viridis"),
        ui.input_numeric("num_samples", "Samples per graph", 30, min=5, max=200),
        ui.input_numeric("num_rows", "Rows in table", 10, min=1, max=200),
        ui.input_action_button("clear", "Clear History"),
        ui.input_checkbox("freeze", "Freeze Output", False),
        width=300
    ),
    ui.layout_columns(
        ui.card(
            ui.output_plot("cpu_plot", height="350px"),
            title="CPU Usage Graph"
        ),
        ui.card(
            ui.output_data_frame("cpu_table"),
            title="CPU Usage Table"
        )
    )
)

# Hide axis ticks utility
def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# Server logic
def server(input: Inputs, output: Outputs, session: Session):
    # Store CPU usage history
    cpu_history = reactive.Value([])

    # Background thread for polling CPU usage
    def poll_cpu():
        while True:
            if not input.freeze():
                history = cpu_history()
                history = history[-input.num_samples()+1:] if input.num_samples() > 1 else []
                usage = psutil.cpu_percent(interval=None)
                timestamp = pd.Timestamp.now()
                history.append({"Time": timestamp, "CPU %": usage})
                cpu_history.set(history)
            time.sleep(1)

    # Start polling in a thread (not in Pyodide)
    if not is_pyodide():
        threading.Thread(target=poll_cpu, daemon=True).start()

    # In Pyodide, use reactive polling
    @reactive.effect
    @reactive.periodic(1000)
    def _():
        if is_pyodide() and not input.freeze():
            history = cpu_history()
            history = history[-input.num_samples()+1:] if input.num_samples() > 1 else []
            usage = psutil.cpu_percent(interval=None)
            timestamp = pd.Timestamp.now()
            history.append({"Time": timestamp, "CPU %": usage})
            cpu_history.set(history)

    # Clear history
    @reactive.effect
    def _():
        if input.clear() > 0:
            cpu_history.set([])

    # DataFrame for table and plot
    @reactive.calc
    def cpu_df():
        df = pd.DataFrame(cpu_history())
        if not df.empty:
            df = df.tail(input.num_samples())
        return df

    # Plot output
    @output
    @render.plot
    def cpu_plot():
        df = cpu_df()
        fig, ax = plt.subplots(figsize=(6, 3))
        if not df.empty:
            cmap = plt.get_cmap(input.colormap())
            normed = [v/100 for v in df["CPU %"]]
            ax.plot(df["Time"], df["CPU %"], color=cmap(0.6), marker="o")
            ax.set_ylim(0, 100)
            ax.set_ylabel("CPU Usage (%)")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            # Optionally hide ticks
            # hide_ticks(ax)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    # Table output
    @output
    @render.data_frame
    def cpu_table():
        df = cpu_df()
        if not df.empty:
            df = df.tail(input.num_rows())
            df = df.copy()
            df["Time"] = df["Time"].dt.strftime("%H:%M:%S")
        return render.DataTable(df, width="100%")

app = App(app_ui, server)
```