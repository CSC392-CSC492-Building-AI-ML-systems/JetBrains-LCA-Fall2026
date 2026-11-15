```python
import sys
import asyncio
import time

from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# Detect Pyodide environment
IN_PYODIDE = sys.platform == "emscripten"

# psutil or fake psutil
if IN_PYODIDE:
    import random

    class FakePsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return random.uniform(0, 100)

    psutil = FakePsutil()
else:
    import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

# Helper to hide axis ticks
def hide_axis_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# UI
app_ui = ui.page_fluid(
    ui.h2("CPU Usage Monitor"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select(
                "colormap",
                "Colormap",
                {
                    "viridis": "Viridis",
                    "plasma": "Plasma",
                    "inferno": "Inferno",
                    "magma": "Magma",
                    "cividis": "Cividis",
                    "jet": "Jet",
                    "gray": "Gray",
                },
                selected="viridis"
            ),
            ui.input_numeric("samples", "Samples per graph", 30, min=5, max=200),
            ui.input_numeric("rows", "Rows in table", 10, min=1, max=200),
            ui.input_action_button("clear", "Clear History"),
            ui.input_checkbox("freeze", "Freeze Output", False),
        ),
        ui.panel_main(
            ui.output_plot("cpu_plot"),
            ui.output_table("cpu_table"),
        ),
    ),
)

# Server
def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values for CPU usage history
    cpu_history = reactive.Value([])

    # Timer for polling CPU usage
    @reactive.effect
    @reactive.periodic(1)
    def _poll_cpu():
        if input.freeze():
            return
        history = cpu_history()
        usage = psutil.cpu_percent(interval=None)
        timestamp = time.strftime("%H:%M:%S")
        history.append({"Time": timestamp, "CPU (%)": usage})
        # Keep only the last N samples
        max_samples = input.samples()
        if len(history) > max_samples:
            history = history[-max_samples:]
        cpu_history.set(history)

    # Clear history
    @reactive.effect
    @reactive.event(input.clear)
    def _clear_history():
        cpu_history.set([])

    # DataFrame for table and plot
    @reactive.calc
    def cpu_df():
        df = pd.DataFrame(cpu_history())
        if not df.empty:
            df = df.tail(input.rows())
        return df

    # Plot output
    @output
    @render.plot
    def cpu_plot():
        df = cpu_df()
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            hide_axis_ticks(ax)
            return fig
        fig, ax = plt.subplots(figsize=(6, 3))
        cmap = plt.get_cmap(input.colormap())
        normed = [v/100 for v in df["CPU (%)"]]
        colors = cmap(normed)
        ax.plot(df["Time"], df["CPU (%)"], color="black", marker="o")
        ax.fill_between(df["Time"], 0, df["CPU (%)"], color=input.colormap(), alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_ylabel("CPU (%)")
        ax.set_xlabel("Time")
        ax.set_title("CPU Usage Over Time")
        ax.grid(True, linestyle="--", alpha=0.5)
        # Optionally hide ticks if too many samples
        if len(df) > 20:
            hide_axis_ticks(ax)
        plt.tight_layout()
        return fig

    # Table output
    @output
    @render.table
    def cpu_table():
        df = cpu_df()
        if df.empty:
            return pd.DataFrame({"Time": [], "CPU (%)": []})
        return df.reset_index(drop=True)

app = App(app_ui, server)
```