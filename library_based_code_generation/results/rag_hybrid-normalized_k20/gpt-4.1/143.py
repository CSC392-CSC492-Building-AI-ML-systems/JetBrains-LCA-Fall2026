```python
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_bootstrap(
    ui.layout_columns(
        # First column: Controllers
        ui.column(
            4,
            ui.input_slider("num_inputs", "Number of Inputs (2-5)", min=2, max=5, value=3),
            ui.input_select("input_type", "Input Type", choices=["slider", "numeric"], selected="slider"),
        ),
        # Second column: Controlled Inputs
        ui.column(
            4,
            ui.output_ui("controlled_inputs"),
        ),
        # Third column: More Inputs and Tabset
        ui.column(
            4,
            ui.output_ui("third_col_inputs"),
            ui.tabs(
                ui.tab("Tab 1", ui.output_text_verbatim("tab1_output")),
                ui.tab("Tab 2", ui.output_text_verbatim("tab2_output")),
            ),
        ),
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    def controlled_inputs():
        n = input.num_inputs()
        input_type = input.input_type()
        controls = []
        for i in range(1, n+1):
            if input_type == "slider":
                controls.append(
                    ui.input_slider(f"ctrl_slider_{i}", f"Slider {i}", min=0, max=100, value=50)
                )
            else:
                controls.append(
                    ui.input_numeric(f"ctrl_numeric_{i}", f"Numeric {i}", value=0)
                )
        return controls

    @output
    @render.ui
    def third_col_inputs():
        n = input.num_inputs()
        input_type = input.input_type()
        controls = []
        for i in range(1, n+1):
            if input_type == "slider":
                controls.append(
                    ui.input_slider(f"third_slider_{i}", f"Third Col Slider {i}", min=0, max=100, value=25)
                )
            else:
                controls.append(
                    ui.input_numeric(f"third_numeric_{i}", f"Third Col Numeric {i}", value=10)
                )
        return controls

    @output
    @render.text
    def tab1_output():
        n = input.num_inputs()
        input_type = input.input_type()
        vals = []
        for i in range(1, n+1):
            if input_type == "slider":
                val = input.get(f"ctrl_slider_{i}", None)
            else:
                val = input.get(f"ctrl_numeric_{i}", None)
            vals.append(f"Input {i}: {val}")
        return "Controlled Inputs:\n" + "\n".join(vals)

    @output
    @render.text
    def tab2_output():
        n = input.num_inputs()
        input_type = input.input_type()
        vals = []
        for i in range(1, n+1):
            if input_type == "slider":
                val = input.get(f"third_slider_{i}", None)
            else:
                val = input.get(f"third_numeric_{i}", None)
            vals.append(f"Third Col Input {i}: {val}")
        return "Third Column Inputs:\n" + "\n".join(vals)

app = App(app_ui, server)

if __name__ == "__main__":
    from shiny import run_app
    run_app(app, debug=True)
```