```python
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_fluid(
    ui.layout_columns(
        # First column: Controllers
        ui.panel_well(
            ui.input_slider("num_inputs", "Number of Inputs (2-5)", min=2, max=5, value=3),
            ui.input_select("input_type", "Input Type", {"slider": "Slider", "numeric": "Numeric Input"}, selected="slider"),
            width=4
        ),
        # Second column: Controlled Inputs
        ui.panel_well(
            ui.output_ui("controlled_inputs"),
            width=4
        ),
        # Third column: More Inputs and Tabset
        ui.panel_well(
            ui.output_ui("extra_inputs"),
            ui.output_ui("tabset"),
            width=4
        )
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
                    ui.input_numeric(f"ctrl_numeric_{i}", f"Numeric {i}", value=0, min=0, max=100)
                )
        return controls

    @output
    @render.ui
    def extra_inputs():
        n = input.num_inputs()
        input_type = input.input_type()
        controls = []
        for i in range(1, n+1):
            if input_type == "slider":
                controls.append(
                    ui.input_slider(f"extra_slider_{i}", f"Extra Slider {i}", min=0, max=100, value=25)
                )
            else:
                controls.append(
                    ui.input_numeric(f"extra_numeric_{i}", f"Extra Numeric {i}", value=10, min=0, max=100)
                )
        return controls

    @output
    @render.ui
    def tabset():
        n = input.num_inputs()
        tabs = []
        for i in range(1, n+1):
            tabs.append(
                ui.tab_panel(
                    f"Tab {i}",
                    ui.markdown(f"**This is Tab {i}**\n\nInput type: {input.input_type().capitalize()}")
                )
            )
        return ui.navset_tab(*tabs)

app = App(app_ui, server, debug=True)
```