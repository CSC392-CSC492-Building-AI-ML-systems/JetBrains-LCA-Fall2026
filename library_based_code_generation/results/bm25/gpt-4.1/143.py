```python
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_fluid(
    ui.layout_columns(
        # First column: controlling inputs
        ui.card(
            ui.input_slider("num_inputs", "Number of Inputs (2-5)", min=2, max=5, value=3),
            ui.input_select("input_type", "Input Type", choices=["Numeric", "Text"], selected="Numeric"),
            title="Controls"
        ),
        # Second column: controlled inputs
        ui.card(
            ui.output_ui("controlled_inputs"),
            title="Controlled Inputs"
        ),
        # Third column: more inputs and a tabset
        ui.card(
            ui.output_ui("extra_inputs"),
            ui.output_ui("tabset"),
            title="Extras"
        ),
        col_widths=[4, 4, 4]
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def controlled_inputs():
        n = input.num_inputs()
        t = input.input_type()
        inputs = []
        for i in range(1, n+1):
            if t == "Numeric":
                inputs.append(
                    ui.input_numeric(f"num_{i}", f"Numeric Input {i}", value=i, min=0, max=100)
                )
            else:
                inputs.append(
                    ui.input_text(f"text_{i}", f"Text Input {i}", value=f"Text {i}")
                )
        return inputs

    @output
    @render.ui
    def controlled_inputs():
        return controlled_inputs()

    @reactive.Calc
    def extra_inputs():
        t = input.input_type()
        if t == "Numeric":
            return [
                ui.input_slider("extra_slider", "Extra Slider", min=0, max=50, value=25),
                ui.input_numeric("extra_num", "Extra Numeric", value=10)
            ]
        else:
            return [
                ui.input_text("extra_text", "Extra Text", value="Hello"),
                ui.input_password("extra_pass", "Extra Password")
            ]

    @output
    @render.ui
    def extra_inputs():
        return extra_inputs()

    @output
    @render.ui
    def tabset():
        t = input.input_type()
        if t == "Numeric":
            return ui.navset_tab(
                ui.nav("Summary", ui.output_text("summary")),
                ui.nav("Details", ui.output_text("details"))
            )
        else:
            return ui.navset_tab(
                ui.nav("Overview", ui.output_text("overview")),
                ui.nav("Info", ui.output_text("info"))
            )

    @output
    @render.text
    def summary():
        n = input.num_inputs()
        vals = [input.get(f"num_{i}") for i in range(1, n+1)]
        return f"Sum: {sum([v or 0 for v in vals])}"

    @output
    @render.text
    def details():
        n = input.num_inputs()
        vals = [input.get(f"num_{i}") for i in range(1, n+1)]
        return f"Values: {vals}"

    @output
    @render.text
    def overview():
        n = input.num_inputs()
        vals = [input.get(f"text_{i}") for i in range(1, n+1)]
        return f"Texts: {vals}"

    @output
    @render.text
    def info():
        return "This is the Info tab for text inputs."

app = App(app_ui, server, debug=True)
```