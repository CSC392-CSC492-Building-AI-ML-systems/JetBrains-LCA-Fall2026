```python
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_bootstrap(
    ui.layout_columns(
        # First column: Controllers
        ui.card(
            ui.input_slider("num_inputs", "Number of Inputs (2-5)", min=2, max=5, value=3),
            ui.input_select("input_type", "Input Type", {"slider": "Slider", "numeric": "Numeric"}, selected="slider"),
            title="Controllers"
        ),
        # Second column: Controlled Inputs
        ui.card(
            ui.output_ui("controlled_inputs"),
            title="Controlled Inputs"
        ),
        # Third column: More Inputs and Tabset
        ui.card(
            ui.output_ui("extra_inputs"),
            ui.navset_tab(
                ui.nav("Tab 1", ui.output_text_verbatim("tab1_content")),
                ui.nav("Tab 2", ui.output_text_verbatim("tab2_content"))
            ),
            title="Extra & Tabs"
        ),
        col_widths=[4, 4, 4]
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def controlled_input_ids():
        return [f"ctrl_input_{i+1}" for i in range(input.num_inputs())]

    @output
    @render.ui
    def controlled_inputs():
        ids = controlled_input_ids()
        input_type = input.input_type()
        controls = []
        for i, cid in enumerate(ids):
            label = f"Input {i+1}"
            if input_type == "slider":
                controls.append(ui.input_slider(cid, label, min=0, max=100, value=50))
            else:
                controls.append(ui.input_numeric(cid, label, value=0))
        return controls

    @output
    @render.ui
    def extra_inputs():
        ids = controlled_input_ids()
        controls = []
        for i, cid in enumerate(ids):
            controls.append(ui.input_checkbox(f"extra_chk_{i+1}", f"Extra Checkbox {i+1}", value=False))
        return controls

    @output
    @render.text
    def tab1_content():
        ids = controlled_input_ids()
        values = []
        for i, cid in enumerate(ids):
            val = input.get(cid)
            values.append(f"{cid}: {val}")
        return "Controlled Inputs:\n" + "\n".join(values)

    @output
    @render.text
    def tab2_content():
        ids = controlled_input_ids()
        values = []
        for i, cid in enumerate(ids):
            val = input.get(f"extra_chk_{i+1}")
            values.append(f"extra_chk_{i+1}: {val}")
        return "Extra Checkboxes:\n" + "\n".join(values)

app = App(app_ui, server, debug=True)

if __name__ == "__main__":
    app.run()
```