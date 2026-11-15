```python
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_fluid(
    ui.h2("Three Column Py-Shiny App"),
    ui.layout_columns(
        # First Column: Controllers
        ui.panel_well(
            ui.input_slider("num_options", "Number of Options", min=2, max=10, value=5),
            ui.input_select("color_theme", "Color Theme", choices=["Red", "Green", "Blue"], selected="Red"),
            width=4
        ),
        # Second Column: Controlled Inputs
        ui.panel_well(
            ui.input_select("option_select", "Select Option", choices=[f"Option {i}" for i in range(1, 6)]),
            ui.input_slider("option_slider", "Option Slider", min=1, max=5, value=3),
            width=4
        ),
        # Third Column: More Inputs and Tabset
        ui.panel_well(
            ui.input_checkbox("show_extra", "Show Extra", value=True),
            ui.input_numeric("numeric_input", "Numeric Input", value=10, min=0, max=100),
            ui.navset_tab(
                ui.nav("Tab 1", ui.output_text_verbatim("tab1_out")),
                ui.nav("Tab 2", ui.output_text_verbatim("tab2_out")),
            ),
            width=4
        )
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.effect
    def _():
        # Update option_select and option_slider based on num_options
        num = input.num_options()
        session.send_input_message("option_select", {
            "choices": [f"Option {i}" for i in range(1, num + 1)],
            "selected": f"Option 1"
        })
        session.send_input_message("option_slider", {
            "min": 1,
            "max": num,
            "value": min(input.option_slider() or 1, num)
        })

    @reactive.effect
    def _():
        # Update numeric_input and show_extra based on color_theme
        theme = input.color_theme()
        if theme == "Red":
            session.send_input_message("numeric_input", {"min": 0, "max": 50, "value": 10})
            session.send_input_message("show_extra", {"value": True})
        elif theme == "Green":
            session.send_input_message("numeric_input", {"min": 10, "max": 100, "value": 55})
            session.send_input_message("show_extra", {"value": False})
        else:  # Blue
            session.send_input_message("numeric_input", {"min": 20, "max": 80, "value": 30})
            session.send_input_message("show_extra", {"value": True})

    @output
    @render.text
    def tab1_out():
        return f"Selected Option: {input.option_select()}"

    @output
    @render.text
    def tab2_out():
        return f"Numeric Input: {input.numeric_input()} | Show Extra: {input.show_extra()}"

app = App(app_ui, server)

if __name__ == "__main__":
    from shiny import run_app
    run_app(app, debug=True)
```