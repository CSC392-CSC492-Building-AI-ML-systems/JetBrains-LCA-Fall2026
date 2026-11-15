```python
from bokeh.io import curdoc, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Select, HoverTool, Span, Legend, LegendItem
from bokeh.plotting import figure
import pandas as pd
import numpy as np

# Sample data generation
years = list(range(2000, 2031))
locations = ['Country A', 'Country B']
age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

def generate_population_data():
    data = []
    for loc in locations:
        for year in years:
            for age in age_groups:
                male = np.random.randint(1000, 10000) if year <= 2020 else np.random.randint(1000, 10000)
                female = np.random.randint(1000, 10000) if year <= 2020 else np.random.randint(1000, 10000)
                data.append({
                    'location': loc,
                    'year': year,
                    'age_group': age,
                    'male': male,
                    'female': female
                })
    return pd.DataFrame(data)

def generate_total_population_data():
    data = []
    for loc in locations:
        for year in years:
            known = year <= 2020
            total = np.random.randint(500000, 1000000) if known else np.random.randint(500000, 1000000)
            data.append({
                'location': loc,
                'year': year,
                'population': total,
                'known': known
            })
    return pd.DataFrame(data)

pop_df = generate_population_data()
total_pop_df = generate_total_population_data()

# Widgets
year_select = Select(title="Year", value=str(years[0]), options=[str(y) for y in years])
location_select = Select(title="Location", value=locations[0], options=locations)

# Data sources
def get_pyramid_source(location, year):
    df = pop_df[(pop_df['location'] == location) & (pop_df['year'] == int(year))]
    return ColumnDataSource(data={
        'age_group': age_groups,
        'male': -df['male'].values,  # Negative for left side
        'female': df['female'].values
    })

def get_line_source(location):
    df = total_pop_df[total_pop_df['location'] == location]
    known = df[df['known']]
    predicted = df[~df['known']]
    return ColumnDataSource(data={
        'year': known['year'],
        'population': known['population']
    }), ColumnDataSource(data={
        'year': predicted['year'],
        'population': predicted['population']
    })

pyramid_source = get_pyramid_source(location_select.value, year_select.value)
line_known_source, line_pred_source = get_line_source(location_select.value)

# Population Pyramid Plot
pyramid_plot = figure(y_range=age_groups, plot_height=400, plot_width=400, title="Population Pyramid",
                      toolbar_location=None, tools="")

pyramid_plot.hbar(y='age_group', right='male', height=0.4, source=pyramid_source, color="#718dbf", legend_label="Male")
pyramid_plot.hbar(y='age_group', right='female', height=0.4, source=pyramid_source, color="#e84d60", legend_label="Female")

pyramid_plot.ygrid.grid_line_color = None
pyramid_plot.xaxis.axis_label = "Population"
pyramid_plot.legend.location = "top_left"
pyramid_plot.legend.orientation = "horizontal"

# Line Chart Plot
line_plot = figure(plot_height=400, plot_width=600, title="Population Over Time",
                   x_axis_label="Year", y_axis_label="Population", tools="pan,wheel_zoom,box_zoom,reset,save")

line_known = line_plot.line('year', 'population', source=line_known_source, line_width=2, color="navy", legend_label="Known")
line_pred = line_plot.line('year', 'population', source=line_pred_source, line_width=2, color="orange", line_dash="dashed", legend_label="Predicted")

line_plot.legend.location = "top_left"
line_plot.legend.click_policy = "hide"

# Update function
def update(attr, old, new):
    location = location_select.value
    year = year_select.value
    # Update pyramid
    new_pyramid_source = get_pyramid_source(location, year)
    pyramid_source.data.update(new_pyramid_source.data)
    # Update line chart
    new_known, new_pred = get_line_source(location)
    line_known_source.data.update(new_known.data)
    line_pred_source.data.update(new_pred.data)

year_select.on_change('value', update)
location_select.on_change('value', update)

# Layout
layout = column(row(location_select, year_select), row(pyramid_plot, line_plot))

# Save layout to HTML
output_file("widget.html")
save(layout)

# Add to document for Bokeh server
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
```