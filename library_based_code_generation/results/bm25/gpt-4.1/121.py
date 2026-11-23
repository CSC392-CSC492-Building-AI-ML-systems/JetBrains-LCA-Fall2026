```python
from bokeh.io import curdoc, output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, HoverTool, Legend, LegendItem
from bokeh.plotting import figure
import pandas as pd
import numpy as np

# Simulate population data
np.random.seed(42)
years = list(range(2000, 2031))
locations = ['Country A', 'Country B', 'Country C']
age_groups = [f'{i}-{i+4}' for i in range(0, 85, 5)] + ['85+']

def generate_population_data():
    data = []
    for loc in locations:
        for year in years:
            for age in age_groups:
                male = np.random.randint(1000, 10000)
                female = np.random.randint(1000, 10000)
                data.append({
                    'location': loc,
                    'year': year,
                    'age_group': age,
                    'male': male,
                    'female': female
                })
    return pd.DataFrame(data)

def generate_total_population():
    data = []
    for loc in locations:
        for year in years:
            known = year <= 2020
            total = np.random.randint(500000, 1000000) + (year-2000)*np.random.randint(10000, 20000)
            data.append({
                'location': loc,
                'year': year,
                'population': total,
                'known': known
            })
    return pd.DataFrame(data)

pop_df = generate_population_data()
total_pop_df = generate_total_population()

# Widgets
year_select = Select(title="Year", value=str(years[0]), options=[str(y) for y in years])
location_select = Select(title="Location", value=locations[0], options=locations)

# Data sources
def get_pyramid_source(location, year):
    df = pop_df[(pop_df['location'] == location) & (pop_df['year'] == int(year))]
    return ColumnDataSource(data={
        'age_group': df['age_group'],
        'male': -df['male'],  # Negative for left side
        'female': df['female']
    })

def get_line_source(location):
    df = total_pop_df[total_pop_df['location'] == location]
    return ColumnDataSource(data={
        'year': df['year'],
        'population': df['population'],
        'known': df['known']
    })

pyramid_source = get_pyramid_source(location_select.value, year_select.value)
line_source = get_line_source(location_select.value)

# Population Pyramid Plot
pyramid_plot = figure(y_range=age_groups, plot_height=400, plot_width=400, title="Population Pyramid",
                      toolbar_location=None, tools="")

pyramid_plot.hbar(y='age_group', right='male', left=0, height=0.8, color="#718dbf", source=pyramid_source, legend_label="Male")
pyramid_plot.hbar(y='age_group', right=0, left='female', height=0.8, color="#e84d60", source=pyramid_source, legend_label="Female")

pyramid_plot.ygrid.grid_line_color = None
pyramid_plot.xaxis.axis_label = "Population"
pyramid_plot.legend.location = "top_left"
pyramid_plot.title.align = "center"

# Line Chart Plot
line_plot = figure(plot_height=400, plot_width=600, title="Population Over Time",
                   x_axis_label="Year", y_axis_label="Population", tools="pan,wheel_zoom,box_zoom,reset,save")

known_mask = line_source.data['known']
years_arr = np.array(line_source.data['year'])
pop_arr = np.array(line_source.data['population'])

known_years = years_arr[known_mask]
known_pop = pop_arr[known_mask]
pred_years = years_arr[~known_mask]
pred_pop = pop_arr[~known_mask]

known_line = line_plot.line(known_years, known_pop, line_width=2, color="navy", legend_label="Known")
pred_line = line_plot.line(pred_years, pred_pop, line_width=2, color="orange", line_dash="dashed", legend_label="Predicted")

line_plot.legend.location = "top_left"
line_plot.title.align = "center"

# Update functions
def update_data(attr, old, new):
    # Update pyramid
    new_pyramid_source = get_pyramid_source(location_select.value, year_select.value)
    pyramid_source.data.update(new_pyramid_source.data)
    # Update line
    new_line_source = get_line_source(location_select.value)
    line_source.data.update(new_line_source.data)
    # Update line plot
    known_mask = np.array(line_source.data['known'])
    years_arr = np.array(line_source.data['year'])
    pop_arr = np.array(line_source.data['population'])
    known_years = years_arr[known_mask]
    known_pop = pop_arr[known_mask]
    pred_years = years_arr[~known_mask]
    pred_pop = pop_arr[~known_mask]
    known_line.data_source.data = {'x': known_years, 'y': known_pop}
    pred_line.data_source.data = {'x': pred_years, 'y': pred_pop}

year_select.on_change('value', update_data)
location_select.on_change('value', update_data)

# Layout
layout = column(row(location_select, year_select), row(pyramid_plot, line_plot))

# Save layout to HTML
output_file("widget.html")
save(layout)

# Add to document for Bokeh server
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
```