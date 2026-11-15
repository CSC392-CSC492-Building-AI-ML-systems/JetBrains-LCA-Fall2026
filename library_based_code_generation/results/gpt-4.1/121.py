```python
import numpy as np
import pandas as pd
from bokeh.io import curdoc, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Select, Slider, Legend, LegendItem
from bokeh.plotting import figure

# Generate mock data
np.random.seed(42)
years = np.arange(2000, 2031)
locations = ['City A', 'City B', 'City C']
age_groups = [f'{i}-{i+4}' for i in range(0, 85, 5)] + ['85+']

data = []
for loc in locations:
    for year in years:
        for age in age_groups:
            male = np.random.randint(1000, 5000) if year <= 2020 else np.random.randint(1000, 5000)
            female = np.random.randint(1000, 5000) if year <= 2020 else np.random.randint(1000, 5000)
            data.append({
                'location': loc,
                'year': year,
                'age_group': age,
                'male': male,
                'female': female
            })
df = pd.DataFrame(data)

# Aggregate total population for line chart
pop_total = df.groupby(['location', 'year']).agg({'male': 'sum', 'female': 'sum'}).reset_index()
pop_total['total'] = pop_total['male'] + pop_total['female']
pop_total['known'] = pop_total['year'] <= 2020
pop_total['predicted'] = pop_total['year'] > 2020

# Widgets
location_select = Select(title="Location", value=locations[0], options=locations)
year_slider = Slider(start=years.min(), end=years.max(), value=2020, step=1, title="Year")

# Data sources
def get_pyramid_source(location, year):
    d = df[(df['location'] == location) & (df['year'] == year)]
    return ColumnDataSource(data={
        'age_group': d['age_group'],
        'male': -d['male'],  # Negative for left side
        'female': d['female']
    })

def get_line_source(location):
    d = pop_total[pop_total['location'] == location]
    return ColumnDataSource(data={
        'year': d['year'],
        'total': d['total'],
        'known': d['known'],
        'predicted': d['predicted']
    })

pyramid_source = get_pyramid_source(location_select.value, year_slider.value)
line_source = get_line_source(location_select.value)

# Population Pyramid Plot
pyramid_plot = figure(y_range=age_groups, plot_height=400, plot_width=400, title="Population Pyramid",
                      x_axis_label='Population', y_axis_label='Age Group', toolbar_location=None)

male_bar = pyramid_plot.hbar(y='age_group', right='male', left=0, height=0.8, color="#718dbf", source=pyramid_source, legend_label="Male")
female_bar = pyramid_plot.hbar(y='age_group', right=0, left='female', height=0.8, color="#e84d60", source=pyramid_source, legend_label="Female")

pyramid_plot.ygrid.grid_line_color = None
pyramid_plot.legend.location = "top_left"
pyramid_plot.xaxis.axis_label = "Population"
pyramid_plot.yaxis.axis_label = "Age Group"

# Line Chart Plot
line_plot = figure(plot_height=400, plot_width=600, title="Population Over Time",
                   x_axis_label='Year', y_axis_label='Total Population', toolbar_location=None)

known = pop_total[(pop_total['location'] == location_select.value) & (pop_total['known'])]
predicted = pop_total[(pop_total['location'] == location_select.value) & (pop_total['predicted'])]

known_line = line_plot.line('year', 'total', source=line_source, line_width=2, color="navy", legend_label="Known", 
                            line_dash='solid', alpha='known')
predicted_line = line_plot.line('year', 'total', source=line_source, line_width=2, color="orange", legend_label="Predicted", 
                                line_dash='dashed', alpha='predicted')

line_plot.legend.location = "top_left"

# Update functions
def update_data(attr, old, new):
    # Update pyramid
    new_pyramid_source = get_pyramid_source(location_select.value, year_slider.value)
    pyramid_source.data.update(new_pyramid_source.data)
    # Update line
    new_line_source = get_line_source(location_select.value)
    line_source.data.update(new_line_source.data)

location_select.on_change('value', update_data)
year_slider.on_change('value', update_data)

# Layout
layout = column(
    row(location_select, year_slider),
    row(pyramid_plot, line_plot)
)

# Save layout to HTML
output_file("widget.html")
save(layout)

curdoc().add_root(layout)
curdoc().title = "Population Visualization"
```