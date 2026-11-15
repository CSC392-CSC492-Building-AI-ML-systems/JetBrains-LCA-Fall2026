```python
from bokeh.io import curdoc, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Select, Slider, Legend, LegendItem
from bokeh.plotting import figure
import pandas as pd
import numpy as np

# Dummy data generation for demonstration
def load_population():
    # Simulate population data
    years = np.arange(2000, 2031)
    locations = ['Country A', 'Country B']
    age_groups = [f'{i}-{i+4}' for i in range(0, 85, 5)] + ['85+']
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
    df = pd.DataFrame(data)
    # Simulate known/predicted population totals
    pop_total = []
    for loc in locations:
        for year in years:
            known = year <= 2022
            total = df[(df['location'] == loc) & (df['year'] == year)][['male', 'female']].sum().sum()
            pop_total.append({
                'location': loc,
                'year': year,
                'population': total,
                'known': known
            })
    pop_total_df = pd.DataFrame(pop_total)
    return df, pop_total_df, years, locations, age_groups

# Load data
pop_df, pop_total_df, years, locations, age_groups = load_population()

# Widgets
location_select = Select(title="Location", value=locations[0], options=locations)
year_slider = Slider(start=years[0], end=years[-1], value=years[0], step=1, title="Year")

# Data sources
def get_pyramid_source(location, year):
    df = pop_df[(pop_df['location'] == location) & (pop_df['year'] == year)]
    return ColumnDataSource(data={
        'age_group': age_groups,
        'male': -df['male'].values,  # Negative for left side
        'female': df['female'].values
    })

def get_line_source(location):
    df = pop_total_df[pop_total_df['location'] == location]
    return ColumnDataSource(data={
        'year': df['year'],
        'population': df['population'],
        'known': df['known']
    })

pyramid_source = get_pyramid_source(location_select.value, year_slider.value)
line_source = get_line_source(location_select.value)

# Population Pyramid Plot
pyramid_plot = figure(y_range=age_groups, plot_height=400, plot_width=400, title="Population Pyramid",
                      toolbar_location=None, x_axis_label='Population', y_axis_label='Age Group')
male_bar = pyramid_plot.hbar(y='age_group', right='male', left=0, height=0.8, color="#718dbf", source=pyramid_source, legend_label="Male")
female_bar = pyramid_plot.hbar(y='age_group', right=0, left='female', height=0.8, color="#e84d60", source=pyramid_source, legend_label="Female")
pyramid_plot.legend.location = "top_left"
pyramid_plot.xaxis.formatter.use_scientific = False

# Line Chart Plot
line_plot = figure(plot_height=400, plot_width=600, title="Population Over Time",
                   x_axis_label='Year', y_axis_label='Population', toolbar_location=None)
known_line = line_plot.line('year', 'population', source=line_source, line_width=2, color="green", legend_label="Known", 
                            line_dash="solid", muted_alpha=0.2)
predicted_line = line_plot.line('year', 'population', source=line_source, line_width=2, color="orange", legend_label="Predicted", 
                                line_dash="dashed", muted_alpha=0.2)

# Custom legend filtering for known/predicted
def update_line_renderers():
    df = pop_total_df[pop_total_df['location'] == location_select.value]
    known_mask = df['known']
    predicted_mask = ~df['known']
    known_line.data_source.data = {
        'year': df['year'][known_mask],
        'population': df['population'][known_mask],
        'known': df['known'][known_mask]
    }
    predicted_line.data_source.data = {
        'year': df['year'][predicted_mask],
        'population': df['population'][predicted_mask],
        'known': df['known'][predicted_mask]
    }

update_line_renderers()
line_plot.legend.click_policy = "mute"

# Callbacks
def update(attr, old, new):
    # Update pyramid
    new_source = get_pyramid_source(location_select.value, year_slider.value)
    pyramid_source.data.update(new_source.data)
    # Update line chart
    update_line_renderers()

location_select.on_change('value', update)
year_slider.on_change('value', update)

# Layout
layout = column(row(location_select, year_slider), row(pyramid_plot, line_plot))

# Save layout to HTML
output_file("widget.html")
save(layout)

# Add to document for Bokeh server
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
```