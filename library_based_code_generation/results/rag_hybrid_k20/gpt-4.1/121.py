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
    age_groups = [f"{i}-{i+4}" for i in range(0, 85, 5)] + ['85+']
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
    # Simulate total population (known and predicted)
    pop_data = []
    for loc in locations:
        for year in years:
            known = year <= 2022
            total = df[(df['location'] == loc) & (df['year'] == year)]['male'].sum() + \
                    df[(df['location'] == loc) & (df['year'] == year)]['female'].sum()
            pop_data.append({
                'location': loc,
                'year': year,
                'population': total,
                'known': known
            })
    pop_df = pd.DataFrame(pop_data)
    return df, pop_df

# Load data
population_df, total_pop_df = load_population()

# Widgets
locations = sorted(population_df['location'].unique())
years = sorted(population_df['year'].unique())
age_groups = list(population_df['age_group'].unique())

location_select = Select(title="Location", value=locations[0], options=locations)
year_slider = Slider(start=years[0], end=years[-1], value=years[0], step=1, title="Year")

# Data sources
def get_pyramid_source(location, year):
    df = population_df[(population_df['location'] == location) & (population_df['year'] == year)]
    df = df.sort_values('age_group')
    return ColumnDataSource(data={
        'age_group': df['age_group'],
        'male': -df['male'],  # Negative for left side
        'female': df['female']
    })

def get_line_source(location):
    df = total_pop_df[total_pop_df['location'] == location].sort_values('year')
    return ColumnDataSource(data={
        'year': df['year'],
        'population': df['population'],
        'known': df['known']
    })

pyramid_source = get_pyramid_source(location_select.value, year_slider.value)
line_source = get_line_source(location_select.value)

# Population Pyramid Plot
pyramid_plot = figure(y_range=age_groups, plot_height=400, plot_width=400,
                      title="Population Pyramid", toolbar_location=None, x_axis_label="Population", y_axis_label="Age Group")
male_bar = pyramid_plot.hbar(y='age_group', right='male', left=0, height=0.8, color="#718dbf", source=pyramid_source, legend_label="Male")
female_bar = pyramid_plot.hbar(y='age_group', right=0, left='female', height=0.8, color="#e84d60", source=pyramid_source, legend_label="Female")
pyramid_plot.legend.location = "top_left"
pyramid_plot.xaxis.formatter.use_scientific = False

# Line Chart Plot
line_plot = figure(plot_height=400, plot_width=600, title="Population Over Time",
                   x_axis_label="Year", y_axis_label="Population", toolbar_location=None)
known_line = line_plot.line('year', 'population', source=line_source, line_width=2, color="green", legend_label="Known")
predicted_line = line_plot.line('year', 'population', source=line_source, line_width=2, color="orange", line_dash="dashed", legend_label="Predicted")
# Add circle markers for known and predicted
known_circles = line_plot.circle('year', 'population', source=line_source, size=6, color="green", legend_label="Known",
                                 view=ColumnDataSource({'year': total_pop_df[total_pop_df['known']].year,
                                                        'population': total_pop_df[total_pop_df['known']].population}))
predicted_circles = line_plot.circle('year', 'population', source=line_source, size=6, color="orange", legend_label="Predicted",
                                     view=ColumnDataSource({'year': total_pop_df[~total_pop_df['known']].year,
                                                            'population': total_pop_df[~total_pop_df['known']].population}))
line_plot.legend.location = "top_left"
line_plot.xaxis.ticker = years

# Update functions
def update_data(attr, old, new):
    location = location_select.value
    year = year_slider.value
    # Update pyramid
    new_pyramid = get_pyramid_source(location, year)
    pyramid_source.data = new_pyramid.data
    # Update line
    new_line = get_line_source(location)
    line_source.data = new_line.data

location_select.on_change('value', update_data)
year_slider.on_change('value', update_data)

# Layout
layout = column(row(location_select, year_slider), row(pyramid_plot, line_plot))

# Save layout to HTML
output_file("widget.html")
save(layout)

# Add to document for Bokeh server
curdoc().add_root(layout)
curdoc().title = "Population Visualization"
```