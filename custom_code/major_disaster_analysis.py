import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas    
import geoviews as gv
from cartopy import crs
gv.extension('bokeh')

def death_vs_gdp(natural_disaster_df, gdp_df):
    """
    Plots a graph that shows the Total Death counts vs the GDP per capita value
    for disasters that caused over 5000 deaths.
    
    **:natural_disaster_df: pd.DataFrame**
        The natural disaster dataframe.
    
    
    **:gdp_df: pd.DataFrame**
        The GDP per capita dataframe
        
    
    **Returns: None**
    """
    assert isinstance(natural_disaster_df, pd.DataFrame)
    assert isinstance(gdp_df, pd.DataFrame)

    # Join the disaster dataframe with GDP dataframe on Country and Year
    new_df = pd.merge(natural_disaster_df, gdp_df,  how='left', left_on=['ISO','Year'], right_on = ['Code','Year'])
    # Filter out the major disasters
    new_df = new_df[new_df['Total Deaths'] > 5000]
    new_df = new_df[['Total Deaths','GDP per capita']].dropna()
    # Perform scatter plot
    plt.figure(figsize=(20, 12), dpi=80)
    new_df.plot.scatter(x = 'Total Deaths', y = 'GDP per capita',figsize=(8,5))
    plt.title('Disaster Total Deaths Vs GDP per capita')
    plt.xlabel('Total Deaths')
    plt.ylabel('GDP per capita (dollars per person)')
    plt.show()
    return 

def major_disaster_distribution(natural_disaster_df):
    """
    Plots a world map that displays the density of the countries that major disasters took place.
    
    **:natural_disaster_df: pd.DataFrame**
        The natural disaster dataframe
    
    **Returns: gv.Polygons object**
    """
    assert isinstance(natural_disaster_df, pd.DataFrame)
    # Do a count over the country
    country_count = natural_disaster_df.groupby('ISO').count()['Year']
    # Import the geopandas map
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # Avoid same column name problem
    country_count = pd.DataFrame(country_count).rename(columns = {'Year':'Year_2'})
    country_count['inde'] = country_count.index
    # Plot the count on the map
    world = world.merge(country_count, left_on = 'iso_a3', right_on = 'inde', how = 'left').fillna(0)
    ax = gv.Polygons(world, vdims =[('Year_2','# disasters'), ('name','Country'),]).opts(
        tools=['hover'], width=600,height=500, #projection=crs.Robinson()
    )
    return ax

