import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas    
import geoviews as gv
from cartopy import crs
import panel as pn
import datetime as dt
import holoviews as hv
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

def map_distribution_cases(natural_disaster_df):
    """
    Plots a world map that displays the density of the countries that major disasters
    took place. Can also display multiple types of dependent features. Expand on this
    function to handle multiple cases to combine multiple visualizations into one.
    
    **:natural_disaster_df: pd.DataFrame**
        The natural disaster dataframe
    
    **Returns: None**
    """
    assert isinstance(natural_disaster_df, pd.DataFrame)
    selector_opts = {
        '# Disasters':{'dep_var':'Year', "dep_var_2":'Year_2','feature_name':'# Disasters','group_op':'count'},
        '# Deaths':{'dep_var':'Total Deaths', "dep_var_2":'TD_2','feature_name':'# Deaths','group_op':'sum'},
        '# Injured':{'dep_var':'No Injured','dep_var_2':'num_inj','feature_name':'# Injured','group_op':'sum'},
        '# Affected':{'dep_var':'No Affected','dep_var_2':'num_aff','feature_name':'# Affected','group_op':'sum'},
        '# Homeless':{'dep_var':'No Homeless','dep_var_2':'num_homeless','feature_name':'# Homeless','group_op':'sum'},
        'Reconstruction Costs':{'dep_var':'Reconstruction Costs (\'000 US$)','dep_var_2':'num_recon_costs','feature_name':'Reconstruction Costs','group_op':'sum'},
        'Total Damanges':{'dep_var':'Total Damages (\'000 US$)','dep_var_2':'num_damages','feature_name':'Total Damanges','group_op':'sum'},
    }
    def major_disaster_map(start_time,end_time,vis_type=None):
        scale = 3/2+1/7
        width=int(300*scale)
        height=int(250*scale)
        
        restricted_df = natural_disaster_df[(start_time.year<=natural_disaster_df['Year']) &
                                            (natural_disaster_df['Year']<=end_time.year)]
        grouped = restricted_df.groupby('ISO')
        country_var_dict = {}
        for i in selector_opts.keys():
            if selector_opts[i]['group_op']=='sum':
                country_var_dict[selector_opts[i]['dep_var']] = grouped[selector_opts[i]['dep_var']].sum()
            elif selector_opts[i]['group_op']=='count':
                country_var_dict[selector_opts[i]['dep_var']] = grouped[selector_opts[i]['dep_var']].count()
        
        # Import the geopandas map
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        # Avoid same column name problem
        columns_names = {selector_opts[vis_type_temp]['dep_var']:selector_opts[vis_type_temp]['dep_var_2'] 
                         for vis_type_temp in selector_opts.keys()}
        
        country_var = pd.DataFrame(data=country_var_dict).rename(
            columns = columns_names
        )
        country_var['inde'] = country_var.index
        # Plot the count on the map
        world = world.merge(country_var, left_on = 'iso_a3', right_on = 'inde', how = 'left').fillna(0)
        keyDims = [(selector_opts[vis_type_temp]['dep_var_2'],selector_opts[vis_type_temp]['feature_name'])
                         for vis_type_temp in selector_opts.keys()]
        keyDims = [('name','Country')]+ keyDims
        ax = gv.Polygons(world, vdims =keyDims).opts(
            tools=['hover'], 
            width=width,
            height=height, 
            color_index=selector_opts[vis_type]['dep_var_2'],
            title="Where are disasters concentrated?"
        )
        bar = hv.Bars(country_var.sort_values(selector_opts[vis_type]['dep_var_2'],
                      ascending=False).iloc[:10],
                      ('inde','Country'),
                      (selector_opts[vis_type]['dep_var_2'],selector_opts[vis_type]['feature_name'])).opts(
            tools=['hover'],
            width=width,
            height=height, 
            xrotation=70,
            title="Which countries are affected the most?"
        )
        
        return ax+bar
        
    min_date = dt.datetime(natural_disaster_df['Year'].min(),1,1)
    max_date = dt.datetime(natural_disaster_df['Year'].max()+1,1,1)
    dateslider= pn.widgets.DateRangeSlider(start=min_date,end=max_date,value=(min_date,max_date),name='Year Range')
    selector = pn.widgets.Select(options=list(selector_opts.keys()), name='What to see?')
    
    streams = dict(start_time=dateslider.param.value_start,
                   end_time=dateslider.param.value_end,
                   vis_type=selector.param.value)
    
    
#     widget_dmap = hv.DynamicMap(major_disaster_map, streams=streams)
    widget_dmap = hv.DynamicMap(pn.bind(major_disaster_map, start_time=dateslider.param.value_start,
                   end_time=dateslider.param.value_end,
                   vis_type=selector.param.value))
    widget_dmap.opts(height=100,framewise=True)
    
    description = pn.pane.Markdown('''
    # Where are disasters happening?
    ''')
    return pn.ipywidget(pn.Column(description,pn.Row(dateslider,selector),widget_dmap))
