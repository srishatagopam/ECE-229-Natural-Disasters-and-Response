from datetime import datetime
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas    
import geoviews as gv
from cartopy import crs
import panel as pn
import datetime as dt
import holoviews as hv
import jupyter_bokeh
gv.extension('bokeh')

def dis_analysis_(natural_disaster_df, voila=True):
    """
    Plots multiple disaster statistics over specified time period.
    
    **:natural_disaster_df: pd.DataFrame**
        The natural disaster dataframe
    
    **:voila: bool**
        Use to return plot that is visible in a notebook
    
    **Returns: ipywidget.widget**
        a widget incorporating panel controlling parameters and a holoviews dynamic map
    """
    assert isinstance(natural_disaster_df, pd.DataFrame)
    assert isinstance(voila, bool)
    selector_opts = {
        '# Disasters':{'dep_var':'Year', "dep_var_2":'Year_2','feature_name':'# Disasters','group_op':'count'},
        '# Deaths':{'dep_var':'Total Deaths', "dep_var_2":'TD_2','feature_name':'# Deaths','group_op':'sum'},
        '# Injured':{'dep_var':'No Injured','dep_var_2':'num_inj','feature_name':'# Injured','group_op':'sum'},
        '# Affected':{'dep_var':'No Affected','dep_var_2':'num_aff','feature_name':'# Affected','group_op':'sum'},
        '# Homeless':{'dep_var':'No Homeless','dep_var_2':'num_homeless','feature_name':'# Homeless','group_op':'sum'},
    
      
    }
    def major_disaster_map(start_time,end_time,vis_type=None):
        scale = 3/2+1/7
        
        width=int(400*scale)
        height=int(250*scale)
#         print("start",start_time.year)
#         print("end",end_time.year)
        restricted_df = natural_disaster_df[(start_time.year<=natural_disaster_df['Year']) &
                                            (natural_disaster_df['Year']<=end_time.year)]
        grouped = restricted_df.groupby('Year')
        
        country_var_dict = {}
        for i in selector_opts.keys():
            if selector_opts[i]['group_op']=='sum':
                country_var_dict[selector_opts[i]['dep_var']] = grouped[selector_opts[i]['dep_var']].sum()
            elif selector_opts[i]['group_op']=='count':
                country_var_dict[selector_opts[i]['dep_var']] = grouped[selector_opts[i]['dep_var']].count()
        

        columns_names = {selector_opts[vis_type_temp]['dep_var']:selector_opts[vis_type_temp]['dep_var_2'] 
                         for vis_type_temp in selector_opts.keys()}
        country_var = pd.DataFrame(data=country_var_dict).rename(columns = columns_names)
        country_var['year'] = country_var.index

        bar = hv.Bars(country_var,
                      ('year'),
                      (selector_opts[vis_type]['dep_var_2'],selector_opts[vis_type]['feature_name'])).opts(
            tools=['hover'],
            width=width,
            height=height, 
            xrotation=70,
            title="Trends of disasters?"
        )
        
        return bar
        
    min_date = dt.datetime(natural_disaster_df['Year'].min(),1,1)
    max_date = dt.datetime(natural_disaster_df['Year'].max()+1,1,1)
    #print(min_date, max_date)
    dateslider= pn.widgets.DateRangeSlider(start=min_date,end=max_date,value=(min_date,max_date),name='Year Range')
    selector = pn.widgets.Select(options=list(selector_opts.keys()), name='What to see?')
    
    widget_dmap = hv.DynamicMap(pn.bind(major_disaster_map, start_time=dateslider.param.value_start,
                   end_time=dateslider.param.value_end,
                   vis_type=selector.param.value))
    widget_dmap.opts(height=500,framewise=True)
    
    description = pn.pane.Markdown('''
    ## How are disasters distributed over time?
    
    We can try looking at the same data, but from a different perspective. What if we filtered by year instead of by 
    country? We could look at the overall trend of disaster occurrences, deaths, homeless, etc. over the past century. 
    Like before, you can try filtering by year or by looking at the aforementioned factors that you want to plot. 
    It's important to get a holistic understanding of how disasters affect people around the globe - it's only from 
    beyond there that we can start to get a better understanding of how to help people rebuild and guard against 
    future natural disasters.
    ''')
    if voila:
        return pn.ipywidget(pn.Column(description,pn.Row(dateslider,selector),widget_dmap))
    else:
        return pn.Column(description,pn.Row(dateslider,selector),widget_dmap)