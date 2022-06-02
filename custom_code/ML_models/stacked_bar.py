import holoviews as hv
import panel as pn
import datetime as dt
import matplotlib as plt
hv.extension("bokeh")

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

def stacked_plot_damages(df):
    """
    Prepares dataframe to be used as a stacked bar plot for damages incurred over time.
    
    **:df: pd.DataFrame**
        The natural disaster dataframe
    
    **Returns: pd.DataFrame**
        Prepared dataframe to be used for stacked bar plot.
    """
    assert isinstance(df, pd.DataFrame) #df should be a pandas dataframe
 
    df = df.loc[:,['Year','Disaster Type','Total Damages (\'000 US$)']]
    #selecting all rows but only columns containing Year, Disaster Type and Total Damages
    
    df = df.fillna(0)
    new_df=df.pivot_table(index=['Year'], 
                          columns='Disaster Type', 
                          values='Total Damages (\'000 US$)', 
                          aggfunc='sum').reset_index().rename_axis(None, axis=1)
    #pivoting the table and accumulating the sum of the total damages incurred for each disaster.
    #also resetting the index to be the year

    new_df = new_df.fillna(0)   
    new_df = new_df[new_df["Year"] >= 1970]

    return new_df


def stacked_plot_occurences(df):
    """
    Prepares dataframe to be used as a stacked bar plot for disaster occurances over time.
    
    **:df: pd.DataFrame**
        The natural disaster dataframe
    
    **Returns: pd.DataFrame**
        Prepared dataframe to be used for stacked bar plot.
    """
    assert isinstance(df, pd.DataFrame) 
    #df should be a pandas dataframe

    new_df = df[(df['Year'] >= 1970) & (df['Year'] <= 2020)]
    new_df = new_df.groupby(['Disaster Type','Year']).size().reset_index(name="Count")
    
    new_df['id'] = np.divmod(np.arange(len(new_df)), 10)[0] + 1 
    
    new_df = new_df.set_index(['id', 'Disaster Type', 'Year']).unstack('Disaster Type') # similar to pivot
    new_df.columns = [x[1] for x in new_df.columns] # replace the MultiIndex column names if you don't need them

    new_df.reset_index(inplace=True) 
    # Now you're back in the format you needed originally, with `Class` as a column and each row as a point in space

    new_df = new_df.fillna(0)
    new_df=new_df.drop('id', axis=1)
    new_df=new_df.groupby('Year', as_index=False).agg(lambda x: x.sum())

    new_df = new_df[new_df["Year"] >= 1970]
    return new_df

    
def stacked_plot_deaths(df):
    """
    Prepares dataframe to be used as a stacked bar plot for deaths ocurred over time.
    
    **:df: pd.DataFrame**
        The natural disaster dataframe
    
    **Returns: pd.DataFrame**
        Prepared dataframe to be used for stacked bar plot.
    """
    assert isinstance(df, pd.DataFrame) 
    #df should be a pandas dataframe

    df = df.loc[:,['Year','Disaster Type','Total Deaths']]
    df = df.fillna(0)
    
    df['Total'] = df.groupby(['Year', 'Disaster Type'])['Total Deaths'].transform('sum')
    #making a column Total which adds the total deaths for each disaster type
    
    new_df = df.drop_duplicates(subset=['Year', 'Disaster Type'])

    new_df=new_df.pivot_table(index=['Year'], 
                      columns='Disaster Type', 
                      values='Total', 
                      aggfunc='sum').reset_index().rename_axis(None, axis=1)
    #pivoting the table to have columns as disaster type name

    new_df = new_df[new_df["Year"] >= 1970]
    return new_df



def dis_analysis(df,voila=True):
    """
    Does natural disaster trends analysis using Holoviews Dynamic Map.
    
    **:df: pd.DataFrame**
        The natural disaster dataframe.
    
    **:voila: bool**
        Use to return plot that is visible in a notebook
    
    **Returns: ipywidget.widget**
        a widget incorporating panel controlling parameters and a holoviews dynamic map

    """
    assert isinstance(df, pd.DataFrame) 

    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]    
    
    def make_bar_stacked(barType,):
        """
        barType: list
        Constructs holoviews bar natural for disaster trends analysis using Holoviews 
        """
        
        values=list(set(df['Disaster Type'].value_counts().index)-set(['Animal accident', 'Insect infestation', 'Mass movement (dry)', 'Fog','Glacial lake outburst']))
        if barType=="Total Deaths":
            df_stacked_bar_melt=stacked_plot_deaths(df).melt(id_vars=['Year'], value_vars=values)
            bar_stacked= hv.Bars(df_stacked_bar_melt, kdims=["Year", "variable"], vdims=["value"])
            bar_stacked.opts(width=700, xlabel="year", title="Stacked chart for analysis", color="species", cmap="Category20", legend_position='right',stacked=True)
            return bar_stacked
        elif barType=="Total Damages ('000 US$)":
            values=list(set(df['Disaster Type'].value_counts().index)-set(['Animal accident', 'Insect infestation', 'Mass movement (dry)', 'Fog','Glacial lake outburst']))
            df_stacked_bar_melt=stacked_plot_damages(df).melt(id_vars=['Year'], value_vars=values)
            bar_stacked= hv.Bars(df_stacked_bar_melt, kdims=["Year", "variable"], vdims=["value"])
            bar_stacked.opts(width=700, xlabel="year", title="Stacked chart for analysis", color="species", cmap="Category20", legend_position='right',stacked=True)
            return bar_stacked
        elif barType=="Disaster Occurence":
            values=list(set(df['Disaster Type'].value_counts().index)-set(['Animal accident', 'Insect infestation', 'Mass movement (dry)', 'Fog', 'Glacial lake outburst']))
            df_stacked_bar_melt=stacked_plot_occurences(df).melt(id_vars=['Year'], value_vars=values)
            bar_stacked= hv.Bars(df_stacked_bar_melt, kdims=["Year", "variable"], vdims=["value"])
            bar_stacked.opts(width=700, xlabel="year", title="Stacked chart for analysis", color="species", cmap="Category20", legend_position='right',stacked=True)
            return bar_stacked       
    scale = 3/2
    width=int(300*scale)*2
    height=int(250*scale)
    
    barTypes=['Disaster Occurence','Total Deaths', "Total Damages ('000 US$)"]
    select_damage = pn.widgets.Select(options=barTypes, name='Damage Type')
    dmap = hv.DynamicMap(pn.bind(make_bar_stacked,barType=select_damage.param.value))
    dmap = dmap.redim.values(Bar_Type = barTypes)

    dmap.opts(framewise=True)
    dmap.opts(height=height, width=width, line_width=1.0, tools=['hover'], xrotation=70)
    description = pn.pane.Markdown('''
    # What type of disasters are most common?
    ''')
    final_module = pn.Column(description,pn.Row(select_damage),dmap)
    if voila:
        return pn.ipywidget(final_module)
    else:
        return final_module
                       