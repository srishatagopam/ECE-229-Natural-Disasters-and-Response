#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as cls
import numpy as np

def bubble(data):
    '''
    :parameter data of type dataframe
    :implements a graph that shows death caused by each disasters from 1900 to 2021
    :filters out of columns not needed
    :keep year, disaster type and total death
    :filters out natural disaster natural that do not have a lot of total death
    :for each type of disaster, plot all of the total deaths for each year 
    :show the plot for total death by size
    :the point size increase the higher the total death is
    '''
    assert isinstance(data, pd.DataFrame)
    
    data=data[['Year', 'Disaster Type', 'Total Deaths']]
    tempdata=data.loc[(data['Disaster Type']== 'Drought') | (data['Disaster Type']== 'Volcanic activity') ]

    data=data.loc[~((data['Disaster Type'] == 'Animal accident') | (data['Disaster Type'] == 'Glacial lake outburst') 
            | (data['Disaster Type'] == 'Impact') | (data['Disaster Type'] == 'Insect infestation') 
             | (data['Disaster Type'] == 'Wildfire') | (data['Disaster Type'] == 'Mass movement (dry)')
            | (data['Disaster Type'] == 'Fog') | (data['Disaster Type']== 'Volcanic activity') | (data['Disaster Type']== 'Drought') ) ] 
    groups = data.groupby("Disaster Type")
    tempdata = tempdata.groupby('Disaster Type')
    fig,ax=plt.subplots()
    colors=plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    colors[4] = cls.to_rgba("khaki")
    colors = colors[[1,2,3,4,5,6,0,7]] 
    i=0
    for name, group in groups:
        x=ax.scatter(group['Year'], group['Disaster Type'], s=group['Total Deaths'] * .002, alpha=0.7, color=colors[i])
        i+=1
    for name, group in tempdata:
        ax.scatter(group['Year'], group['Disaster Type'], s=group['Total Deaths'] * .002, alpha=0.7, color=colors[i])
        i+=1

    ax.annotate('3M', ( 1929,'Drought'),ha='center' )
    ax.annotate('1.25M', ( 1900,'Drought'), ha='center')
    ax.annotate('1.5M', ( 1965, 5.75),ha='center', va='bottom' )
    ax.annotate('1.2M', ( 1921,'Drought'), ha='center')
    ax.annotate('3.7M', ( 1931,'Flood'), ha='center')
    ax.annotate('2.0M', ( 1958,'Flood'), ha='center')
    ax.annotate("1.3M", 
            xy=(1906.5, 1.5), 
            xytext=(1906.5, 2),
            arrowprops=dict(facecolor='blue',arrowstyle= 'simple'))
    ax.annotate("1.5M", 
            xy=(1909.5, .6), 
            xytext=(1909.5, 0.25),
            arrowprops=dict(facecolor='blue',arrowstyle= 'simple'))

    ax.annotate("2.0M", 
            xy=(1920.5, 1.5), 
            xytext=(1920.5, 1.9),
            arrowprops=dict(facecolor='blue',arrowstyle= 'simple'))
    ax.annotate("2.5M", 
            xy=(1917, .5), 
            xytext=(1917.5, 0.15),
            arrowprops=dict(facecolor='blue', arrowstyle= 'simple'))
    ax.annotate("1.5M", 
            xy=(1941, 6.10), 
            xytext=(1938, 6.75),
            arrowprops=dict(facecolor='blue', arrowstyle= 'simple'))
    ax.annotate("1.9M", 
            xy=(1945, 6.4), 
            xytext=(1946, 6.8),
            arrowprops=dict(facecolor='blue', arrowstyle= 'simple'))

    fig.set_size_inches(10,8)
    plt.title('Total Deaths due to Natural Disasters from 1900 to 2021')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




