import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def stacked_plot_damages(df):
    """
    df: pandas DataFrame
    
    Creates a stacked bar plot for the trend of Total damages incurred for each disaster. The disaster dataset has
    been loaded in the df pandas dataframe. 
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
    new_df.set_index('Year', inplace=True)
    
    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    new_df.loc[1970:].plot.bar(width = 0.8, stacked = True, color = colors, figsize = (15, 8), linewidth=1, edgecolor='black')

    plt.title('Total Damages (\'000 US$) caused due to natural disasters for 1970-2020', fontsize = 19)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Damages', fontsize = 15)
    plt.legend(loc = 0, prop = {'size': 10})

    plt.show()    

def stacked_plot_occurences(df):
    """
    df: pandas DataFrame

    
    Creates a stacked bar plot for the trend of the number of occurences of each disaster. The disaster dataset has
    been loaded in the df pandas dataframe.
    """
    assert isinstance(df, pd.DataFrame) #df should be a pandas dataframe

    rec_dis = df[(df['Year'] >= 1970) & (df['Year'] <= 2020)]
    rec_dis = rec_dis.groupby(['Disaster Type','Year']).size().reset_index(name="Count")
    
    rec_dis['id'] = np.divmod(np.arange(len(rec_dis)), 10)[0] + 1 
    
    rec_dis = rec_dis.set_index(['id', 'Disaster Type', 'Year']).unstack('Disaster Type') # similar to pivot
    rec_dis.columns = [x[1] for x in rec_dis.columns] # replace the MultiIndex column names if you don't need them

    rec_dis.reset_index(inplace=True) 
    # Now you're back in the format you needed originally, with `Class` as a column and each row as a point in space

    rec_dis = rec_dis.fillna(0)
    rec_dis=rec_dis.drop('id', axis=1)
    rec_dis=rec_dis.groupby('Year', as_index=False).agg(lambda x: x.sum())
    rec_dis.set_index('Year', inplace=True)

    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]

    rec_dis.loc[1970:].plot.bar(width = 0.8, stacked = True, color = colors, figsize = (15, 8), linewidth=1, edgecolor='black')

    plt.title('Global occurrences of natural disasters for 1970-2020', fontsize = 19)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Occurrence', fontsize = 15)
    plt.legend(loc = 2, prop = {'size': 12})

    plt.show()

    
def stacked_plot_deaths(df):
    """
    df: pandas DataFrame

    Creates a stacked bar plot for the trend of the number of deaths cause by each disaster. The disaster dataset has
    been loaded in the df pandas dataframe. 
    """
    assert isinstance(df, pd.DataFrame) #df should be a pandas dataframe

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

    new_df.set_index('Year', inplace=True)

    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]

    new_df.loc[1970:].plot.bar(width = 0.8, stacked = True, color = colors, figsize = (15, 8), linewidth=1, edgecolor='black')

    plt.title('Global deaths caused due to natural disasters for 1970-2020', fontsize = 19)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Deaths', fontsize = 15)
    plt.legend(loc = 0, prop = {'size': 10})

    plt.show()