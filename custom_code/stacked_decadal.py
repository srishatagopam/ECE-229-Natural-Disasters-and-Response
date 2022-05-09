import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def stacked_plot_death_decadal(df):
    """
    df: pandas DataFrame
    
    Creates a stacked bar plot for the decadal trend of the number of deaths caused by each disaster. 
    The disaster dataset has been loaded in the df pandas dataframe. 
    """
    assert isinstance(df, pd.DataFrame) #df should be a pandas dataframe

    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2020)]
    df = df.loc[:,['Year','Disaster Type','Total Deaths']]
    df = df.fillna(0)
    
    df['Decade'] = df.apply(lambda row: get_decade(row), axis=1)
    
    df['Total'] = df.groupby(['Decade', 'Disaster Type'])['Total Deaths'].transform('sum')
    df = df.drop_duplicates(subset=['Decade', 'Disaster Type'])

    new_df=df.pivot_table(index=['Decade'], 
                      columns='Disaster Type', 
                      values='Total', 
                      aggfunc='sum').reset_index().rename_axis(None, axis=1)
    #pivoting the table to have columns as disaster type name

    new_df.set_index('Decade', inplace=True)

    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]

    new_df.loc['1900s':].plot.bar(width = 0.8, stacked = True, color = colors, figsize = (15, 8), linewidth=1, edgecolor='black')

    plt.title('Decadal global deaths due to natural disasters ', fontsize = 19)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Occurrence', fontsize = 15)
    plt.legend(loc = 2, prop = {'size': 12})

    plt.show()
    
def get_decade(row):
    if 1900 <= row['Year'] <= 1910 :
        return '1900s'
    if 1911 <= row['Year'] <= 1920 :
        return '1910s'
    if 1921 <= row['Year'] <= 1930 :
        return '1920s'
    if 1931 <= row['Year'] <= 1940 :
        return '1930s'
    if 1941 <= row['Year'] <= 1950 :
        return '1940s'
    if 1951 <= row['Year'] <= 1960 :
        return '1950s'
    if 1961 <= row['Year'] <= 1970 :
        return '1960s'
    if 1971 <= row['Year'] <= 1980 :
        return '1970s'
    if 1981 <= row['Year'] <= 1990 :
        return '1980s'
    if 1991 <= row['Year'] <= 2000 :
        return '1990s'
    if 2001 <= row['Year'] <= 2010 :
        return '2000s'
    if 2011 <= row['Year'] <= 2021:
        return '2010s'

def stacked_plot_occurence_decadal(df):
    """
    df: pandas DataFrame

    Creates a stacked bar plot for the decadal trend of the number of occurences of each disaster. 
    The disaster dataset has been loaded in the df pandas dataframe. 
    """
    assert isinstance(df, pd.DataFrame) #df should be a pandas dataframe

    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2020)]
    df = df.groupby(['Disaster Type','Year']).size().reset_index(name="Count")
    df['Decade'] = df.apply(lambda row: get_decade(row), axis=1)

    rec_dis = df.groupby(['Disaster Type','Decade']).size().reset_index(name="Count")
    rec_dis['id'] = np.divmod(np.arange(len(rec_dis)), 10)[0] + 1 
    rec_dis = rec_dis.set_index(['id', 'Disaster Type', 'Decade']).unstack('Disaster Type') # similar to pivot
                       
    #pivoting the table to have columns as disaster type name

    rec_dis.columns = [x[1] for x in rec_dis.columns] # replace the MultiIndex column names if you don't need them

    rec_dis.reset_index(inplace=True) # Now you're back in the format you needed originally, with `Class` as a column and each row as a point in space

    rec_dis = rec_dis.fillna(0)
    rec_dis=rec_dis.drop('id', axis=1)

    rec_dis=rec_dis.groupby('Decade', as_index=False).agg(lambda x: x.sum())
    rec_dis.set_index('Decade', inplace=True)
    colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    rec_dis.loc['1900s':].plot.bar(width = 0.8, stacked = True, color = colors, figsize = (15, 8), linewidth=1, edgecolor='black')

    plt.title('Decadal global occurrences of natural disasters ', fontsize = 19)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Occurrence', fontsize = 15)
    plt.legend(loc = 2, prop = {'size': 12})

    plt.show()
    

