import pandas as pd
import numpy as np
import geopandas as gpd
import geoviews as gv

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

gv.extension("bokeh")


def plot_worldwide_choroplethmap(df, title_name):
    """
    Plot earthquake locations worldwide scatter map.

    **:df: pd.DataFrame:**
        Dataset with 'country_name' and 'CPI' info.
    **:title_name: str:**
        Plot title name.

    Returns: gv.Polygons object
    """

    # WorldMap Background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.name!="Antarctica")]
    
    preds = world.merge(df, left_on='name', right_on='country_name')

    img = gv.Polygons(data=preds, vdims=["CPI", "name"]).opts(height=400, width=600, title=title_name, colorbar=True, tools=['hover',], hover_color='lime')

    return img

# Worldwide disaster CPI Prediction
def one_hot_encode(df, name):
    """
    Do one-hot enconding on dataframe columns.
    
    **:df: pd.DataFrame:**
        Total dataset.
    **:title_name: str:**
        One-hot encoding target.
        

    Returns: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(name, str)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df[[name]]).toarray())

    return encoder_df

def prepare_worldwide_disaster_data(file_name = '1970-2022_DISASTERS.xlsx - emdat data.csv'):
    """
    Do dataset extraction and preparation for model-building. Returns tuple of cleaned dataframe and feature list.
    
    **:file_name: str**
        File path for .csv dataset.

    Returns: (pd.DataFrame, list)
    """
    #df = pd.read_csv ('./datasets/1970-2022_DISASTERS.xlsx - emdat data.csv')
    # Pretreatment
    df = pd.read_csv(file_name)
    df=df.drop(columns=['Dis No','Seq','Glide','Disaster Group', 'OFDA Response', 'Appeal', 'Declaration', 'Aid Contribution', 
                        'Associated Dis', 'Associated Dis2', 'Dis Mag Value', 'Dis Mag Scale',
                        'Insured Damages (\'000 US$)', 'Adm Level', 'Admin1 Code', 'Admin2 Code', 'Event Name'])

    df = df[~(df['Disaster Type'].isin(["Insect infestation", "Animal accident","Impact","Glacial lake outburst","Fog","Mass movement (dry)", 'Epidemic', 'Drought', 'Volcanic activity']))]
    # Feature Extraction
    df_sub = df[['Year', 'Disaster Type', 'Country', 'ISO', 'Region', 'Continent', 'Location', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Deaths', 'Total Damages (\'000 US$)', 'CPI']]
    df_sub = df_sub.rename(columns={'Total Damages (\'000 US$)': 'Total Damages'})
    # Filter ISO frequency less than 100
    df_sub_ = df_sub.copy().groupby('ISO').filter(lambda x: len(x)>100)
    # HKG not shown
    df_sub_ = df_sub_.loc[df_sub_['ISO'] !=  "HKG"]
    # disasters and ISO codebook
    #disaster_codebook = dict(zip([i for i in range(len(df_sub_['Disaster Type'].unique()))], df_sub_['Disaster Type'].unique()))
    #country_codebook = dict(zip([i for i in range(len(df_sub_['ISO'].unique()))], df_sub_['ISO'].unique()))
    # Filter Year Interval bigger than 0
    df_sub_['Year Interval'] = df_sub_['End Year'] - df_sub_['Start Year']
    df_sub_ = df_sub_[df_sub_['Year Interval']<1]
    # Get features what we need
    df_sample = df_sub_[['Year', 'Disaster Type', 'Country', 'ISO', 'Region', 'Continent', 'Location', 'Start Month', 'End Month', 'Total Deaths', 'Total Damages','CPI']]
    df_sample = df_sample.reset_index(drop=True)
    # One-Hot encoding
    # Disaster Type encoding
    encoder_df1 = one_hot_encode(df_sample, 'Disaster Type')
    df_encoder1 = df_sample.copy().join(encoder_df1)
    disaster_cnt = len(df_encoder1['Disaster Type'].unique())
    assert disaster_cnt == 6
    disaster_codebook = dict(zip([i for i in range(disaster_cnt)], sorted(df_encoder1['Disaster Type'].unique())))
    df_encoder1.drop('Disaster Type', axis=1, inplace=True)
    df_encoder1 = df_encoder1.rename(columns=disaster_codebook)
    # ISO encoding
    encoder_df2 = one_hot_encode(df_encoder1, 'ISO')
    df_encoder2 = df_encoder1.copy().join(encoder_df2)
    country_cnt = len(df_encoder2['ISO'].unique())
    assert country_cnt == 26
    country_codebook = dict(zip([i for i in range(country_cnt)], sorted(df_encoder2['ISO'].unique())))
    df_encoder2.drop('ISO', axis=1, inplace=True)
    df_encoder2 = df_encoder2.rename(columns=country_codebook)
    # Output
    df_output = df_encoder2.copy().dropna(subset=['CPI'])
    df_output = df_output.dropna(subset=['Start Month'])
    df_output = df_output.reset_index(drop=True)
    df_output['Month Interval'] = df_output['End Month']-df_output['Start Month']
    # features
    features = ['Year', 'Start Month', 'Month Interval'] + list(disaster_codebook.values()) + list(country_codebook.values()) + ['CPI']
    df_final = df_output[features]

    return df_final, features

def predict_cpi_model(df, features, year=2023, month=7, disaster='Wildfire'):
    """
    Finds CPI of a specific disaster type across the world in the future. Returns prediction dataframe (country_name, CPI) and title of plot.
    
    **:df: pd.DataFrame**
        Dataset after preparation and feature extraction.
    **:features: list**
        Feature extraction list.
    **:year: int**
        Year for inference.
    **:month: int**
        Month for inference.
    **:disaster: str**
        One type from ['Earthquake', 'Extreme temperature', 'Flood', 'Landslide', 'Storm', 'Wildfire'].
    
    Returns: (pd.DataFrame, str)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(features, list)
    assert isinstance(year, int)
    assert isinstance(month, int)
    assert isinstance(disaster, str)
    assert 1 <= month <= 12
    assert disaster in ['Earthquake', 'Extreme temperature', 'Flood', 'Landslide', 'Storm', 'Wildfire']
    
    X = df.loc[:, df.columns != 'CPI']
    y = df.loc[:, df.columns == 'CPI']
    
    data_dmatrix = xgb.DMatrix(data=X,label=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                              max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg = xg_reg.fit(X_train,y_train)
    
    # ['Year', 'Start Month', 'Month Interval'] + list(disaster_codebook.values()) + list(country_codebook.values()) + ['CPI']
    X_preds = pd.DataFrame(0, index=[i for i in range(26)], columns=features[:-1])
    X_preds.loc[:, 'Year'] = year
    X_preds.loc[:, 'Start Month'] = month

    country_codebook_names = ['AFG', 'ARG', 'AUS', 'BGD', 'BRA', 'CAN', 'CHN', 'COL', 'FRA', 'HTI', 'IDN', 'IND', 'IRN', 'ITA', 'JPN', 'KOR', 'MEX', 'NPL', 'PAK', 'PER', 'PHL', 'RUS', 'THA', 'TUR', 'USA', 'VNM']
    for i in X_preds.index:
        X_preds.at[i, country_codebook_names[i]] = 1

    X_preds.loc[:, disaster] = 1

    y_preds = xg_reg.predict(X_preds)

    country_name = np.array(['Afghanistan', 'Argentina', 'Australia', 'Bangladesh', 'Brazil', 'Canada', 'China', 'Colombia', 'France', 'Haiti', 'Indonesia',
                             'India', 'Iran', 'Italy', 'Japan', 'South Korea', 'Mexico', 'Nepal', 'Pakistan', 'Peru', 'Philippines', 'Russia', 'Thailand',
                             'Turkey', 'United States of America', 'Vietnam'])
    month_codebook = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    title_name = 'CPI Prediction ' + 'of ' + disaster + ' in ' + month_codebook[month] + ' ' + str(year)
    df_preds = pd.DataFrame({'country_name': country_name, 'CPI': y_preds})

    return df_preds, title_name



    