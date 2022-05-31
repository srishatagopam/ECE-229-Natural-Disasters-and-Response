import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
import geoviews as gv

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import holoviews as hv
import panel as pn
hv.extension('bokeh')
gv.extension("bokeh")

from custom_code.ML_models.disaster_cpi_prediction import *

path = './custom_code/ML_models/1970-2022_DISASTERS.xlsx - emdat data.csv'

@pytest.fixture
def total_disaster_dataset():
    return pd.read_csv(path, nrows=10)

@pytest.fixture
def cleaned_disaster_dataset():
    return prepare_worldwide_disaster_data(path)[0].head(10)
    
@pytest.fixture
def pred_disaster_dataset():
    df, features = prepare_worldwide_disaster_data(path)
    return predict_cpi_model(df, features)[0].head(10)
    
@pytest.fixture
def features():
    df, features = prepare_worldwide_disaster_data(path)
    return features
    
@pytest.fixture
def name():
    return 'Disaster Type'
    
def test_plot_type(pred_disaster_dataset, name):
    assert isinstance(plot_worldwide_choroplethmap(pred_disaster_dataset, name), gv.Polygons)
    
def test_onehot_type(total_disaster_dataset, name):
    assert isinstance(one_hot_encode(total_disaster_dataset, name), pd.DataFrame)
    
def test_onehot_empty(total_disaster_dataset, name):
    assert not one_hot_encode(total_disaster_dataset, name).empty

def test_worldwide_path(): 
    with pytest.raises(Exception):
        prepare_worldwide_disaster_data(path + '.csv')

def test_worldwide_type():
    out = prepare_worldwide_disaster_data(path)
    assert isinstance(out[0], pd.DataFrame) and isinstance(out[1], list)
    
def test_worldwide_empty():
    out = prepare_worldwide_disaster_data(path)
    assert not out[0].empty and len(out[1]) > 0
    
def test_worldwide_feature_type():
    assert all(isinstance(i, str) for i in prepare_worldwide_disaster_data(path)[1])
    
def test_cpi_prediction_type(cleaned_disaster_dataset, features):
    out = predict_cpi_model(cleaned_disaster_dataset, features)
    assert isinstance(out[0], pd.DataFrame) and isinstance(out[1], str)
    
def test_cpi_prediction_empty(cleaned_disaster_dataset, features):
    out = predict_cpi_model(cleaned_disaster_dataset, features)
    assert not out[0].empty and len(out[1]) > 0
    
def test_cpi_prediction_module():
    assert isinstance(cpi_prediction_module(min_date = 1950,max_date =2050,voila=True), jupyter_bokeh.BokehModel)
    
def test_cpi_prediction_module():
    assert isinstance(cpi_prediction_module(min_date = 1950,max_date =2050,voila=False), pn.Column)
    