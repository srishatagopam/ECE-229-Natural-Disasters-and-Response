import pytest
import pandas as pd
import xgboost as xgb
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import geoviews as gv
from sklearn.model_selection import train_test_split

import panel as pn
import holoviews as hv
import jupyter_bokeh

gv.extension("bokeh")

from custom_code.ML_models.realtime_earthquake_prediction import *

@pytest.fixture
def df_past():
    return prepare_earthquake_data()[0]
    
@pytest.fixture
def df_future():
    return prepare_earthquake_data()[1]
    
@pytest.fixture
def title():
    return 'plausible title'
    

def test_plot_show(monkeypatch, df_past, title):
    monkeypatch.setattr(plt, 'show', lambda: None)
    plot_geomap(df_past, title)
    
def test_inter_scatter_type1(df_past, title):
    assert isinstance(plot_interactive_scattermap(df_past, title, point_size=5, point_color="tomato", indep=True), hv.Overlay)
    
def test_inter_scatter_type2(df_past, title):
    out = plot_interactive_scattermap(df_past, title, point_size=5, point_color="tomato", indep=False)
    assert isinstance(out[0], gv.Polygons) and isinstance(out[1], gv.Points)
    
def test_prepare_type():
    out = prepare_earthquake_data()
    assert isinstance(out[0], pd.DataFrame) and isinstance(out[1], pd.DataFrame)
    
def test_prepare_empty():
    out = prepare_earthquake_data()
    assert (not out[0].empty) and (not out[1].empty)

def test_predict_type1(df_past, df_future):
    out = predict_earthquake_model(df_past, df_future)
    assert isinstance(out[0], list) and isinstance(out[1], pd.DataFrame)
    
def test_predict_type2(df_past, df_future):
    assert all(isinstance(i, str) for i in predict_earthquake_model(df_past, df_future)[1])
    
def test_predict_empty(df_past, df_future):
    out = predict_earthquake_model(df_past, df_future)
    assert (len(out[0]) > 0) and (not out[1].empty)
    
def test_roc_show(monkeypatch, df_past):
    monkeypatch.setattr(plt, 'show', lambda: None)
    plot_roc(df_past)
    
def test_prediction_module():
    assert isinstance(predicted_earthquake_module(voila=True), jupyter_bokeh.BokehModel)
    
def test_prediction_module():
    assert isinstance(predicted_earthquake_module(voila=False), pn.Column)