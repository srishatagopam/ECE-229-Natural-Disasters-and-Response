import pytest
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

from custom_code.ML_models.stacked_bar import *

path = './custom_code/ML_models/1970-2022_DISASTERS.xlsx - emdat data.csv'

@pytest.fixture
def disaster_dataset():
    df1 = pd.read_csv(path)
    return df1.head(100)
    
def test_damage_type(disaster_dataset):
    assert isinstance(stacked_plot_damages(disaster_dataset), pd.DataFrame) 
    
def test_damage_empty(disaster_dataset):
    assert (not stacked_plot_damages(disaster_dataset).empty) 

def test_occur_type(disaster_dataset):
    assert isinstance(stacked_plot_occurences(disaster_dataset), pd.DataFrame) 
    
def test_occur_empty(disaster_dataset):
    assert (not stacked_plot_occurences(disaster_dataset).empty) 
    
def test_death_type(disaster_dataset):
    assert isinstance(stacked_plot_deaths(disaster_dataset), pd.DataFrame) 
    
def test_death_empty(disaster_dataset):
    assert (not stacked_plot_deaths(disaster_dataset).empty) 
    
def test_stacked_module1(disaster_dataset):
    assert isinstance(dis_analysis(disaster_dataset, voila=True), jupyter_bokeh.BokehModel)
    
def test_stacked_module2(disaster_dataset):
    assert isinstance(dis_analysis(disaster_dataset, voila=True), pn.Column)