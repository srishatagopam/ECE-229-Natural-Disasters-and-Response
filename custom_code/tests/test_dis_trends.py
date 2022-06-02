import pytest
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

from custom_code.ML_models.dis_trends import *

path = './custom_code/ML_models/1970-2022_DISASTERS.xlsx - emdat data.csv'

@pytest.fixture
def disaster_dataset():
    df1 = pd.read_csv(path)
    return df1.head(100)
    
def test_distrends_module1(disaster_dataset):
    assert isinstance(dis_analysis_(disaster_dataset, voila=True), jupyter_bokeh.BokehModel)
    
def test_distrends_module2(disaster_dataset):
    assert isinstance(dis_analysis_(disaster_dataset, voila=True), pn.Column)
    
def test_bad_input1():
    with pytest.raises(AssertionError):
        dis_analysis_([1,2,3,4,5], voila=True)
    
def test_bad_input2(disaster_dataset):
    with pytest.raises(AssertionError):
        dis_analysis_(disaster_dataset, [1,2,3,4,5])