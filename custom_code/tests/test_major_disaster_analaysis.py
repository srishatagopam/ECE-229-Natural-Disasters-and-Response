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

from custom_code.major_disaster_analysis import *

path = './datasets/'

@pytest.fixture
def disaster_dataset():
    return pd.read_csv(path + '1970-2022_DISASTERS.xlsx - emdat data.csv', nrows=10)

@pytest.fixture
def gdp_dataset():
    return pd.read_csv(path + 'gdp_per_capita.csv', nrows=10)
    
@pytest.fixture
def assert_check():
    return [1,2,3,4,5]
    

def test_plot_show(monkeypatch, disaster_dataset, gdp_dataset):
    monkeypatch.setattr(plt, 'show', lambda: None)
    death_vs_gdp(disaster_dataset, gdp_dataset)
    
def test_same_input1(monkeypatch, disaster_dataset):
    with pytest.raises(Exception):
        monkeypatch.setattr(plt, 'show', lambda: None)
        death_vs_gdp(disaster_dataset, disaster_dataset)
    
def test_same_input2(monkeypatch, gdp_dataset):
    with pytest.raises(Exception):
        monkeypatch.setattr(plt, 'show', lambda: None)
        death_vs_gdp(gdp_dataset, gdp_dataset)

def test_ax_type(disaster_dataset):
    assert isinstance(major_disaster_distribution(disaster_dataset), gv.Polygons)
    
def test_map_dist_voila(disaster_dataset):
    assert isinstance(map_distribution_cases(disaster_dataset, voila=True), jupyter_bokeh.BokehModel)
    
def test_map_dist_column(disaster_dataset):
    assert isinstance(map_distribution_cases(disaster_dataset, voila=False), pn.Column)

def test_same_input3(gdp_dataset):
    with pytest.raises(Exception):
        map_distribution_cases(gdp_dataset)
    
def test_check_input1(assert_check, gdp_dataset):
    with pytest.raises(AssertionError):
        death_vs_gdp(assert_check, gdp_dataset)
    
def test_check_input2(assert_check):
    with pytest.raises(AssertionError):
        major_disaster_distribution(assert_check)
    
def test_check_input3(assert_check):
    with pytest.raises(AssertionError):
        map_distribution_cases(assert_check)
    