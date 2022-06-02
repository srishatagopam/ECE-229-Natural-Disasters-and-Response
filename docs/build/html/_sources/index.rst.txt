.. ECE-229-Natural-Disasters-and-Response documentation master file, created by
   sphinx-quickstart on Sun May 22 19:06:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**ECE 229 - Natural Disasters and Response**
==================================================================

As a non-profit organization we want our end users/donators to understand the importance of
our work. Our goal is to convince English speaking potential donors to donate to our company
so we can continue our company’s mission: “help others through providing humanitarian aid.”
Through our dashboard we can educate potential donors on the magnitude of damage that
natural disasters can cause. Utilizing interactive visuals provide a digestible way to learn the
impact of natural disasters around the world such as the number of deaths, and monetary
damage.




.. toctree::
   :maxdepth: 2
   :caption: Contents:


..
	Indices and tables
	==================

	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

API Documention
===============

There are four main components to our dashboard that create a collection of visualizations that serve 
educate our end users. The following sections will show each of these four main components and their
associated modules.

Disaster Analysis - Geographical
--------------------------------

The main visualization here is allowing an end user to visualize disaster statistics such as the distribution 
of disasters, deaths, homeless, injured, affected, damages, and reconstruction costs on a geographical basis: 
the maximum amount of granularity offered here is per country. This is shown both on a chloropleth map with 
a colormap for intensity of each statistic as well as a bar plot showing the top ten countries per statistic. 
The user is able to pick each statistic from a dropdown menu and fine-tune the period of the distribution in 
terms of years.

.. automodule:: custom_code.major_disaster_analysis
    :members:
	
Disaster Analysis - Temporal
----------------------------

There are two main visualizations showcasing the distributions of disasters over time. The first is a bar plot 
that similarly shows the previously mentioned disaster statistics over a specified time period (again accessible 
through a dropdown menu and range bar), as well as a dynamic stacked bar plot that filters for specific types of 
disasters and measures disaster occurance, deaths, and economic damages (stratified via a dropdown menu).

.. automodule:: custom_code.ML_models.dis_trends
	:members:
	
.. automodule:: custom_code.ML_models.stacked_bar
	:members:
	
CPI Prediction Model
--------------------

Similar to the geographical disaster analysis, this module predicts a country's CPI: community preparedness index 
based on the features of the dataset. Users can pick a year, month, and disaster type of predict each country's CPI; 
a bar plot will rank the lowest ten CPIs. Prediction is done using an XGBoost model.
	
.. automodule:: custom_code.ML_models.disaster_cpi_prediction
	:members:
	
Earthquake Prediction Model
---------------------------

The purpose of this visualization is twofold: first, users can pick from a dropdown whether to highlight historical or 
predicted earthquakes in the next seven days. Prediction is again done using XGBoost.
	
.. automodule:: custom_code.ML_models.realtime_earthquake_prediction
	:members:



