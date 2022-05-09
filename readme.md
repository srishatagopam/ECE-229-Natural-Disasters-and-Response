<h1>Exploratory Analysis of Natural Disasters</h1>

![alt text](image/bubble.png)

<h2>Team Members</h2>

By Jaya Konda, Pujika Kumar, Gaopo Huang, Jiawei Zheng, Andy Liu

<h2>Problem </h2>

Can natural disasters increase public awareness of climate change?
Is there any trend in natural disaster incidents and casualties from natural disasters over the years? 
How impactful is the media interest of climate change over the natural disaster incidents?

<h2> Motivation </h2>
Our primary motivation was to analyse the natural disasters to find if their occurrence or the deaths caused by them have stirred up any awareness about the climate change. Along the way, we have tried to draw important insights about the trends of natural disasters, and their impact on human lives.

<h2>Dataset</h2>

Natural Disaster incident over years ( https://www.kaggle.com/brsdincer/all-natural-disasters-19002021-eosdis). This dataset contains 2 csv files, with each containing 45 columns having information about year, disaster type, country, etc. 

Media (specifically TV news) interest in climate change over years: (https://blog.gdeltproject.org/a-new-dataset-for-exploring-climate-change-narratives-on-television-news-2009-2020/). This data contains 418 csv files, with each containing information about TV news reported on these natural disasters. 

Capital over gdp. This dataset contains columns and 19879 rows
(https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD)

<h2>File Structure</h2>

    .
    ├── Code                                # .py files that contains our code to generate the graph
    │   ├──__init__.py
    │   ├──bubble.py
    │   ├──major_disaster_analysis.py
    │   ├──natural_disaster_climate_news_analysis.py
    │   ├──pies.py  
    │   ├──stacked_decadal.py
    │   └──stacked_plots.py  
    ├── dataset				                # all the datasets we used
    │   ├── TelevisionNews
    |   |   └──*.CSV
    │   ├──1900-2021_DISASTERS.xlsx - emdat data.csv
    │   ├──1970-2021_DISASTERS.xlsx - emdat data.csv
    │   └──gdp_per_capita.csv
    ├── image                               # the graph we generated
    │   └── bubble.png
    │   
    ├── Final presentation PPT.pdf          # pdf file for our presentation
    ├── Final_project_code.ipynb		    # notebook to display all our visualizations
    ├── readme.md							# readme file
    └──.gitignore

<h2>Required Packages</h2>

* pandas

```
pip install pandas
```

* numpy

```
pip install numpy
```

* matplotlib

```
pip install matplotlib
```

* geopandas

```
pip install geopandas
```

* seaborn

```
pip install seaborn
```

<h2> Visualization </h2>
[Visualization Notebook](https://github.com/js-konda/ece-143/blob/main/Final_project_code.ipynb)
