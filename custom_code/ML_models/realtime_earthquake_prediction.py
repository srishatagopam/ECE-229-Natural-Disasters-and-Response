
# ML Model_
# Realtime Earthquake Prediction in Next 7 Days using xgboost model

# Module Requirements
# numpy
# pandas
# matplotlib.pyplot
# geopandas
# geoviews
# xgboost == 0.80
# sklearn

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

gv.extension("bokeh")

def plot_geomap(df, title):
    """
    Show scatter plots on world map. Returns 
    
    **:df: pd.DataFrame:**
        Dataframe for dataset (including 'longitude' and 'latitude' columns).
    **:title: str:**
        Title of scatter plots.

    Returns: None
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(title, str)

    # WorldMap Background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]

    with plt.style.context(("seaborn", "ggplot")):
        world.plot(figsize=(15,10),
                color="white",
                edgecolor = "grey");
    
    plt.scatter(df['longitude'], df['latitude'], s=df['mag']*10, color="red", alpha=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

def plot_interactive_scattermap(df, title_name, point_size,point_color="tomato",indep=True):
    """
    Plot earthquake locations on worldwide scatter map.
    
    **:df: pd.DataFrame:**
        Dataframe for dataset (including 'longitude' and 'latitude' columns).
    **:title_name: str:**
        Title of scatter plots.
    **:point_size: int:**
        Scatter point size.

    Returns: geoviews map objects
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(title_name, str)

    # WorldMap Background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#     world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    world['empty_col'] = 0
    world_map = gv.Polygons(world,vdims='empty_col').opts(cmap="gray")
    points_on_map = gv.Points(df,
                              kdims=["longitude", "latitude"],
                              vdims=["mag"]).opts(color=point_color,
                                                  size=point_size,
                                                  line_color="black",
                                                  hover_color="lime",
                                                  title=title_name,
                                                  height=400,
                                                  width=600,
                                                 )
    if indep:
        return world_map * points_on_map
    else:
        return world_map, points_on_map

def prepare_earthquake_data(days_out_to_predict = 7):
    """
    Gets dataset about historical earthquakes with aggregated longitude and latitude. Returns dataset after feature extraction and dataset used to predict in next seven days.
    
    **:days_out_to_predict: int**
        Rolling window size.
    
    **Returns: (pd.DataFrame, pd.DataFrame)**
    """
    # get latest data from USGS servers
    df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')

    # Feature Extraction
    df = df.sort_values('time', ascending=True)
    # truncate time from datetime
    df['date'] = df['time'].str[0:10]

    # only keep the columns needed
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    df = df[df['mag']>0]
    df = df[df['date']<=dt.date.today().strftime('%Y-%m-%d')]
    temp_df = df['place'].str.split(', ', expand=True) 
    df['place'] = temp_df[1]
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]

    # calculate mean lat lon for simplified locations
    df_coords = df[['place', 'latitude', 'longitude']]
    df_coords = df_coords.groupby(['place'], as_index=False).mean()
    df_coords = df_coords[['place', 'latitude', 'longitude']]

    # Merge the two dataframes of mean latitude and longitude locations calculated above with dataframe only considering ['date' , 'depth', 'mag', 'place']
    df = df[['date', 'depth', 'mag', 'place']]
    df = pd.merge(left=df, right=df_coords, how='inner', on=['place'])

    # Feature Engineering and Data wrangling
    # loop through each zone and apply MA
    #eq_tmp = df.copy()
    eq_data = []
    df_live = []

    for symbol in list(set(df['place'])):
        temp_df = df[df['place'] == symbol].copy()
        temp_df['depth_avg_22'] = temp_df['depth'].rolling(window=22,center=False).mean() 
        temp_df['depth_avg_15'] = temp_df['depth'].rolling(window=15,center=False).mean()
        temp_df['depth_avg_7'] = temp_df['depth'].rolling(window=7,center=False).mean()
        temp_df['mag_avg_22'] = temp_df['mag'].rolling(window=22,center=False).mean() 
        temp_df['mag_avg_15'] = temp_df['mag'].rolling(window=15,center=False).mean()
        temp_df['mag_avg_7'] = temp_df['mag'].rolling(window=7,center=False).mean()
        temp_df.loc[:, 'mag_outcome'] = temp_df.loc[:, 'mag_avg_7'].shift(days_out_to_predict * -1)

        df_live.append(temp_df.tail(days_out_to_predict))

        eq_data.append(temp_df)

    # concat all location-based dataframes into master dataframe
    df = pd.concat(eq_data)

    # remove any NaN fields
    df = df[np.isfinite(df['depth_avg_22'])]
    df = df[np.isfinite(df['mag_avg_22'])]
    df = df[np.isfinite(df['mag_outcome'])]
    
    # prepare outcome variable
    # considered magnitude above 2.5 as dangerous hence prediction outcome as '1' elso '0'
    df['mag_outcome'] = np.where(df['mag_outcome'] > 2.5, 1,0)

    df = df[['date',
             'mag',
             'latitude',
             'longitude',
             'depth_avg_22',
             'depth_avg_15',
             'depth_avg_7',
             'mag_avg_22', 
             'mag_avg_15',
             'mag_avg_7',
             'mag_outcome']]

    # keep only data where we can make predictions
    df_live = pd.concat(df_live)
    df_live = df_live[np.isfinite(df_live['mag_avg_22'])]
    
    return df, df_live

def predict_earthquake_model(df, df_live, days_out_to_predict = 7, max_depth=3, eta=0.1):
    """
    Perform earthquake prediction using XGBoost. Returns list of next seven days ('yyyy-mm-dd') and dataframe indicating most likely earthquake occurance in next seven days.
    
    **:df: pd.DataFrame**
        Dataset after feature extraction after getting from 'prepare_earthquake_data'.
    **:df_live: pd.DataFrame**
        Dataset used to predict in next seven days from 'prepare_earthquake_data'.
    **:days_out_to_predict: int** 
        Rolling window size.
    **:max_depth: int** 
        Max tree depth; hyperparameter for XGBoost model.
    **:eta: float**
        Step size shrinkage value for addressing overfitting; hyperparameter for XGBoost model.
   

    Returns: (list, pd.DataFrame)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_live, pd.DataFrame)
    assert isinstance(days_out_to_predict, int)
    assert isinstance(max_depth, int)
    assert isinstance(eta, float)
    assert 0 < eta < 1

    # Selection of features that are needed for prediction and hence consider only them rest are just ignored for prediction purpose.
    features = [f for f in list(df) if f not in ['date', 'mag', 'mag_outcome', 'latitude', 'longitude']]

    X_train, X_test, y_train, y_test = train_test_split(df[features], df['mag_outcome'], test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtest = xgb.DMatrix(X_test[features], label=y_test)

    param = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'max_depth': max_depth,  # the maximum depth of each tree
            'eta': eta,  # the training step for each iteration
            'silent':1
            }  # logging mode - quiet}  # the number of classes that exist in this datset

    num_round = 1000  # the number of training iterations    
    early_stopping_rounds=30
    xgb_model = xgb.train(param, dtrain, num_round) 


    # train on live data
    dlive = xgb.DMatrix(df_live[features])  
    preds = xgb_model.predict(dlive)

    # add preds to live data
    df_live_set = df_live[['date', 'place', 'latitude', 'longitude']].copy()
    # add predictions back to dataset 
    df_live_set.loc[:, 'mag'] = preds
    #df_live_set = df_live_set.assign(preds=pd.Series(preds).values)

    # aggregate down dups
    df_live_set = df_live_set.groupby(['date', 'place'], as_index=False).mean()

    # increment date to include DAYS_OUT_TO_PREDICT
    df_live_set['date']= pd.to_datetime(df_live_set['date'],format='%Y-%m-%d')
    #print(df_live_set.tail())
    df_live_set['date'] = df_live_set['date'] + pd.to_timedelta(days_out_to_predict,unit='d')

    #
    days_new = list(set([d for d in df_live_set['date'].astype(str) if d > dt.datetime.today().strftime('%Y-%m-%d')]))
    days_new.sort()
    #days_new = days_new[0:7]
    #print(days_new)
    assert len(days_new) == days_out_to_predict
    days_new_interval = (df_live_set['date'] >= days_new[0]) & (df_live_set['date'] <= days_new[-1])
    df_output = df_live_set.loc[days_new_interval]
    
    return days_new, df_output

def plot_roc(df, max_depth=3, eta=0.1):
    """
    Create Receiver Operating Characteristic plot.
    
    **:df: pd.DataFrame**
        Dataset after feature extraction after getting from 'prepare_earthquake_data'.
    **:max_depth: int** 
        Max tree depth; hyperparameter for XGBoost model.
    **:eta: float**
        Step size shrinkage value for addressing overfitting; hyperparameter for XGBoost model.
    
    Returns: None
    """
    # Modules Import
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix

    assert isinstance(df, pd.DataFrame)
    assert isinstance(max_depth, int)
    assert isinstance(eta, float)
    assert 0 < eta < 1

    # Selection of features that are needed for prediction and hence consider only them rest are just ignored for prediction purpose.
    features = [f for f in list(df) if f not in ['date', 'mag', 'mag_outcome', 'latitude', 'longitude']]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['mag_outcome'], test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtest = xgb.DMatrix(X_test[features], label=y_test)

    param = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'max_depth': max_depth,  # the maximum depth of each tree
            'eta': eta,  # the training step for each iteration
            'silent': 1}  # logging mode - quiet}  # the number of classes that exist in this datset
    num_round = 5000  # the number of training iterations    
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)

    print (roc_auc_score(y_test, preds))
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print('AUC:', np.round(roc_auc,4))

    ypred_bst = np.array(bst.predict(dtest,ntree_limit=bst.best_iteration))
    ypred_bst  = ypred_bst > 0.5
    ypred_bst = ypred_bst.astype(int)  

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("Confusion Matrix: \n",confusion_matrix(y_test,ypred_bst))
    print("\nRecall 'TP/TP+FN' = ", recall_score(y_test,ypred_bst))
    
    
def predicted_earthquake_module(voila=True):
    """
    Plot the historical average locations of earthquakes as well as predicted earthquakes in the next 7 days.
    
    **:voila: bool**
        Use to convert between ipywidget output and bokeh backend output
    
    Returns: ipywidget converted panel & bokeh plot, or a panel & bokeh plot.
    """
    df_past, df_future = prepare_earthquake_data()
    df_past = df_past.drop_duplicates()
    _,ax_past = plot_interactive_scattermap(df_past, '', 5,indep=True)
    days_inter, df_output = predict_earthquake_model(df_past, df_future)
    
    df_inter = df_output[df_output['mag']>0.01]
    
    wm, ax_future = plot_interactive_scattermap(df_inter, '', 8,point_color="teal",indep=True)
    
    def pred_earth_maps(selected):
            if selected=="Future":
                ax_future.opts(color="yellow")
                ax_past.opts(color="red")
            elif selected=="Historical":
                ax_future.opts(color="red")
                ax_past.opts(color="yellow")
            return (wm*ax_past*ax_future)
    selector_options =["Future","Historical"]
    selector = pn.widgets.Select(options=selector_options, name='Highlight Points')
    widget_dmap = hv.DynamicMap(pn.bind(pred_earth_maps, selected=selector.param.value))
    widget_dmap.opts(height=500,framewise=True,title="Predicted eathquake and historical average locations")
    description = pn.pane.Markdown('''
    # Where will the next disaster strike?
    In the plots below we predict where earthquakes will occur in the next 7 days.
    ''')
    
    if voila:
        return pn.ipywidget(pn.Column(description,selector,widget_dmap))
    else:
        return pn.Column(description,selector,widget_dmap)

    