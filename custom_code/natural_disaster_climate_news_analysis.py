
import pandas as pd 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as cls

def all_occurrences(df): 
    """
    Show all occurrences of natural disasters of the dataset as visualization
    by each disaster type

    :param df: natural disaster dataset dataframe
    :type df: pd.Dataframe
    """
    # Group the dataframe by the disaster type
    result = df.groupby("Disaster Type").size().sort_values(ascending=False)
    result.index = [ind.strip() for ind in result.index]
    # Plot the result 
    fig,ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.bar(result.index, result.values, color='steelblue')
    # Do not show axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(bottom = False, left = False)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=45, ha="right" );
    plt.title("Occurence of natural disasters over the past century",size=14,fontweight="bold")
    ax.set_ylabel('Number of occurences', rotation=90)


def date_transform(date_list):
    """
    Transform date from a list of [year,month,day] or [year, month] to parse-able datetime
    string format. Used by *disaster_data_cleaning_for_news_analysis()* method to add a date 
    column.

    :param date_list: A list of date information, can be either [year,month,day] or [year,month]
    :type date_list: list
    :return out_date: The output date string 
    :rtype: string
    """
    assert len(date_list)==2 or len(date_list)==3
    # Generate the datetime object then use its string format from the columns of [year,month,day]
    if len(date_list)==2:
        out_date = '{:04d}-{:02d}'.format(date_list[0].astype(int),date_list[1].astype(int))
        out_date = datetime.datetime.strptime(out_date,"%Y-%m").strftime("%Y-%b")
    else:
        out_date = '{:04d}-{:02d}-{:02d}'.format(date_list[0].astype(int),date_list[1].astype(int),date_list[2].astype(int))
        out_date = datetime.datetime.strptime(out_date,"%Y-%m-%d").strftime("%Y-%b")
    return out_date

def disaster_data_cleaning_for_news_analysis(df):
    """
    Clean the Disaster dataset to match the time interval Television News Dataset 

    :param df: 1970-2019 Natural Disaster dataset
    :type df: pd.Dataframe
    :return df: Processed dataframe with filtered information
    :rtype df: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    # Generate the start date column for filtering
    df = df.copy()
    df.dropna(subset=['Start Month'],inplace=True)
    startdate_cols = ['Start Year','Start Month']
    df['startdate'] = df[startdate_cols].apply(lambda x: date_transform(x),axis="columns")
    # Keep only the US and time data needed
    df = df.loc[(df["ISO"] == "USA") & (df["Year"]<2020) & (df["Year"]>=2009)]
    df.index = np.arange(len(df.index))
    # Remove very rare disasters with little impact 
    removing_disaster_type = ['Mass movement (dry)','Insect infestation', 'Impact']
    for removing_tp in removing_disaster_type:
        df.drop(df[df['Disaster Type']==removing_tp].index, inplace = True)
    # Further truncate the time interval to exactly match the news dataset
    df['startdate'] = pd.to_datetime(df['startdate'])
    df.drop(df[df['startdate']<pd.Timestamp(2009,6,30)].index, inplace=True)
    # Remove unneeded space from the dataset
    df['Disaster Type'] = df['Disaster Type'].apply(lambda x: x.strip())
    return df

def make_news_df(news_csv_path): 
    """
    Make dataframe from a dataset folder of all television news csv files

    :param news_csv_path: The folder path that contains all news csv files
    :type news_csv_path: str
    :return news_df: The concatenated news Dataframe
    :rtype: pd.DataFrame
    """
    assert isinstance(news_csv_path,str)
    from glob import glob 
    # Read all csv files from news dataset and combine into 1 panda dataframe
    all_files = glob(news_csv_path + "/*.csv")
    df_list = []
    for filename in all_files:
        # If an empty csv file is read, skip it
        try:
            df = pd.read_csv(filename, index_col=None, header=0)
        except pd.errors.EmptyDataError :
            continue
        if len(df)>0:
            df.drop(['URL', 'Show','IAShowID','IAPreviewThumb'],axis=1,inplace=True)
        df_list.append(df)
    news_df = pd.concat(df_list, axis=0, ignore_index=True)
    # Add the date column to match the disaster dataset
    news_df['Date']=news_df['MatchDateTime'].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y %H:%M:%S").strftime("%Y-%b")) 
    # Add disaster type column to keep track if a disaster is associated with this news
    news_df['Disaster Type'] = news_df['Snippet'].apply(lambda x: disaster_mentioned(x))
    return news_df

def disaster_mentioned(snippet):
    """
    Record the disaster mentioned in a news snippet. This is used in *make_news_df()* mfunction
    to add a column that records what type of disaster is mentioned in the climate change news

    :param snippet: News snippet provided as a string
    :type snippet: str
    :return: A comma separated string of disastered mentioned in snippet
    :rtype: str
    """
    snippet = snippet.lower()
    out_list = []
    major_disaster_type = ['Drought','Earthquake','Epidemic','Extreme temperature','Flood','Landslide',
                            'Storm','Volcanic activity','Wildfire'] # Major disaster type analyzed
    disaster_types = [disaster_tp.lower().strip() for disaster_tp in major_disaster_type]
    # Add any disaster type mentioned in the news, or NaN if none
    for disaster_tp in disaster_types:
        if disaster_tp in snippet:
            out_list.append(disaster_tp)
    return ",".join(out_list) if len(out_list)>0 else "NaN"




def make_disaster_group(df, groupby='occurrence',sum_month=3):
    """
    Make a new Dataframe from the result of grouping of natural disaster type and another 
    aspect, such as occurrence or total death.

    :param df: Dataset used to perform group operation
    :type df: pd.Dataframe
    :param groupby: The feature used to group the dataset, defaults to 'occurrence'
    :type groupby: str, optional
    :param sum_month: The month interval used to group the dataset, defaults to 3
    :type sum_month: int, optional
    :return grouped: Dataframe after grouping
    :rtype: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(sum_month,int) and 1<=sum_month<=12
    assert isinstance(groupby, str) and (groupby in df.columns or groupby == 'occurrence')
    df = disaster_data_cleaning_for_news_analysis(df)
    major_disaster_type = ['Drought','Earthquake','Epidemic','Extreme temperature','Flood','Landslide',
                            'Storm','Volcanic activity','Wildfire']
    grouped = df.groupby(['startdate','Disaster Type'])
    # Group the dataset based on the given feature of the disaster type
    if groupby=='occurrence':
        grouped = grouped.size().unstack().fillna(0)
    else:
        grouped = grouped[[groupby]].apply(lambda x: x.sum()).unstack().fillna(0)
        grouped.columns=grouped.columns.droplevel() # Convert multi-index to single level
    # If a major disaster type never occurred, add a column with 0 for the type
    for disaster_tp in major_disaster_type:
        if disaster_tp not in grouped.columns:
            grouped[disaster_tp]=0.0
    grouped = grouped.reindex(sorted(grouped.columns), axis=1)
    # Sum and record for all types
    grouped["All types"] = grouped.sum(axis=1)
    # Sum over a set number of months interval for general trend
    if grouped is not None:
        grouped = grouped.resample('{:d}M'.format(sum_month), kind='period', convention='start').agg('sum')
        grouped.index = [ind.strftime('%Y-%b') for ind in grouped.index]
    return grouped 

def make_news_group(news_df,sum_month=3):
    """
    Make a new Dataframe from the result of grouping of the date of the news

    :param news_df: The news dataset 
    :type news_df: pd.DataFrame
    :param sum_month: The month interval used to group the dataset, defaults to 3
    :type sum_month: int, optional
    :return news_gpd: News dataframe grouped by date
    :rtype: pd.DataFrame
    """
    assert isinstance(news_df, pd.DataFrame)
    assert isinstance(sum_month,int) and 1<=sum_month<=12
    # Add a Date column with Datetime object as index to better store and track the date
    news_gpd = news_df.groupby('Date').size()
    news_gpd.index = pd.to_datetime(news_gpd.index)
    news_gpd.sort_index(inplace=True)
    # Sum over a set number of months interval for general trend
    if news_gpd is not None:
        news_gpd = news_gpd.resample('{:d}M'.format(sum_month), kind='period', convention='start').agg('sum')
        news_gpd.index = [ind.strftime('%Y-%b') for ind in news_gpd.index]
    return news_gpd


def comp_all_disaster_with_news(disaster_df, news_df, disaster_property="occurrence",sum_month=3,title=None):
    """
    Compare all disaster types statistics with the news statistics, the disaster feature can be chosen
    from occurrence, total deaths, or the number of affected. The time interval can be summed over a set month
    interval for a more general trend. A visualization will be shown as output

    :param disaster_df: disaster dataset dataframe
    :type disaster_df: pd.DataFrame
    :param news_df: news dataset dataframe
    :type news_df: pd.DataFrame
    :param disaster_property: disaster feature used for comparison, defaults to "occurrence"
    :type disaster_property: str, optional
    :param sum_month: Combined month unit, defaults to 3
    :type sum_month: int, optional
    :param title: Plot title, defaults to None
    :type title: str, optional
    """
    acceptable_property = ["occurrence", "Total Deaths", "Total Affected"]
    assert disaster_property in acceptable_property
    assert isinstance(disaster_df, pd.DataFrame)
    assert isinstance(news_df, pd.DataFrame)

    # Set the color scheme
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
    presentation_colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    presentation_colors = np.vstack((presentation_colors,cls.to_rgba("grey")))

    # Plot the first axis with natural disaster data
    # Generate the group instructed by the disaster feature
    disasterdata_group = make_disaster_group(disaster_df,groupby=disaster_property,sum_month=sum_month)
    # Reorder the dataframe based on the date for plotting 
    disasterdata_group.index = pd.to_datetime(disasterdata_group.index)
    disasterdata_group.sort_index(inplace=True)
    xticks = pd.date_range(disasterdata_group.index[0], disasterdata_group.index[-1], freq='3MS')
    disasterdata_group.plot(kind='line',ax=ax,xticks=xticks.to_pydatetime(), xlabel='',color=presentation_colors,marker='.')
    ax.set_ylabel("Natural disaster occurred",fontsize=14)
    plt.xticks(rotation=90);
    if title is not None:
        plt.title(title,fontweight="bold",fontsize=15)
    else:
        plt.title("Television news reported vs. {:s} of Disaster over past decade in America".format(disaster_property),
                    fontweight="bold",fontsize=15)

    # Plot the second axis with news report data
    ax2=ax.twinx()
    # Generate the news data group based on date for plotting
    news_gpd = make_news_group(news_df,sum_month=sum_month)
    news = news_gpd.iloc[:len(xticks)]
    news.index = pd.to_datetime(disasterdata_group.index)
    news.sort_index(inplace=True)
    news.plot(kind='line',  ax=ax2, xticks=xticks.to_pydatetime(),color='steelblue',marker="o",lw=3,)
    ax2.set_xticklabels([x.strftime('%Y-%b') for x in xticks])
    ax2.set_ylabel("Television news reported",fontsize=14,rotation=-90)
    ax2.yaxis.set_label_coords(1.08,0.5)

    # Set the legend clearly
    from matplotlib.lines import Line2D
    custom_lines=[Line2D([0],[0],color='steelblue',lw=3,marker='o')]
    legend_list = ["Television news reported"]
    ax2.legend(custom_lines,legend_list,fontsize=11)

def comp_disaster_with_news(disaster_df,news_df,type_name='All types',disaster_property="occurrence",sum_month=1,title=None,ax1label=None):
    """
    Compare a particular disaster type statistics with the news statistics, the disaster feature can be chosen
    from occurrence, total deaths, or the number of affected. The time interval can be summed over a set month
    interval for a more general trend. A visualization will be shown as output

    :param disaster_df: disaster dataset dataframe
    :type disaster_df: pd.DataFrame
    :param news_df: news dataset dataframe
    :type news_df: pd.DataFrame
    :param type_name: The specific disaster type to be used, defaults to "All types"
    :type type_name: str, optional
    :param disaster_property: disaster feature used for comparison, defaults to "occurrence"
    :type disaster_property: str, optional
    :param sum_month: Combined month unit, defaults to 3
    :type sum_month: int, optional
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param ax1label: Axis 1 y label, defaults to None
    :type ax1label: str, optional
    """

    acceptable_property = ["occurrence", "Total Deaths", "Total Affected"]
    assert disaster_property in acceptable_property
    assert isinstance(disaster_df, pd.DataFrame)
    assert isinstance(news_df, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)

    # Plot the first axis with natural disaster data
    # Generate the group instructed by the disaster feature
    disasterdata_group = make_disaster_group(disaster_df,groupby=disaster_property,sum_month=sum_month)
    # Reorder the dataframe based on the date for plotting 
    disasterdata_group.index = pd.to_datetime(disasterdata_group.index)
    disasterdata_group.sort_index(inplace=True)
    xticks = pd.date_range(disasterdata_group.index[0], disasterdata_group.index[-1], freq='{:d}MS'.format(sum_month))
    disasterdata_group.plot(kind='line', y=type_name, ax=ax,xticks=xticks.to_pydatetime(), xlabel='',color='olivedrab',marker='.')
    if title is not None:
        plt.title(title,fontweight="bold",fontsize=15)
    else:
        plt.title("Television news reported vs. {:s} of {:s} over past decade in \
                    America".format(disaster_property,type_name), fontweight="bold",fontsize=15)
   
    plt.xticks(rotation=90);
    if title is not None:
        plt.title(title,fontweight="bold",fontsize=15)
    else:
        plt.title("Television news reported vs. {:s} of Disaster over past decade in America".format(disaster_property),
                    fontweight="bold",fontsize=15)

    # Plot the second axis with news report data
    ax2=ax.twinx()
    # Generate the news data group based on date for plotting
    news_gpd = make_news_group(news_df,sum_month=sum_month)
    news = news_gpd.iloc[:len(xticks)]
    news.index = pd.to_datetime(disasterdata_group.index)
    news.sort_index(inplace=True)
    news.plot(kind='line',  ax=ax2, xticks=xticks.to_pydatetime(),color='steelblue',marker=".",)
    ax2.set_xticklabels([x.strftime('%Y-%b') for x in xticks],fontsize=4)
    # make a plot with different y-axis using second axis object
    ax2.set_ylabel("Television news reported",fontsize=14,rotation=-90)
    ax2.yaxis.set_label_coords(1.08,0.5)

    # Set the legend clearly
    from matplotlib.lines import Line2D
    custom_lines=[Line2D([0],[0],color='olivedrab',lw=2),Line2D([0],[0],color='steelblue',lw=2)]
    legend_list = [type_name,"Television news reported"]
    if ax1label != None:
        legend_list[0] = ax1label
    ax2.legend(custom_lines,legend_list,fontsize=14)

def news_mention_count(news_df,col_name="Disaster Type",out_col_name="mentions"):
    """
    Count the total number of disasters by each disaster type mentioned by climate change
    television news 

    :param news_df: News dataset dataframe
    :type news_df: pd.DataFrame
    :param col_name: The used column name to count, defaults to "Disaster Type"
    :type col_name: str, optional
    :param out_col_name: Output column name, defaults to "mentions"
    :type out_col_name: str, optional
    :return: A dataframe of the total number of disastered mentioned by news
    :rtype: pd.DataFrame
    """
    from collections import defaultdict
    assert isinstance(news_df, pd.DataFrame)
    count_dict = defaultdict(int)
    # For each row, parse the disaster type and add to the count in dictionary
    for row in news_df[col_name]:
        disaster_tps = row.split(",")
        if len(disaster_tps)==1 and disaster_tps[0]=="NaN":
            continue
        for item in disaster_tps:
            count_dict[item.strip()]+=1
    # Convert the count dictionary to a pandas dataframe
    news_mention_count = pd.DataFrame.from_dict(count_dict, orient='index', columns=[out_col_name]).sort_values(by=[out_col_name])
    # Capitalize the index (disaster type) to match the disaster database
    new_index = [ind.capitalize() for ind in news_mention_count.index]
    news_mention_count.index = new_index
    return news_mention_count

def news_disaster_ratio(disaster_df,news_df,disaster_property="occurrence"):
    """
    Calculates and visualizes the ratio between number of total news mentioned on the topic of a specific disaster type
    and a feature of that specific disaster type, such as occurrence, or total deaths, etc. A visualization will be 
    shown as output.

    :param disaster_df: Disaster dataset dataframe
    :type disaster_df: pd.DataFrame
    :param news_df: news dataset dataframe
    :type news_df: pd.DataFrame
    :param disaster_property: Feature of disaster, defaults to "occurrence"
    :type disaster_property: str, optional
    """

    # Generate the group instructed by the disaster feature
    disaster_grp = make_disaster_group(disaster_df,groupby=disaster_property)
    grp_count = disaster_grp.sum(axis=0).drop("All types")
    grp_count.name = "count"

    # Remove rare disasters with little impacts 
    removing_disaster_type = ['Mass movement (dry)','Insect infestation', 'Impact']
    try:
        grp_count.drop(removing_disaster_type,inplace=True)
    except:
        pass

    # Concatenat the disaster group and news mention count dataframe to 1 for easier operation
    concatenated = pd.concat([grp_count,news_mention_count(news_df)],axis=1)
    # Calculates the ratio between number of total news mentioned on the topic of a specific 
    # disaster type and a feature of that specific disaster type
    new_col_name = "news mentions per disaster {:s}".format(disaster_property)
    concatenated[new_col_name] = concatenated["mentions"] / concatenated["count"]
    # Special handling of edge cases with NaN and Infinite value
    concatenated.replace([np.inf, -np.inf,"inf"], np.nan,inplace=True)
    concatenated.fillna(0,inplace=True)

    # visualize the result on bar chart with numbers labeled and sorted values
    fig,ax=plt.subplots()
    concatenated[new_col_name].sort_values().plot(kind='barh', ax=ax,color = 'steelblue')
    for i, v in enumerate(concatenated[new_col_name].sort_values()):
        ax.text(v+1, i-0.1, "{:d}".format(round(v)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title("Number of {:s}".format(new_col_name),fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=10)


