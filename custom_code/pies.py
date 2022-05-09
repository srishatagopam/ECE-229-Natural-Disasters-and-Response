#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls

def occurence_pie(data):
    '''
    :parameter data dataframe
    :graph disasters appearance and their percentage relative to other disaters
    :use dictionary to count each disaster appeearance 
    :and plot it keys and values
    '''
    assert isinstance(data, pd.DataFrame)
    #declare an empty dict and count the number of time each disaster appear
    x={}
    for i in data['Disaster Type']:
        try:
            x[i]+=1
        except:
            x[i]=1
    
    explode = (0.1, 0,0,0,0,0,0) #explode is used to make the highest percentage on pie chart to stand out
    tot= sum(x.values())#sum of all the value in dict
    #n is an empty dict
    #if values of the disaster is less than 4.6% of the total sum of all the value
    #add it to other in the new dict
    
    n={}
    for i in x:
        if x[i]/tot > .046:
            n[i]=x[i]
        else:
            try:
                n['other']+=x[i]
            except:
                n['other']=x[i]
    #sort the dict, and hard custom_code the color for consistency in coloring for presentation purpose
    n = dict(sorted(n.items(), key=lambda item: item[1], reverse=True))
    presentation_colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    presentation_colors = np.vstack((presentation_colors,cls.to_rgba("grey")))
    colorsfixed = []
    colorsfixed.append(presentation_colors[4])
    colorsfixed.append(presentation_colors[6])
    colorsfixed.append(presentation_colors[2])
    colorsfixed.append(presentation_colors[9])
    colorsfixed.append(presentation_colors[1])
    colorsfixed.append(presentation_colors[5])
    colorsfixed.append(presentation_colors[0])
    fig1, ax1 = plt.subplots()
    ax1.pie(n.values(), labels= n.keys(), autopct='%1.f%%', explode = explode,
        shadow=True, startangle=0, colors = colorsfixed, textprops={'fontsize': 14})
    fig1.set_size_inches(5, 5)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Total disasters',fontdict = {'fontsize' : 16})
    plt.show()


# In[1]:


def string_pie(data,string):
    assert isinstance(data, pd.DataFrame)
    assert isinstance(string, str)
    assert len(string) > 3
    assert string == 'Total Deaths' or string=='Total Damages (\'000 US$)'

    data[string]=data[string].fillna(0)
    a=zip(data['Disaster Type'])
    b=zip(data[string])
    disaster_list=list(a)
    string_list=list(b)
    l={}
    #declare an empty dict and store the total death/total damage for each disaster
    #l will give a list with key = disaster_type and values= list of floats
    for i in range(len(disaster_list)):
        
        if disaster_list[i] in l:
            l[disaster_list[i]]+=string_list[i]
            
        else:
            l[disaster_list[i]]=string_list[i]
    
    #x a new dict
    # for each type in l dict, sum all values up for each disaster type and put it in new dict
    # x will gives key = disaster and value =  a float (sum of all the value in l for each key)
    #treshold for total deaths is .06 and treshold for damage is .04. 
    x={}
    tresh= .04
    if string=='Total Deaths':
        tresh=.06
    for i in l:
        x[i]=sum(l[i])
    tot= sum(x.values())
    n={}
    
    #n is an empty dict
    #if values of the disaster is less than 4.6% of the total sum of all the value
    #add it to other in the new dict
    for i in x:
        if x[i]/tot > tresh:
            n[i]=x[i]
        else:
            try:
                n['other']+=x[i]
            except:
                n['other']=x[i]
    #sort the dict, and hard custom_code the color for consistency in coloring for presentation purpose

    n = dict(sorted(n.items(), key=lambda item: item[1], reverse=True))
    explode =[0.1] + [0] * (len(n)-1)
    fig1, ax1 = plt.subplots()
    presentation_colors = plt.cm.Spectral(np.linspace(0, 1, 9))[::-1]
    presentation_colors = np.vstack((presentation_colors,cls.to_rgba("grey")))
    
    colorsfixed = []
    if (string == 'Total Damages (\'000 US$)'):
        colorsfixed.append(presentation_colors[6])
        colorsfixed.append(presentation_colors[4])
        colorsfixed.append(presentation_colors[1])
        colorsfixed.append(presentation_colors[9])
        colorsfixed.append(presentation_colors[0])
    else: 
        colorsfixed.append(presentation_colors[1])
        colorsfixed.append(presentation_colors[6])
        colorsfixed.append(presentation_colors[0])
        colorsfixed.append(presentation_colors[4])
        colorsfixed.append(presentation_colors[9])
        colorsfixed.append(presentation_colors[2])
    labels = [str(each).strip("(").strip(")").strip(",").strip("'") for each in n.keys()]
    ax1.pie(n.values(), labels= labels, explode=explode, autopct='%1.f%%',
        shadow=True, startangle=0, colors = colorsfixed, textprops={'fontsize': 14})
    ax1.axis('equal')
    fig1.set_size_inches(5, 5)
    plt.title('Which disaster causes the ' + string, fontdict = {'fontsize' : 16})
    plt.show()


# In[ ]:





# In[ ]:




