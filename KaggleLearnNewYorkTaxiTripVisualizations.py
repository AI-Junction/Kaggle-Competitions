# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 10:07:37 2017

@author: echtpar
"""

#%%

import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
#import simplejson    #getting JSON in simplified format
import urllib        #for url stuff
#import gmaps       #for using google maps to visulalize places on maps
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os  # for os commands
from scipy.misc import imread, imresize, imsave  # for plots 
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_notebook, show
from IPython.display import HTML
from matplotlib.pyplot import *
from matplotlib import cm
from matplotlib import animation
import io
import base64
output_notebook()
plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



#%%

def getuniquecounts(df, listofcols):
    uniquecount = {}
    for col in listofcols:
        #print(col, " - unique counts", len(df[col].value_counts()), "\t\t", len(pd.unique(df[col])))
        uniquecount[col]= len(pd.unique(df[col]))
    return (uniquecount)
        
def getcountofNaNs (df):
    dictdtype = {}
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ['colname','type']
    list_dtype = np.array(pd.unique(dtypes.type))
    #print(list_dtype[0].name)
    for dt in list_dtype:
        #print(dt.name)
        df_dtype = pd.DataFrame()
        df_dtype = df.loc[:,list(dtypes.type == dt)].isnull().sum().reset_index()
        dictdtype[dt.name] = df_dtype
    return (dictdtype)
    


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def pipeliner(df, attribs_toImpute, attribs_toBinarize, attribs_toEncode):
    piped_dict = {}
    pipe_impute = Pipeline([
            ('selector',DataFrameSelector(attribs_toImpute)),
            ('imputer',Imputer(strategy = 'mean')),
            #('std_scaler',StandardScaler())
                ])
    
    df_prepared = pipe_impute.fit_transform(df)
    df_prepared = pd.DataFrame(df_prepared, columns=attribs_toImpute)
    piped_dict['Imputed'] = df_prepared
    
    for label in attribs_toBinarize:
        df_binarized = pd.DataFrame()
        pipe_label_binarizer = Pipeline([
                ('selector',DataFrameSelector(label)),
                ('LabelBinarizer',LabelBinarizer()),
                #('std_scaler',StandardScaler())
                    ])
    
        df_binarized = pipe_label_binarizer.fit_transform(df)
        df_binarized = pd.DataFrame(df_binarized)
        piped_dict[label+"_Binarized"] = df_binarized


    for label in attribs_toEncode:
        df_encoded = pd.DataFrame()
        '''
        pipe_label_encoder = Pipeline([
                ('selector',DataFrameSelector(label)),
                ('labelencoder',LabelEncoder()),
                #('std_scaler',StandardScaler())
                    ])
    
        df_encoded = pipe_label_encoder.fit(df)
        '''
        lblEnc = LabelEncoder()
        lblEnc.fit(df[label].astype(str))
        df_encoded = lblEnc.transform(df[label].astype(str))    
        #print(range(df_encoded.shape[0]))
        #print(label)
        #print([label])
        
        #print(df_encoded.shape)
        df_encoded = pd.DataFrame(df_encoded, columns = [label], index = range(df_encoded.shape[0]))
        piped_dict[label+"_Encoded"] = df_encoded
        
        
    return(piped_dict)
    
def dualvariateplots(df, attrib_pairs, typeofplot):
    start = time.time
    noofattribs = len(attrib_pairs)
    plottypes = ['boxplot', 'violinplot', 'stripplot', 'swarmplot']
    noofplots = len(plottypes)
    print('noofattribs = ', noofattribs)
    #rows = noofattribs/3
    fig = plt.figure(figsize=(10, noofattribs*5))
    i = 0
    for attrib_pair in attrib_pairs:
        for plottype in typeofplot:
            if plottype == 'boxplot':
                i += 1 
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.boxplot(x=attrib_pair[0], y=attrib_pair[1], data=df)
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'violinplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.violinplot(x=attrib_pair[0], y=attrib_pair[1], data=df)
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'stripplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.stripplot(x=attrib_pair[0], y=attrib_pair[1], data=df)
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'swarmplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.swarmplot(x=attrib_pair[0], y=attrib_pair[1], data=df, color=".25")                
    end = time.time()
    print("Time taken by univariateplots is {}.".format((end-start)))
    #return

                
def univariateplots(df, attribs, typeofplot):
    start = time.time
    noofattribs = len(attribs)
    plottypes = ['boxplot', 'violinplot', 'stripplot', 'histogram']
    noofplots = len(plottypes)
    print('noofattribs = ', noofattribs)
    #rows = noofattribs/3
    fig = plt.figure(figsize=(10, noofattribs*3))
    i = 0
    for attrib in attribs:
        for plottype in plottypes:
            if plottype == 'boxplot':
                i += 1 
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.boxplot(df[attrib])
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'violinplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.violinplot(df[attrib])
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'stripplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.stripplot(df[attrib])
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
            if plottype == 'histogram':
                i += 1
                #ax.xlabel(attrib, fontsize=12)
                #ax.ylabel('count', fontsize=12)
                max = np.max(df[attrib])
                min = np.min(df[attrib])  
                bins = 50
                ax = fig.add_subplot(noofattribs,noofplots,i)
                ax = plt.hist(df[attrib].ravel(), bins = bins, range = [min, max])
            '''
            if plottype == 'distplot':
                i += 1
                ax = fig.add_subplot(noofattribs,noofplots,i)
                g = sns.distplot(df[attrib], kde=False, rug=True)
            '''        
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    end = time.time()
    print("Time taken by univariateplots is {}.".format((end-start)))
    plt.show()
    #return
    
    

def jointplots(df, attribpairs, plottypes):
    start = time.time
    noofattribpairs = len(attribpairs)
    #plotkinds = ['none','reg','hex','kde','kde_special']
    #noofplots = len(plotkinds)
    print('noofattribs = ', noofattribpairs)
    #sns.set(style="white", color_codes=True)
    #rows = noofattribs/3
    fig = plt.figure(figsize=(10, noofattribpairs*3))
    i = 0
    for attribpair in attribpairs:
        for plottype in plottypes:
            if plottype == 'none':
                i += 1 
                #ax = fig.add_subplot(noofattribpairs,noofplots,i)
                g = sns.jointplot(x=attribpair[0], y=attribpair[1], data=df)
            if plottype == 'reg':
                i += 1 
                #ax = fig.add_subplot(noofattribpairs,noofplots,i)
                g = sns.jointplot(x=attribpair[0], y=attribpair[1], data=df, kind="reg")
            if plottype == 'hex':
                i += 1 
                #ax = fig.add_subplot(noofattribpairs,noofplots,i)
                g = sns.jointplot(x=attribpair[0], y=attribpair[1], data=df, kind="hex")
            if plottype == 'kde':
                i += 1 
                #ax = fig.add_subplot(noofattribpairs,noofplots,i)
                g = sns.jointplot(x=attribpair[0], y=attribpair[1], data=df, kind="kde", space=0, color="g")
            if plottype == 'kde_special':
                i += 1 
                #ax = fig.add_subplot(noofattribpairs,noofplots,i)
                g = (sns.jointplot(attribpair[0], attribpair[1],
                    data=df, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))  
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    end = time.time()
    print("Time taken by jointplots is {}.".format((end-start)))
    plt.show()
    #return                
'''                
def jointplots(df, col, row, plotattrib):
    #kws = dict(s=50, linewidth=.5, edgecolor="w")
    plt.figure(figsize=(8,8))
    g = sns.jointplot(df[row], df[col], size=10)
    plt.xlabel(row, fontsize=12)
    plt.ylabel(col, fontsize=12)

    
    g = (g.map(plt.scatter, plotattrib, plotattrib, color="r", **kws)
      .set(xlim=(0, 60), ylim=(0, 12),
           xticks=[10, 30, 50], yticks=[2, 6, 10]))
    
    
'''    


def sortcolumns(df, columnstosort):
    start = time.time()
    sorted_columns_dict = {}
    fig = plt.figure(figsize = (10,len(columnstosort)*5))
#    fig = plt.figure(figsize = (10,16))
#    #fig, axes = plt.subplots(len(columnstosort), 1)
    i = 0
    for col in columnstosort:
        i += 1
        sorted_columns_dict[col] = df[col].sort_values(ascending = True)
        ax = fig.add_subplot(len(columnstosort),1,i)
        ax.scatter(range(len(sorted_columns_dict[col])), sorted_columns_dict[col], alpha = 0.25)
        ax.set_title(col)
    end = time.time()
    print("Time taken by sortcolumns is {}.".format((end-start)))
    plt.show()
    return (sorted_columns_dict)        


def remove_outliers(df, colnametotrim, upper_percentile, lower_percentile):
    ulimit = np.percentile(df[colnametotrim],upper_percentile)
    llimit = np.percentile(df[colnametotrim],lower_percentile)
    
    df.loc[df[colnametotrim] > ulimit, colnametotrim] = ulimit
    df.loc[df[colnametotrim] < llimit, colnametotrim] = llimit

    return (df)        
    

    
def df_get_coltypes(df):
    df_col_types = df.dtypes.reset_index()
    df_col_types.columns = ['colname','type']
    #df_col_types.loc[df_col_types.type == 'object']
    df_return1 = df_col_types.type.value_counts().reset_index()
    df_return2 = df_col_types.groupby(['type']).aggregate('count').reset_index()
    df_return1.columns = ['type','counts']
    df_return2.columns = ['type','counts']
    return (df_return1, df_return2)        

#%%

s = time.time()
#train_fr_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
#train_fr_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

train_fr_1 = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "fastest_routes_train_part_1.csv"))
train_fr_2 = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "fastest_routes_train_part_2.csv"))

train_fr = pd.concat([train_fr_1, train_fr_2], axis=0)
print(train_fr_1.shape, train_fr_2.shape, train_fr.shape)

train_fr.columns

#train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train_fr_set = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train_df = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "train.csv"))

train_fr_set[:10]

train = pd.merge(train_df, train_fr_set, on = 'id', how = 'left')
train_df = train.copy()

end = time.time()
print("Time taken by above cell is {}.".format((end-s)))
train_df.columns

#%%

df_train_coltypes1, df_train_coltypes2 = df_get_coltypes(train_df)
df_train_coltypes1
df_train_coltypes2 

uniquecounts = getuniquecounts(train_df, ['dropoff_longitude', 'dropoff_latitude'])
print(uniquecounts)


tmp = train_df['total_distance'].isnull().sum()
print(tmp)
train_df['total_distance'].fillna(0, inplace=True)

tmp = train_df['number_of_steps'].isnull().sum()
print(tmp)
train_df['number_of_steps'].fillna(0, inplace=True)



sorted_columns_dict = sortcolumns(train_df, ['total_distance', 'number_of_steps'])

type(sorted_columns_dict['number_of_steps'][0].astype(int))


fig = plt.figure(figsize = (10,10))
plt.scatter(train_df.dropoff_longitude, train_df.dropoff_latitude, alpha = .25)
plt.show()

train_data = train_df.copy()



#%%

start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['trip_duration'].values+1), axlabel = 'Log(trip_duration)', label = 'log(trip_duration)', bins = 50, color="r")
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()

#%%

start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(train_df['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(train_df['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])
sns.distplot(train_df['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(train_df['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


#%%

df = train_df.loc[(train_df.pickup_latitude > 40.6) & (train_df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude>40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
train_data_new = df.copy()

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(12, 12), sharex=False, sharey = False)#
sns.despine(left=False)
sns.distplot(train_data_new['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(train_data_new['pickup_longitude'].values, label = 'pickup_longitude',color="g",bins =100, ax=axes[0,1])
sns.distplot(train_data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(train_data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="g",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
print(df.shape[0], train_data.shape[0])
plt.show()

#%%

start = time.time()
temp = train_data.copy()
train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:, 'pick_date'] = train_data['pickup_datetime'].dt.date
train_data.head()

ts_v1 = pd.DataFrame(train_data.loc[train_data['vendor_id']==1].groupby('pick_date')['trip_duration'].mean())
ts_v1.reset_index(inplace = True)
ts_v2 = pd.DataFrame(train_data.loc[train_data.vendor_id==2].groupby('pick_date')['trip_duration'].mean())
ts_v2.reset_index(inplace = True)

#from bokeh.palettes import Spectral4
from bokeh.plotting import output_notebook
#from bokeh.sampledata.stocks import AAPL, IBM, MSFT, GOOG
output_notebook()
p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
p.title.text = 'Click on legend entries to hide the corresponding lines'


for data, name, color in zip([ts_v1, ts_v2], ["vendor 1", "vendor 2"], Spectral4):
    df = data
    p.line(df['pick_date'], df['trip_duration'], line_width=2, color=color, alpha=0.8, legend=name)

p.legend.location = "top_left"
#p.legend.click_policy="hide"
show(p)
end = time.time()
train_data = temp
print("Time Taken by above cell is {}.".format(end - start))



#%%
start = time.time()
rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)
rgb[..., 0] = 0
rgb[..., 1] = 0
rgb[..., 2] = 0

train_data_new['pickup_latitude'][:10]

train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))
train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))
train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))
train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))

summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())

summary_plot.reset_index(inplace = True)
summary_plot.head(120)
summary_plot.columns
lat_list = summary_plot['pick_lat_new'].unique()


for i in lat_list:
    lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()
    unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()
    for j in lon_list:
        a = unit[lon_list.index(j)]
        print('j = ', j , 'unit[lon_list.index(j)] = ', unit[lon_list.index(j)])                 
        if (a//50) >0:
            rgb[i][j][0] = 255
            rgb[i,j, 1] = 0
            rgb[i,j, 2] = 255
        elif (a//10)>0:
            rgb[i,j, 0] = 0
            rgb[i,j, 1] = 255
            rgb[i,j, 2] = 0
        else:
            rgb[i,j, 0] = 255
            rgb[i,j, 1] = 0
            rgb[i,j, 2] = 0
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
ax.imshow(rgb, cmap = 'hot')
ax.set_axis_off() 

#%%

start = time.time()
def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)

def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b

import math
def bearing_array(lat1, lng1, lat2, lng2):
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

end = time.time()
print("Time taken by above cell is {}.".format((end-start)))

#%%

start = time.time()
train_data = temp.copy()
train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month
train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour
train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear
train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear
train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek
train_data.loc[:,'hvsine_pick_drop'] = haversine_(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)
train_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)
train_data.loc[:,'bearing'] = bearing_array(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)

end = time.time()
print("Time taken by above cell is {}.".format(end-start))

#%%

start = time.time()
def color(hour):
    """function for color change in animation"""
    return(10*hour)

def Animation(hour, temp, rgb):
    """Function to generate return a pic of plotings"""
    #ax.clear()
    train_data_new = temp.loc[temp['hour'] == hour]
    start = time.time()
    rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)
    rgb[..., 0] = 0
    rgb[..., 1] = 0
    rgb[..., 2] = 0
    train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))
    train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))
    train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))
    train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))

    summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())

    summary_plot.reset_index(inplace = True)
    summary_plot.head(120)
    lat_list = summary_plot['pick_lat_new'].unique()
    for i in lat_list:
        #print(i)
        lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()
        unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()
        for j in lon_list:
            #j = int(j)
            a = unit[lon_list.index(j)]
            #print(a)
            if (a//50) >0:
                rgb[i][j][0] = 255 - color(hour)
                rgb[i,j, 1] = 255 - color(hour)
                rgb[i,j, 2] = 0 + color(hour)
            elif (a//10)>0:
                rgb[i,j, 0] = 0 + color(hour)
                rgb[i,j, 1] = 255 - color(hour)
                rgb[i,j, 2] = 0 + color(hour)
            else:
                rgb[i,j, 0] = 255 - color(hour)
                rgb[i,j, 1] = 0 + color(hour)
                rgb[i,j, 2] = 0 + color(hour)
    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))
    end = time.time()
    print("Time taken by above cell is {} for {}.".format((end-start), hour))
    return(rgb)
end = time.time()
print("Time taken by above cell is {}.".format(end -start))

#%%

start = time.time()
images_list=[]
train_data_new['pickup_datetime'] = pd.to_datetime(train_data_new.pickup_datetime)
train_data_new.loc[:, 'hour'] = train_data_new['pickup_datetime'].dt.hour

for i in list(range(0, 24)):
    im = Animation(i, train_data_new, rgb.copy())
    images_list.append(im)
end = time.time()
print("Time taken by above cell is {}.".format(end -start))

#%%

import matplotlib.animation as animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Reinforcement Learning\\Patacchiola\\src\\MountainCar\\ffmpeg.exe'
plt.rcParams['animation.avconv_path'] = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Reinforcement Learning\\Patacchiola\\src\\MountainCar\\libav-0.8.17-win64\\win64\\usr\\bin\\avconv.exe'
#plt.rcParams['animation.convert_path'] = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Reinforcement Learning\\Patacchiola\\src\\MountainCar\\PythonMagick-0.9.17\\PythonMagick-0.9.17\\PythonMagick'
plt.rcParams['animation.convert_path'] = 'C:\\Program Files\\ImageMagick-7.0.7-Q16\\magick.exe'


start = time.time()
def build_gif(imgs = images_list, show_gif=True, save_gif=True, title=''):
    """function to create a gif of heatmaps"""
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    ax.set_axis_off()
    hr_range = list(range(0,24))
    def show_im(pairs):
        ax.clear()
        ax.set_title('Absolute Traffic - Hour ' + str(int(pairs[0])) + ':00')
        ax.imshow(pairs[1])
        ax.set_axis_off() 
    pairs = list(zip(hr_range, imgs))
    #ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)
    im_ani = animation.FuncAnimation(fig, show_im, pairs,interval=100, repeat_delay=0, blit=False)
    plt.cla()
    if save_gif:
        #im_ani.save('animation.gif', writer='imagemagick') #, writer='imagemagick'
        #im_ani.save("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial", fps=1, writer='avconv', codec='libx264')
        im_ani.save("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Animation.mp4", fps = 20, writer='avconv', codec='libx264')
        #im_ani.save('animation.gif', writer='ffmpeg') #, writer='imagemagick'
    if show_gif:
        plt.show()
    return
end = time.time()
print("Time taken by above cell is {}".format(end-start))


#%%

#    def render(self, file_path='C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Reinforcement Learning\\Patacchiola\\src\\MountainCar\\mountain_car.mp4', mode='mp4'):
#        """ When the method is called it saves an animation
#        of what happened until that point in the episode.
#        Ideally it should be called at the end of the episode,
#        and every k episodes.
#        
#        ATTENTION: It requires avconv and/or imagemagick installed.
#        @param file_path: the name and path of the video file
#        @param mode: the file can be saved as 'gif' or 'mp4'
#        """
#        # Plot init
#        fig = plt.figure()
#        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
#        ax.grid(False)  # disable the grid
#        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
#        y_sin = np.sin(3 * x_sin)
#        # plt.plot(x, y)
#        ax.plot(x_sin, y_sin)  # plot the sine wave
#        # line, _ = ax.plot(x, y, 'o-', lw=2)
#        dot, = ax.plot([], [], 'ro')
#        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#        _position_list = self.position_list
#        _delta_t = self.delta_t
#
#        def _init():
#            dot.set_data([], [])
#            time_text.set_text('')
#            return dot, time_text
#
#        def _animate(i):
#            x = _position_list[i]
#            y = np.sin(3 * x)
#            dot.set_data(x, y)
#            time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
#            return dot, time_text
#
#        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.position_list)),
#                                      blit=True, init_func=_init, repeat=False)
#
#        if mode == 'gif':
#            print('in ani save as gif')
#            ani.save(file_path, writer='imagemagick', fps=int(1/self.delta_t))
#            
#        elif mode == 'mp4':
#            ani.save(file_path, fps=int(1/self.delta_t), writer='avconv', codec='libx264')
#        # Clear the figure
#        fig.clear()
#        plt.close(fig)
#
#



#%%

start = time.time()
build_gif()
end = time.time()
print(end-start)
#%%

filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

#%%

start = time.time()
summary_wdays_avg_duration = pd.DataFrame(train_data.groupby(['vendor_id','day_of_week'])['trip_duration'].mean())
print(type(summary_wdays_avg_duration))
print(summary_wdays_avg_duration.head())
print(summary_wdays_avg_duration.index)



summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1

fig = plt.figure()
sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("poster")
sns.tsplot(data=summary_wdays_avg_duration, time="day_of_week", unit = "unit", condition="vendor_id", value="trip_duration")
sns.despine(bottom = False)
plt.show()
end = time.time()
print(end - start)

#%%

import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.set_context("poster")
train_data2 = train_data.copy()
train_data2['trip_duration']= np.log(train_data['trip_duration'])
fig = plt.figure()
sns.violinplot(x="passenger_count", y="trip_duration", hue="vendor_id", data=train_data2, split=True,
               inner="quart",palette={1: "g", 2: "r"})

sns.despine(left=True)
plt.show()
print(df.shape[0])

#%%

start = time.time()
fig = plt.figure()
sns.set(style="ticks")
sns.set_context("poster")
sns.boxplot(x="day_of_week", y="trip_duration", hue="vendor_id", data=train_data, palette="PRGn")
plt.ylim(0, 6000)
sns.despine(offset=10, trim=True)
plt.show()
print(train_data.trip_duration.max())
end = time.time()
print("Time taken by above cell is {}.".format(end-start))

#%%

summary_hour_duration = pd.DataFrame(train_data.groupby(['day_of_week','hour'])['trip_duration'].mean())
summary_hour_duration.reset_index(inplace = True)
summary_hour_duration['unit']=1
fig = plt.figure()
sns.set(style="white", palette="muted", color_codes=False)
sns.set_context("poster")
sns.tsplot(data=summary_hour_duration, time="hour", unit = "unit", condition="day_of_week", value="trip_duration")
sns.despine(bottom = False)
plt.show()


#%%

start = time.time()
def assign_cluster(df, k):
    """function to assign clusters """
    df_pick = df[['pickup_longitude','pickup_latitude']]
    df_drop = df[['dropoff_longitude','dropoff_latitude']]
    """I am using initialization as from the output of
    k-means from my local machine to save time in this kernel"""
    init = np.array([[ -73.98737616,   40.72981533],
       [-121.93328857,   37.38933945],
       [ -73.78423222,   40.64711269],
       [ -73.9546417 ,   40.77377538],
       [ -66.84140269,   36.64537175],
       [ -73.87040541,   40.77016484],
       [ -73.97316185,   40.75814346],
       [ -73.98861094,   40.7527791 ],
       [ -72.80966949,   51.88108444],
       [ -76.99779701,   38.47370625],
       [ -73.96975298,   40.69089596],
       [ -74.00816622,   40.71414939],
       [ -66.97216034,   44.37194443],
       [ -61.33552933,   37.85105133],
       [ -73.98001393,   40.7783577 ],
       [ -72.00626526,   43.20296402],
       [ -73.07618713,   35.03469086],
       [ -73.95759366,   40.80316361],
       [ -79.20167796,   41.04752096],
       [ -74.00106031,   40.73867723]])
    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    df['label_pick'] = clust_pick.tolist()
    df['label_drop'] = k_means_pick.predict(df_drop)
    return df, k_means_pick

    
end = time.time()
print("time taken by thie script by now is {}.".format(end-start))


#%%

start = time.time()
train_cl, k_means = assign_cluster(train_data, 20)  # make it 100 when extracting features 
print(k_means)
print(k_means.cluster_centers_)
centroid_pickups = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_pick_long', 'centroid_pick_lat'])
print(centroid_pickups)

centroid_dropoff = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_drop_long', 'centroid_drop_lat'])
print(centroid_dropoff)



centroid_pickups['label_pick'] = centroid_pickups.index
centroid_dropoff['label_drop'] = centroid_dropoff.index
#centroid_pickups.head()
train_cl = pd.merge(train_cl, centroid_pickups, how='left', on=['label_pick'])
train_cl = pd.merge(train_cl, centroid_dropoff, how='left', on=['label_drop'])
train_cl.head()
end = time.time()
print("Time taken in clustering is {}.".format(end - start))

#%%

start = time.time()
train_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)
train_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)
train_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)
train_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)
train_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)
train_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)
train_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)
train_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)
train_cl['speed_hvsn'] = train_cl.hvsine_pick_drop/train_cl.total_travel_time
train_cl['speed_manhtn'] = train_cl.manhtn_pick_drop/train_cl.total_travel_time
end = time.time()
print("Time Taken by above cell is {}.".format(end-start))
train_cl.head()

#%%

start = time.time()
def cluster_summary(sum_df):
    """function to calculate summary of given list of clusters """
    #agg_func = {'trip_duration':'mean','label_drop':'count','bearing':'mean','id':'count'} # that's how you use agg function with groupby
    summary_avg_time = pd.DataFrame(sum_df.groupby('label_pick')['trip_duration'].mean())
    summary_avg_time.reset_index(inplace = True)
    summary_pref_clus = pd.DataFrame(sum_df.groupby(['label_pick', 'label_drop'])['id'].count())
    summary_pref_clus = summary_pref_clus.reset_index()
    summary_pref_clus = summary_pref_clus.loc[summary_pref_clus.groupby('label_pick')['id'].idxmax()]
    summary =pd.merge(summary_avg_time, summary_pref_clus, how = 'left', on = 'label_pick')
    summary = summary.rename(columns={'trip_duration':'avg_triptime'})
    return summary
end = time.time()
print("Time Taken by above cell is {}.".format(end-start))

#%%

import folium

def show_fmaps(train_data, path=1):
    """function to generate map and add the pick up and drop coordinates
    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line
    """
    full_data = train_data
    summary_full_data = pd.DataFrame(full_data.groupby('label_pick')['id'].count())
    summary_full_data.reset_index(inplace = True)
    summary_full_data = summary_full_data.loc[summary_full_data['id']>70000]
    map_1 = folium.Map(location=[40.767937, -73.982155], zoom_start=10,tiles='Stamen Toner') # manually added centre
    new_df = train_data.loc[train_data['label_pick'].isin(summary_full_data.label_pick.tolist())].sample(50)
    new_df.reset_index(inplace = True, drop = True)
    for i in range(new_df.shape[0]):
        pick_long = new_df.loc[new_df.index ==i]['pickup_longitude'].values[0]
        pick_lat = new_df.loc[new_df.index ==i]['pickup_latitude'].values[0]
        dest_long = new_df.loc[new_df.index ==i]['dropoff_longitude'].values[0]
        dest_lat = new_df.loc[new_df.index ==i]['dropoff_latitude'].values[0]
        folium.Marker([pick_lat, pick_long]).add_to(map_1)
        folium.Marker([dest_lat, dest_long]).add_to(map_1)
    return map_1
    
#%%    

def clusters_map(clus_data, full_data, tile = 'OpenStreetMap', sig = 0, zoom = 12, circle = 0, radius_ = 30):
    """ function to plot clusters on map"""
    map_1 = folium.Map(location=[40.767937, -73.982155], zoom_start=zoom,tiles= tile) # 'Mapbox' 'Stamen Toner'
    summary_full_data = pd.DataFrame(full_data.groupby('label_pick')['id'].count())
    summary_full_data.reset_index(inplace = True)
    if sig == 1:
        summary_full_data = summary_full_data.loc[summary_full_data['id']>70000]
    sig_cluster = summary_full_data['label_pick'].tolist()
    clus_summary = cluster_summary(full_data)
    for i in sig_cluster:
        pick_long = clus_data.loc[clus_data.index ==i]['centroid_pick_long'].values[0]
        pick_lat = clus_data.loc[clus_data.index ==i]['centroid_pick_lat'].values[0]
        clus_no = clus_data.loc[clus_data.index ==i]['label_pick'].values[0]
        most_visited_clus = clus_summary.loc[clus_summary['label_pick']==i]['label_drop'].values[0]
        avg_triptime = clus_summary.loc[clus_summary['label_pick']==i]['avg_triptime'].values[0]
        pop = 'cluster = '+str(clus_no)+' & most visited cluster = ' +str(most_visited_clus) +' & avg triptime from this cluster =' + str(avg_triptime)
        if circle == 1:
            folium.CircleMarker(location=[pick_lat, pick_long], radius=radius_,
                    color='#F08080',
                    fill_color='#3186cc', popup=pop).add_to(map_1)
        folium.Marker([pick_lat, pick_long], popup=pop).add_to(map_1)
    return map_1

#%%

kanton_map = folium.Map(location=[46.8, 8.33],
                   tiles='Mapbox Bright', zoom_start=7)
#kanton_map

kanton_map.save('map.html')

#%%

#import folium
#
#m = folium.Map([51.5, -0.25], zoom_start=10)
#
#test = folium.Html('<b>Hello world</b>', script=True)
#
#popup = folium.Popup(test, max_width=2650)
#folium.RegularPolygonMarker(
#    location=[51.5, -0.25], popup=popup,
#).add_to(m)
#
#m.save('osm.html')

    
#%%    

osm = show_fmaps(train_data, path=1)
osm.save('osm.html')
#kanton_map.save('map.html')



#%%

clus_map = clusters_map(centroid_pickups, train_cl, sig =0, zoom =3.2, circle =1, tile = 'Stamen Terrain')
#clus_map
clus_map.save('clus_map.html')

#%%

clus_map_sig = clusters_map(centroid_pickups, train_cl, sig =1, circle =1)
clus_map_sig.save("clus_map_sig.html")

#%%

from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(train_data.sample(1200)[['vendor_id','day_of_week', 'passenger_count', 'pick_month','label_pick', 'hour']], 'vendor_id', colormap='rainbow')
plt.show()

#%%


test_df = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "test.csv"))
test_fr = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "fastest_routes_test.csv"))
test_fr_new = test_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]

#test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
#test_fr = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')
#test_fr_new = test_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
test_df = pd.merge(test_df, test_fr_new, on = 'id', how = 'left')
test_df.head()

#%%

start = time.time()
test_data = test_df.copy()
test_data['pickup_datetime'] = pd.to_datetime(test_data.pickup_datetime)
test_data.loc[:, 'pick_month'] = test_data['pickup_datetime'].dt.month
test_data.loc[:, 'hour'] = test_data['pickup_datetime'].dt.hour
test_data.loc[:, 'week_of_year'] = test_data['pickup_datetime'].dt.weekofyear
test_data.loc[:, 'day_of_year'] = test_data['pickup_datetime'].dt.dayofyear
test_data.loc[:, 'day_of_week'] = test_data['pickup_datetime'].dt.dayofweek
end = time.time()
print("Time taken by above cell is {}.".format(end-start))

#%%

strat = time.time()
test_data.loc[:,'hvsine_pick_drop'] = haversine_(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)
test_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)
test_data.loc[:,'bearing'] = bearing_array(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)
end = time.time()
print("Time taken by above cell is {}.".format(end-strat))

#%%

start = time.time()
test_data['label_pick'] = k_means.predict(test_data[['pickup_longitude','pickup_latitude']])
test_data['label_drop'] = k_means.predict(test_data[['dropoff_longitude','dropoff_latitude']])
test_cl = pd.merge(test_data, centroid_pickups, how='left', on=['label_pick'])
test_cl = pd.merge(test_cl, centroid_dropoff, how='left', on=['label_drop'])
#test_cl.head()
end = time.time()
print("Time Taken by above cell is {}.".format(end-start))

#%%

start = time.time()
test_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)
test_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)
test_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)
test_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)
test_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)
test_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)
test_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)
test_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)
test_cl['speed_hvsn'] = test_cl.hvsine_pick_drop/test_cl.total_travel_time
test_cl['speed_manhtn'] = test_cl.manhtn_pick_drop/test_cl.total_travel_time
end = time.time()
print("Time Taken by above cell is {}.".format(end-start))

#%%

test_cl.head()

#%%

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings

#%%

# Lets Add PCA features in the model, reference Beluga's PCA
train = train_cl
test = test_cl
start = time.time()
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
end = time.time()
print("Time Taken by above cell is {}.".format(end - start))

#%%

train['store_and_fwd_flag_int'] = np.where(train['store_and_fwd_flag']=='N', 0, 1)
test['store_and_fwd_flag_int'] = np.where(test['store_and_fwd_flag']=='N', 0, 1)

#%%

feature_names = list(train.columns)
print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))
print("")
do_not_use_for_training = ['pick_date','id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("We will be using following features for training {}.".format(feature_names))
print("")
print("Total number of features are {}.".format(len(feature_names)))

#%%

y = np.log(train['trip_duration'].values + 1)

#%%

start = time.time()
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

# You could try to train with more epoch
model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds=2,
                  maximize=False, verbose_eval=1)
end = time.time()
print("Time taken by above cell is {}.".format(end - start))
print('Modeling RMSLE %.5f' % model.best_score)

#%%

weather = pd.read_csv(os.path.join(os.getcwd(), "AllNewYorkTaxiTripData", "weather_data_nyc_centralpark_2016.csv"))


#weather = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')
weather.head()

