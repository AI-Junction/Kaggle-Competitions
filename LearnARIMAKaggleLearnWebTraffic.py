# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:10:32 2017

@author: echtpar
"""
"""
Web Traffic Time Series Forecasting
Forecast future traffic to Wikipedia pages
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import random as rnd
import pandas as pd
import itertools
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LinearRegression, LassoCV, RidgeClassifierCV

import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

#from datetime import datetime

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from pandas import datetime
from pandas import DataFrame
from pandas.tools.plotting import autocorrelation_plot

import statsmodels as sm1
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic


#from sklearn.cross_validation import cross_val_score

from tqdm import tqdm

from subprocess import check_output

import cv2

########################################
"""
Selected section below

"""
########################################



#%%

path_train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllWebTrafficTimeSeries\\train_1.csv"
path_submit = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllWebTrafficTimeSeries\\sample_submission_1.csv"


#%%

df_train = pd.read_csv(path_train)



df_train_nans = np.isnan(df_train.iloc[:, df_train.columns != 'Page']).sum().reset_index()
print(df_train_nans)

df_train_dtypes = df_train.dtypes.reset_index()
df_train_dtypes.columns = ['col','DType']
df_train_dtypes_grouped = df_train_dtypes.groupby(['DType']).aggregate('count')
print(df_train_dtypes_grouped)

#df_train = df_train.fillna(df_train.ffill())

df_train_traffic = df_train.drop(['Page'], axis = 1)

df_train_traffic = df_train_traffic.fillna(df_train_traffic.ffill())
df_train_traffic = df_train_traffic.fillna(df_train_traffic.bfill())

print(np.any(np.isnan(df_train_traffic)))


dict_pages = {}
daterange_index = pd.date_range('2015-07-01',  freq = 'D', periods = 550)
for i in range(0,10):
    #dict_pages[df_train.loc[i,['Page']]] = df_train.loc[i]
    dict_pages[df_train.loc[i,['Page']][0]] = pd.Series(list(df_train_traffic.loc[i]), index = daterange_index)
print((dict_pages.keys()))

y = dict_pages['2NE1_zh.wikipedia.org_all-access_spider']


for key in dict_pages:
    print(key)
    print(dict_pages[key].index)



#%%
## Function - timeseriesview START
# Function Called: Z = timeseriesview(y, 'co2', timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12)
#def timeseriesview(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):

plt.style.use('fivethirtyeight')
X = y.resample('D').mean()    
print(X)
X.plot(figsize=(15, 6))
plt.show()

f = plt.figure(figsize=(15,6))
autocorrelation_plot(X)
plt.show()
# return X
## Function - timeseriesview END


#%%
## Function - GetARIMApdq START
# Function Called: pdq_res, pdq_seas_res, train1, test1 = GetARIMApdq(Z, 'co2', timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12)
#def GetARIMApdq(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    

X = X.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
    print(train)
    print(test)
history = [x for x in train]
#    print(history)
predictions = list()

print("Step 1")

#    predictions = []

# forward walk
for t in range(len(test)):
#        print("within t loop")
    model = ARIMA(history, order=(10,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
r2score = r2_score (test, predictions)
print('Test MSE: %.3f' % error)
print('Test R2 Score: %.3f' % r2score)

# plot
pyplot.plot(test, color = 'blue')
pyplot.plot(predictions, color='red')
pyplot.show()

p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]

print(seasonal_pdq)
                
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#return pdq, seasonal_pdq, train, test
## Function - GetARIMApdq END


#%%
## Function - BestFitARIMA START
## Function called: df_pdq = BestFitARIMA(Z, pdq_res, pdq_seas_res, 'co2', 'MS', '1998-01-01', 12)
# def BestFitARIMA(train, pdq, seasonal_pdq, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    

seasonality = 7
warnings.filterwarnings("ignore") # specify to ignore warning messages
df_temp = pd.DataFrame()
param_temp = []
param_seasonal_temp = []
results_temp = []

for param in tqdm(pdq, miniters = 10):
    for param_seasonal in tqdm(seasonal_pdq, miniters = 10):
        try:
            mod = sm.tsa.statespace.SARIMAX(train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            param_temp.append(param)
            param_seasonal_temp.append(param_seasonal)
            results_temp.append(results.aic)
                
            print('ARIMA{}x{}{} - AIC:{}'.format(param, param_seasonal, seasonality, results.aic))
        except:
#                print('in exception')
            continue
#    df_temp.columns = ['param','param_seasonal','AIC_Result']
df_temp['param'] = param_temp
df_temp['param_seasonal'] = param_seasonal_temp
df_temp['AIC_Result'] = results_temp


df_pdq = df_temp

print(df_temp)
print(seasonality)

## return df_temp
## Function - BestFitARIMA END

#%%    

## Function - ARIMAResultsShow START
#param_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param
#param_seasonal_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param_seasonal
#
##param_temp_list = ([str(param_temp)[7], str(param_temp)[10], str(param_temp)[13]])                                 
##param_seasonal_temp_list = ([str(param_seasonal_temp)[7], str(param_seasonal_temp)[10], str(param_seasonal_temp)[13], str(param_seasonal_temp)[16:18]])                                 
#                                 
#                                 
#print(list(param_temp), list(param_seasonal_temp))                                 
#                                 

# Function Called: res = ARIMAResultsShow(Z, param_temp, param_seasonal_temp, 12)
## def ARIMAResultsShow(train, pdq_order, pdq_seasonal_order, seasonality):
# uncomment later

pdq_order = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param
pdq_seasonal_order = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param_seasonal

print(pdq_order)
print(pdq_seasonal_order)

p = list(pdq_order)[0][0]
d = list(pdq_order)[0][1]
q = list(pdq_order)[0][2]

print(p,d,q)

p_season = list(pdq_seasonal_order)[0][0]
d_season = list(pdq_seasonal_order)[0][1]
q_season = list(pdq_seasonal_order)[0][2]

print(p_season,d_season,q_season)  

print('before mod')

mod = sm.tsa.statespace.SARIMAX(train,
                                order=(p,d,q),
                                seasonal_order=(p_season, d_season, q_season, seasonality),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

#    mod = sm.tsa.statespace.SARIMAX(train,
#                                    order=(1,1,1),
#                                    seasonal_order=(1,1,1, 12),
#                                    enforce_stationarity=False,
#                                    enforce_invertibility=False)



print('after mod')

results = mod.fit()

print(results.summary().tables[1])

#    model = ARIMA(y, order=(8,1,0), freq = 'A')
#    model_fit = model.fit(disp=0)
#    print(model_fit.summary())
#    # plot residual errors
#    residuals = DataFrame(model_fit.resid)
#    residuals.plot()
#    pyplot.show()
#    residuals.plot(kind='kde')
#    pyplot.show()
#    print(residuals.describe())
results.plot_diagnostics(figsize=(15, 12))
plt.show()
print("before pred")
#    start = None
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
#    print(pred)
#    print(pred)
#    return results

## Function - ARIMAResultsShow END



#%%

## Function - ARIMAPredict START

#start=pd.to_datetime('1998-01-01')
#print(start)

#pred = res.get_prediction(pd.to_datetime('1998-01-01 00:00:00'), dynamic=False)
#pred = res.get_prediction(dynamic=False)

#z = pd.to_datetime('1998-01-01 00:00:00')

## Function Called: pred_ret = ARIMAPredict(Z, res, z)

## def ARIMAPredict(train2, results_arg, predict_start):    

print('within ARIMAPredict')
print(results)
#    print(pd.to_datetime(predict_start_date))
#    startdatetime = pd.to_datetime(predict_start_date)
#    pred = results.get_prediction(start=pd.to_datetime(predict_start_date), dynamic=False)
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred = results.get_prediction(dynamic=False)
print(pred.predicted_mean)
print(type(pred.predicted_mean))
print(train)
#    fig = plt.figure(figsize = (8, 16))
#    pred = results.get_prediction(dynamic=False)
pred_ci = pred.conf_int()
#    ax = train['1990':].plot(label='observed')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([300,380])
#    ax = axes.flatten()

print(train.columns.values)
train = pd.DataFrame(train, index = None)
train.index = pd.date_range('2015-07-01',  freq = 'D', periods = 363)

print(train)

ax = train.plot(label='observed')



#    df_pred = pd.DataFrame(pred.predicted_mean)
#    z = np.arange(predict_start, len(df_pred)+predict_start)
#    print(z)
#    df_pred.index = z
#    print(len(df_pred))
#    print(df_pred.index)

predicted_df = pd.DataFrame(pred.predicted_mean, index = None)
predicted_df.index = pd.date_range('2015-07-01',  freq = 'D', periods = 363)
predicted_df.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

print(predicted_df)

#    print(pd.DataFrame(pred.predicted_mean))
#    pred_ci.index = z
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
    
ax.set_xlabel('Date')
ax.set_ylabel('Quantity or Value')
plt.legend()

plt.show()    
# return pred


## Function - ARIMAPredict END


#%%

## Function - forecast
## Function Called: forecast(z, pred_ret, res, Z, 'Date', 'Pollution Levels')    
# def forecast(forecaststartdate, pred1, results2, train2, xlabel, ylabel):    
forecaststartdate = '2016-07-01'
xlabel, ylabel = 'Time', 'Value'
y_forecasted = pred.predicted_mean
y_truth = train.loc[forecaststartdate:]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#pred_dynamic = results.get_prediction(start=pd.to_datetime(forecaststartdate), dynamic=True, full_results=True)

print()
pred_dynamic = results.get_prediction(start = '2016-07-01 00:00:00', dynamic=True, full_results=True)

pred_dynamic_ci = pred_dynamic.conf_int()

print(pred_dynamic.predicted_mean)

ax = train2['1990':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(forecaststartdate), train2.index[-1],
                 alpha=.1, zorder=-1)

fig = plt.figure()
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

plt.legend()
plt.show()


# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = train2[forecaststartdate:]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)

print(type(pred_uc.predicted_mean))

forecast_index = pd.date_range('2016-06-28',  freq = 'D', periods = 500)
print(len(forecast_index))

print(forecast_index)
df_forecast = pd.DataFrame(pred_uc.predicted_mean, index = forecast_index)
print(df_forecast)


# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

fig1 = plt.figure()
ax = train.plot(label='observed', figsize=(20, 15))
df_forecast.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

plt.legend()
plt.show()

    

























#    f, ax = plt.subplots(1,1, figsize=(15,6))
    
########################################
"""
Trial section below

"""
########################################


path_train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllWebTrafficTimeSeries\\train_1.csv"
path_submit = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllWebTrafficTimeSeries\\sample_submission_1.csv"

def parser(x):
#        print(x)
        return datetime.strptime('190'+x, '%Y-%m')
        
#df_train = pd.read_csv(path_train, parse_dates=[0], dtype = {'2016-12-31': np.int32}, index_col=None, skipinitialspace=False, date_parser=parser(x))
df_train = pd.read_csv(path_train)

print(list(df_train.columns))

df_train_nans = np.isnan(df_train.iloc[:, df_train.columns != 'Page']).sum().reset_index()

print(df_train_nans)

f_train_temp = None
df_train_temp = df_train.loc[:,['2015-07-01', '2015-07-02']]
df_train_temp = df_train.iloc[:,[1,2,3]]
df_train_temp.index[:10]
df_train_temp = df_train.loc[:10,['Page', '2015-07-01', '2015-07-02']]

print(df_train_temp)

df_train_temp.loc[:,['2015-07-01']]

print([x for x in range(1,10,2)])
print(np.arange(1,10,2))
print(range(1,10,2))

for i in np.arange(len(df_train_temp)):
    print(df_train_temp.loc[i,['2015-07-01', '2015-07-02']])

print(df_train_temp.describe())
print(df_train_temp)


df_train_temp = df_train(['Page', '2015-07-01', '2015-07-02'])[:10]

print(df_train.iloc[10, :])


df_train_dtypes = df_train.dtypes.reset_index()
df_train_dtypes.columns = ['col','DType']
print(df_train_dtypes)
df_train_dtypes_grouped = df_train_dtypes.groupby(['DType']).aggregate('count')
print(df_train_dtypes_grouped)

df_train_temp = df_train_temp.fillna(df_train_temp.ffill())
print(type(df_train_temp))
print(df_train_temp)


def timeseriesview(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):

    plt.style.use('fivethirtyeight')

    X  = series[col].resample(timefreq).mean()
    X = X.fillna(X.bfill())

    X.plot(figsize=(15, 6))
    plt.show()
    
#    f, ax = plt.subplots(1,1, figsize=(15,6))
    f = plt.figure(figsize=(15,6))
    autocorrelation_plot(X)
    plt.show()
    return X

    
data = sm.datasets.co2.load_pandas()
y = data.data
print(y)

print(df_train_temp)
df_page1=None

dict_pages = {}

print(df_train.loc[1,['Page']])

print(df_train.columns.values)

print(df_train.loc[0])
df_train.loc[0,['Page']][0]

cols = list(df_train.columns != 'Page')
print(cols)

df_train_traffic = df_train.drop(['Page'], axis = 1)
print(list(df_train_traffic.columns))

for i in range(0,10):
    #dict_pages[df_train.loc[i,['Page']]] = df_train.loc[i]
    dict_pages[df_train.loc[i,['Page']][0]] = list(df_train_traffic.loc[i])
print(dict_pages)
print(pd.Series((dict_pages['3C_zh.wikipedia.org_all-access_spider'])))

y = pd.Series((dict_pages['3C_zh.wikipedia.org_all-access_spider'])).astype(int)

df_page1 = df_train.loc[0]
print(type(df_page1))
print(df_page1[:10])

print(type(y))


#drange_template_overall = pd.date_range('2015-07-01',  '2016-12-31', freq = 'D', periods = 1)
#drange_template2 = pd.date_range('2015-07-01',  freq = 's/min/H/m/D/M/MS/Q/A/AS', periods = 10)
drange_template2 = pd.date_range('2015-07-01',  freq = 'D', periods = 550)

print([x for x in drange_template2])

y.index = drange_template2

print(y)

Z = timeseriesview(X, 0, timefreq='D', predict_start_date = '2015-07-01', seasonality = 12)

plt.style.use('fivethirtyeight')
X = y.resample('M').mean()    
print(X)
X.plot(figsize=(15, 6))
plt.show()

#    f, ax = plt.subplots(1,1, figsize=(15,6))
f = plt.figure(figsize=(15,6))
autocorrelation_plot(X)
plt.show()
return X

