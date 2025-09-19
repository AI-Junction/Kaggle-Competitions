# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:05:05 2017

@author: Chandrakant Pattekar
""" 

#regression from a scratch
 
import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

print (df.head())

print (df.tail())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]

print(df.head())
print(df)


forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.007*len(df)))

print('forecast out', forecast_out)
print (df[forecast_col].head())
print (df[forecast_col].tail())

df['label'] = df[forecast_col].shift(-forecast_out)

print(df['label'].head())
print(df['label'][-35:],df[forecast_col][-35:])
print(df['label'].shape)
print(df['label'])
df.dropna(inplace = True)
print(df.tail())
print(df.shape)

X = np.array(df.drop(['label'], 1))
print(X.shape)
y = np.array(df['label'])
X = preprocessing.scale(X)
#print(X.head())
X_lately = X[-forecast_out:]
print(X_lately)


print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)

print(clf.coef_)
#print(clf.decision_function)
print(clf.rank_)
print(clf._residues)
print(clf._estimator_type)
print(clf.fit_intercept)
#print(clf.degrees[0])



forecast_set = clf.predict(X_lately)
print(forecast_set.shape, X_lately.shape)
print(forecast_set)
print(df[forecast_col][-23:])

df['Forecast'] = np.nan
print(df['Forecast'] )

last_date = df.iloc[-1].name
#print(df.iloc[-3])
new_df = df[['label']]
#print(new_df)
#print(df['Name'].values)
#print(df.iloc)
#print(last_date)
last_unix = last_date.timestamp()
#print(last_unix)

one_day = 86400
next_unix = last_unix + one_day
#print(next_unix)
#print(df.loc[next_date])

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]

print (df.tail())

df['Adj. Close'].plot()                                        
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date ')
plt.ylabel('Price ')
plt.show()


