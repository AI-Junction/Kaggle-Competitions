# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:11:42 2017

@author: echtpar
"""

#how does callback work in fit?

############################


"""

CHEAT CODES 


References

Analytics Vidhya:
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
https://www.analyticsvidhya.com/blog/2016/01/python-tutorial-list-comprehension-examples/

"""





#############################

"""

Check versions of the library packages

"""



#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))







"""

IMPORTS

"""

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import BaggingRegressor 
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as ltf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *
from PIL import Image


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


### IMPORTS END


"""

NUMPY START

"""

# NP Mean start

arr = np.random.randn(3,4)
print(arr)
means_row = np.mean(arr, axis=0)
print(means_row)
print(np.mean(means_row))
means = np.mean(arr, axis = 1)
print(means)
print(np.mean(means))

'''
[[ 0.37643616  0.27837363 -0.50173784  2.12934351]
 [ 1.75628806 -1.05496376 -1.91975051 -0.75540659]
 [-1.93737286 -1.19584108 -1.5163363   0.40814889]]
'''

np.mean([0.37643616, 1.75628806,  -1.93737286])

means_col = np.mean(arr, axis=1)
print(means_col)
np.mean([ 0.37643616,  0.27837363, -0.50173784,  2.12934351])

np.argsort(means_col)[::-1][:2]
print(np.argsort(means_col)[::-1][:2])
np.argsort(means_col)[::-1]
a = np.argsort(means_col)
print(means_col[a])


# NP Mean end

# NP Slice start
# slice works like this: sequence[start:stop:step]

x = np.array([1,2,3,4,5,6])
y = ['a','b','c','d','e','f']
x[::-1]
x[::-2]

z = np.where(x > 3)
print(z)

x[::2]


# NP Slice end



action = np.random.choice(5, 2, p=[0.1,0.2,0.4,0.2,0.1])
print(action)

out = np.random.randn(1)
print(out)

out = np.random.randn(10,2)
print(out)

out = np.random.rand(10,2)
print(out)


out = np.random.randint(2,10, size=(3,4))
print(out)

p = 0.5

x = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]])
print(*x.shape)
q = (*x.shape)
print(np.random.rand(*x.shape))
print(np.random.rand(x.shape)) # does not work
print(np.random.rand(*x.shape) < p)
print(np.random.randint(5,4))
print(x.shape)
mask = (np.random.rand(*x.shape) < p) /p 
print(mask)

print(np.mean(np.random.rand(10)), (np.random.rand(10)))
print(*(np.random.rand(5,4)).shape)



from math import sqrt
n = 100
sqrt_n = int(sqrt(n))
no_primes = {j for i in range(2,sqrt_n) for j in range(i*2, n, i)}
print(no_primes)



np.ceil([2.15,3.04, 5.89])
np.floor([2.15,3.04, 5.89])
np.round([2.15,3.04, 5.89],1)

from random import random
from numpy import array
from numpy import cumsum


# create a sequence of random numbers in [0,1]
X = array([random() for _ in range(10)])
# calculate cut-off value to change class values
limit = 10/4.0
# determine the class outcome for each item in cumulative sequence
y = array([0 if x < limit else 1 for x in cumsum(X)])

print(X, y)
print([x for x in cumsum(X)])


'''
One other possible way to get a discrete distribution 
that looks like the normal distribution is to draw from 
a multinomial distribution where the probabilities are 
calculated from a normal distribution.
'''

'''
Here, np.random.choice picks an integer from [-10, 10]. 
The probability for selecting an element, say 0, is 
calculated by p(-0.5 < x < 0.5) where x is a normal random 
variable with mean zero and standard deviation 3. 
I chooce std. dev. as 3 because this way p(-10 < x < 10) is 
almost 1.
'''




import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11)
print(x)
xU, xL = x + 0.5, x - 0.5 
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
print(prob)
print(sum(prob))

prob = prob / prob.sum() #normalize the probabilities so their sum is 1
nums = np.random.choice(x, size = 10000, p = prob)
plt.hist(nums, bins = len(x))



x = np.arange(-10, 11)
xU, xL = x + 0.5, x - 0.5 
prob = ss.norm.cdf(xU, scale = 5) - ss.norm.cdf(xL, scale = 5)
print(prob)
print(sum(prob))

prob = prob / prob.sum() #normalize the probabilities so their sum is 1
nums = np.random.choice(x, size = 10000, p = prob)
plt.hist(nums, bins = len(x))




#abs(mu - np.mean(s)) < 0.01
#abs(sigma - np.std(s, ddof=1)) < 0.01
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2, figsize = (16,16))
axes = ax.flat


mu, sigma = 10, .5 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
print(s.shape)
print(np.mean(s))
print(np.var(s))
print(np.std(s))
print(np.min(s))
print(np.max(s))
#s = (s-mu)/sigma
sigma = np.std(s)
print(sigma, sigma*sigma)

count, bins, ignored = axes[0].hist(s, 30, normed=True)
axes[0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')




mu, sigma = 0, 0.5 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
print(s.shape)
print(np.mean(s))
print(np.var(s))
print(np.std(s))
print(np.min(s))
print(np.max(s))
s = s/sigma
sigma = np.std(s)
print(sigma)

count, bins, ignored = axes[1].hist(s, 30, normed=True)
axes[1].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')


mu, sigma = 0, 100 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
print(s.shape)
print(np.mean(s))
print(np.var(s))
print(np.std(s))
print(np.min(s))
print(np.max(s))
s = s/sigma
sigma = np.std(s)

count, bins, ignored = axes[2].hist(s, 30, normed=True)
axes[2].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')



mu, sigma = 50, 100 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
print(s.shape)
print(np.mean(s))
print(np.var(s))
print(np.std(s))
print(np.min(s))
print(np.max(s))
s = s/sigma
sigma = np.std(s)

count, bins, ignored = axes[3].hist(s, 30, normed=True)
axes[3].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')



plt.show()


# to convert to Encoded Labels, Binarized Labels, Categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

lblEnc = LabelEncoder()

nb_classes = 4
classes = ['a', 'b', 'c','d']
y_train = ['a','b','c','d','a','b','c','d','a','b']
lblEnc.fit(classes)
lblEnc.classes_
y_train_enc = lblEnc.transform(y_train)
print(y_train_enc)

lblBin = LabelBinarizer()
lblBin.fit(y_train)
y_train_bin = lblBin.transform(y_train)
print(y_train_bin)
lblBin.classes_

Y_train = np_utils.to_categorical(y_train_enc, nb_classes)
print(Y_train)

oneHotEnc = OneHotEncoder()
#oneHotEnc.fit(y_train_enc)
y_train_oneHot = oneHotEnc.fit_transform(y_train_enc.reshape(-1,1))
print(y_train_oneHot)
z = y_train_oneHot.toarray()
print(z)



z = pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3, labels=["good","medium","bad"])
z = pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3)

print(type(z))
print(z)
print(z[0])
print(type(z[0]))
print(list(z))

import math
size_bytes = 1500
print(math.log(size_bytes, 1024))




# numpy stack example

import numpy as np 
a = np.array([1,2,3,4]) 

print ('First Array:') 
print (a) 
print ('\n')
b = np.array([5,6,7,8]) 
b3Dim = np.stack([b]*3, axis=0) 
print(b3Dim)

c = np.array([[1,2,3,4],[5,6,7,8]])
print(c)
c3Dim = np.stack([c]*3, axis=2) 
print(c3Dim)



print ('Second Array:') 
print (b) 
print ('\n')  

print ('Stack the two arrays along axis 0:') 
print (np.stack((a,b),0).shape) 
print (np.stack((a,b),0)) 
print ('\n')  

print ('Stack the two arrays along axis 1:') 
print (np.stack((a,b),1).shape)
print (np.stack((a,b),1))


rgb = np.floor(np.random.rand(3) * 256).astype('int')
print(rgb)


### NUMPY END






"""

NUMPY MATRIX MULTIPLY START

"""


np.random.seed(10000)
A = np.random.randint(2,10, size=(3,4))
print(A)
print(np.matrix(A))

np.random.seed(10001)
B = np.random.randint(2,10, size=(4,3))
print(B)

C = np.matrix(A)*np.matrix(B)
C_ = A*B


print(C)


D = np.dot(A,B)
print(D)



### NUMPY MATRIX MULTIPLY END


"""

DATES START

"""

from datetime import datetime
import datetime as dt

# ref URL: http://www.marcelscharth.com/python/time.html
# ref URL: https://stackoverflow.com/questions/32168848/how-to-create-a-pandas-datetimeindex-with-year-as-frequency

tmp_date = datetime.strptime('2005-06-01 17:59:00', '%Y-%m-%d %H:%M:%S')
tmp_date = tmp_date.astype('datetime64[ns]')

print(type(tmp_date))
print(tmp_date.date())
print(tmp_date.minute)
print(dt.datetime.today().weekday())
print(tmp_date.weekday())
print(type(tmp_date))

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

print(train_flattened[:10].date)
print(train_flattened[:10].date.dt)
print(train_flattened[:10].date.dt.dayofweek)
print(train_flattened[:10].date.dt.month)
print(train_flattened[:10].date.dt.day)
print(train_flattened[:10].date.dt.year)

train_flattened['year']=train_flattened.date.dt.year 
train_flattened['month']=train_flattened.date.dt.month 
train_flattened['day']=train_flattened.date.dt.day


pickupTime = pd.to_datetime(taxiDB['pickup_datetime'])
taxiDB['src hourOfDay'] = (pickupTime.dt.hour*60.0 + pickupTime.dt.minute)   / 60.0


from statsmodels.tsa.tsatools import lagmat


'''
    's' : second
    'min' : minute
    'H' : hour
    'D' : day
    'w' : week
    'm' : month
    'A' : year 
    'AS' : year start
'''

# months
print(dt.datetime.today())
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'm')
print(z)

z = pd.date_range(dt.datetime.today(), periods=10, freq = 'M')
print(z)


#Month Start
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'MS')
print(z)

#seconds
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'S')
print(z)

#min
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'min')
print(z)


#hours
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'H')
print(z)


#Quarterly
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'Q')
print(z)


#Day
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'D')
print(z)


#year end
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'A')
print(z)

#year start
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'AS')
print(z)







### DATES END



"""

ZIP, MAP and LAMBDA


Many novice programmers (and even experienced programmers who are new to python) often get confused when they first see zip, map, and lambda. This post will provide a simple scenario that (hopefully) clarifies how these tools can be used.

To start, assume that you've got two collections of values and you need to keep the largest (or smallest) from each. These could be metrics from two different systems, stock quotes from two different services, or just about anything. For this example we'll just keep it generic.

So, assume you've got a and b: two lists of integers. The goal is to merge these into one list, keeping whichever value is the largest at each index.




"""

a = [1, 2, 3, 4, 5]
b = [2, 2, 9, 0, 9]

print(list(np.sort(b)))
print(type(sorted(b)))

c = ['a','b','d','e']
print(c)
c = c + ['c']
print(c)
c.append('f')


#This really isn't difficult to do procedurally. You could write a simple function that compares each item from a and b, then stores the largest in a new list. It might look something like this:

def pick_the_largest(a, b):
    result = []  # A list of the largest values

    # Assume both lists are the same length
    list_length = len(a)
    for i in range(list_length):
        result.append(max(a[i], b[i]))
    return result


#While that's fairly straightforward and easy to read, there is a more concise, more pythonic way to solve this problem.

zip(a, b)

print([x for x in zip(a,b)])

# You now have one list, but it contains pairs of items from a and b. For more information, check out zip in the python documentation.

# lambda is just a shorthand to create an anonymous function. 
# It's often used to create a one-off function (usually for scenarios when you need 
# to pass a function as a parameter into another function). 
# It can take a parameter, and it returns the value of an expression. 
# For more information, see the Python documentation on lambdas.

lambda pair: max(pair)


# map takes a function, and applies it to each item in an iterable (such as a list). 
# You can get a more complete definition of map from the python documentation, 
# but it essentially looks something like this:
    
z = map(  # apply the lambda to each item in the zipped list
        lambda pair: max(pair),  # pick the larger of the pair
        zip(a, b)  # create a list of tuples
    )    

print([x for x in z])


df = pd.DataFrame(np.random.randint(10,size = (4,3)))
print(type(df))
print(df)

df['newcol'] = df[2].apply(lambda x: 2*x)
df['newcol2'] = df['newcol'].map(lambda x: 2*x)
df['newcol3'] = df['newcol2'].map(lambda x: 2*x)

df.columns = ['col1','col2','col3','col4','col5','col6']
print(df.columns)
df['col7'] = df.apply(lambda x: [2*x[0]])
print(df['col7'])

df.drop(['col7'], inplace=True, axis=1)

z = df.apply(lambda x: sum(x), axis=0)
print(z)

z = df.apply(lambda x: sum(x), axis=1)
print(z)

z = df['newcol2'].map(lambda x: x+4)
print(z)

#z = df.map(lambda x: x+4) # does not work
#print(z)

print(df)
    

# map specific values in a pandas dataframe
dataset= pd.DataFrame(data = ['S', 'C', 'Q', 'Q', 'C', 'S'], columns = ['Embarked'])
print(dataset)
dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
print(dataset)



### ZIP, MAP and LAMBDA END



"""

STRING OPERATIONS START

"""

import pandas as pd
a = "01-03-2017"
z = a.split('-', 1)
print(type(z))
print(z)

df = pd.DataFrame(["01-03-2017", "02-03-2017", "03-03-2017", "04-03-2017"], columns = ['Date'])
print(df)
df1 = pd.get_dummies(df.Date)
print(df1)

import random
sample = list()
n_sample = round(len(prediction) * 0.8)
print(n_sample)
while len(sample) < n_sample:
    index_tmp = random.randrange(len(prediction))
    print('index_tmp', index_tmp) 
    sample.append(prediction[index_tmp])
    print(prediction[index_tmp])



### STRING OPERATIONS END




"""

INDEX OPERATIONS START

"""

dict_test = {'a': [1,2,3,4], 'b' : [5,6,7,8], 'c' : [9,10,11,12]}
dict_test_2 = {'d': [9,10,11,12], 'e' : [5,6,7,8], 'f' : [1,2,3,4]}
print(dict_test_2)    

df_test=None

z = range(10, 14, 1)
print([x for x in z])
df_test = pd.DataFrame(data = dict_test, index = z)
df_test[['d','e','f']] = pd.DataFrame(data = dict_test_2, columns = ['d','e','f'], index = df_test.index)
print(df_test)
print(df_test.f)

    
df_test = pd.DataFrame(dict_test, index = z)
print(df_test.index)
print(df_test)

df_test['new_num'] = df_test['a'].apply(lambda x: x+1)
dt_range = pd.date_range(start='23-08-2017', periods = 4)
print(dt_range)

df_test['date_col'] = dt_range
print(df_test)


df_test.reset_index(drop = False, inplace= True)
print(df_test.index)
print(df_test)

df_test.reset_index(drop = True, inplace= True)
print(df_test.index)
print(df_test)


a = [1,2,3,4,5]
z = lagmat(a, 4)
print(z)



### INDEX OPERATIONS END


"""

PANDAS SERIES AND DATAFRAME START

Ref URL:
https://discuss.analyticsvidhya.com/t/difference-between-map-apply-and-applymap-in-pandas/2365


"""


row_test = None
row_test = pd.Series([1,2,3,4,5,6,7,9,0]
                ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])

print(row_test)
print(row_test['lag7'])


df_row_test = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
print(df_row_test)
df_row_test = df_row_test.append([row_test])


train_flattened['month'].replace('11','11 - November',inplace=True)
train_flattened['month'].replace('12','12 - December',inplace=True)

train_group = train_group.pivot('weekday','month','Visits')


data_pred['weekday'] = data_pred['date'].apply(lambda x:x.weekday())

for i in range(number_of_days-1):
    lag1 = data_lag.tail(1)["diff"].values[0]
    lag2 = data_lag.tail(1)["lag1"].values[0]
    lag3 = data_lag.tail(1)["lag2"].values[0]
    lag4 = data_lag.tail(1)["lag3"].values[0]
    lag5 = data_lag.tail(1)["lag4"].values[0]
    lag6 = data_lag.tail(1)["lag5"].values[0]
    lag7 = data_lag.tail(1)["lag6"].values[0]
    lag8 = data_lag.tail(1)["lag7"].values[0]

    lag9 = data_lag.tail(1)["lag8"].values[0]
    lag10 = data_lag.tail(1)["lag9"].values[0]
    lag11 = data_lag.tail(1)["lag10"].values[0]
    lag12 = data_lag.tail(1)["lag11"].values[0]
    lag13 = data_lag.tail(1)["lag12"].values[0]
    lag14 = data_lag.tail(1)["lag13"].values[0]
    lag15 = data_lag.tail(1)["lag14"].values[0]
    lag16 = data_lag.tail(1)["lag15"].values[0]
    
#        print('lag values for i = ', i, ':', lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8, lag9, lag10, lag11, lag12, lag13, lag14, lag15, lag16)
    
    weekday = data_pred['weekday'][0]
    
    row = pd.Series([lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8,lag9, lag10, lag11, lag12, lag13, lag14, lag15, lag16, weekday]
                    ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','lag9', 'lag10', 'lag11', 'lag12', 'lag13', 'lag14', 'lag15', 'lag16','weekday'])
    
#        print(row)
    
    to_predict = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','lag9', 'lag10', 'lag11', 'lag12', 'lag13', 'lag14', 'lag15', 'lag16','weekday'])
    prediction = pd.DataFrame(columns = ['diff'])
    to_predict = to_predict.append([row])
    prediction = pd.DataFrame(model.predict(to_predict),columns = ['diff'])
    
    last_predict = data_lag["CPULevel"][data_lag.shape[0]-1] + prediction.values[0][0]
#        print(last_predict)
#        print(data_pred['date'][0])
    row['date'] = ''
    row['date'] = data_pred['date'][0]
    row['diff'] = prediction.values[0][0]
    row['CPULevel'] = last_predict
    data_lag = data_lag.append([row], ignore_index = True)


    df2 = np.random.randint(3,10, size=(5,4))
    print(df2)
    df1 = np.random.randn(3,4)
    print(df1)
    df1 = pd.DataFrame(data = df1)
    q = np.arange(3, 3+len(df1))
    print(q)
    df1 = df1.set_index(q)   
    print(df1)
    
    
'''
# Ref Analytics Vidya URL on handling pandas dataset:
# https://www.analyticsvidhya.com/blog/2016/01/python-tutorial-list-comprehension-examples/    
'''


    
#Lets load the dataset:
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("skills.csv")
print (data)    

#Split text with the separator ';'
data['skills_list'] = data['skills'].apply(lambda x: x.split(';'))
print (data['skills_list'])

z = None

#first convert list to just a comma separated string
data['skills_list_new'] = data['skills_list'].apply (lambda x: ", ".join(x))
print(data['skills_list_new'])

# now apply get_dummies function on the data series
w = data['skills_list_new'].str.get_dummies(sep = ", ")
print(w)

print([sport for l in data['skills_list'] for sport in l])


#Initialize the set
skills_unq = set()
#Update each entry into set. Since it takes only unique value, duplicates will be ignored automatically.
skills_unq.update( (sport for l in data['skills_list'] for sport in l) )
print (skills_unq)


lblEncoder = LabelEncoder()
z = lblEncoder.fit(data['skills_list'])
z = lblEncoder.transform(data['skills_list'])
print(z)
print(data['skills_list'])



### PANDAS SERIES AND DATAFRAME END


"""

GROUP BY START

"""


dict_test = {'a': [1,2,3,4,2], 'b' : [5,6,7,8,5], 'c' : [9,10,11,12,9]}
df_test=None
z = range(10, 15, 1)
df_test = pd.DataFrame(dict_test, index = z)
print(np.where(df_test['b'] == 8)[:])


print(df_test)

df_test_grouped = df_test.groupby(['a', 'c'])['b'].aggregate('max').to_frame().reset_index()

print(df_test_grouped)
df_test_grouped.columns = ['A','B','C']

print(df_test)
df_test_grouped_tmp = df_test.groupby(['a', 'c'])
print(df_test_grouped_tmp)
print([x for x in df_test_grouped_tmp])



group_cols = ['a', 'b']
for n_group, n_rows in df_test.groupby(group_cols):
    print([str(col_value) + str(col_value) for col_name, col_value in zip(group_cols, n_group)])
    #c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}

print(c_row)



print(df_test_grouped.index)
print(df_test_grouped.columns.values)
print(df_test_grouped)
print(type(df_test_grouped))
r = range(10,15,1)
print([x for x in r])
df_test_grouped.index = r
print(df_test_grouped.index.values)
print(df_test.iloc[1:,1].as_matrix())
print(df_test.iloc[:len(df_test)-1,1].as_matrix())
print(df_test.iloc[1:,1].as_matrix() - df_test.iloc[:len(df_test)-1,1].as_matrix())
print(df_test.iloc[:,1].as_matrix())
print(type(df_test.iloc[1:,1].as_matrix()))


### GROUP BY END

"""

MATPLOTLIB BY START

"""

import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import pylab
import cv2 as cv2
import numpy as np



# Type 1 = use axis

#fig = plt.figure(figsize = (10,16))
ax = plt.subplot(2,1,1)
ax.scatter(np.arange(100), np.random.randint(100, size = (100)))
ax.set_title("test")

ax = plt.subplot(2,1,2)
ax.scatter(np.arange(100), np.random.randint(100, size = (100)), alpha = 0.25)

plt.tight_layout()
fig.subplots_adjust(hspace=0.5)
plt.show()


# Type 2 no use of axis

plt.figure(1)
plt.subplot(2,2,1)
plt.plot(np.arange(100), np.random.randint(100, size = (100)))
plt.xlabel("X")
plt.ylabel("Y")
plt.title ("test")

plt.subplot(2,2,2)
plt.plot(np.arange(100), np.random.randint(100, size = (100)))
plt.title("test")
plt.xlabel("X")
plt.ylabel("Y")

plt.subplots_adjust(wspace=0.5)
plt.show()



# Type 3 use of axis with add_subplot


fig = plt.figure(figsize = (10,16))
ax1 = fig.add_subplot(2,1,1)
ax1.scatter(np.arange(100), np.random.randint(100, size = (100)), alpha = 0.25)
ax1.set_title("test")

ax2 = fig.add_subplot(2,1,2)
ax2.scatter(np.arange(100), np.random.randint(100, size = (100)), alpha = 0.25)
ax2.set_title("test")

plt.show()



# Type 4 no use of axis with add_subplot

fig = plt.figure(figsize = (10,16))
fig.add_subplot(2,1,1)
plt.scatter(np.arange(100), np.random.randint(100, size = (100)), alpha = 0.25)
plt.title("test")

fig.add_subplot(2,1,2)
plt.scatter(np.arange(100), np.random.randint(100, size = (100)), alpha = 0.25)
plt.title("test")

plt.show()






fig, axes = plt.subplots(4,4, figsize = (28,28))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
z = axes.flat

print(axes.shape)
img = cv2.imread("C:\\Users\\Public\\Pictures\\Sample Pictures\\Koala.jpg")
axes.flat[0].imshow(img)
axes.flat[1].imshow(img)
axes.flat[2].imshow(img)
axes.flat[3].imshow(img)
axes.flat[4].imshow(img)
axes.flat[5].imshow(img)
axes.flat[6].imshow(img)
axes.flat[7].imshow(img)
axes.flat[8].imshow(img)
axes.flat[9].imshow(img)
axes.flat[10].imshow(img)
axes.flat[11].imshow(img)
axes.flat[12].imshow(img)
axes.flat[13].imshow(img)
axes.flat[14].imshow(img)
axes.flat[15].imshow(img)



f = plt.figure(figsize = (12,6))
plt.imshow(img)
plt.show()



img = cv2.imread("C:\\Users\\Public\\Pictures\\Sample Pictures\\Koala.jpg")
plt.figure(figsize = (16,8))

plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(img)

plt.subplot(223)
plt.imshow(img)

plt.subplot(224)
plt.imshow(img)

plt.show()


x = np.arange(1,10,1)
y = np.arange(11,20,1)
z = range(1,10,1)
q = [1,2,4,6,3,5,9,10,3]

print(x,y)

z = [[1,2],[3,4],[5,6],[7,8]]

plt.figure(figsize = (16,8))

plt.subplot(221)
plt.plot(x,y, color = 'g')
 
plt.subplot(222)
plt.scatter(x,y)

plt.subplot(223)
r = np.random.normal(size = 1000)
plt.hist(r, normed=True, bins=10)
plt.ylabel('Probability')

plt.subplot(224)
plt.bar(y, height = q)

plt.show()




f, ax = plt.subplots(2,2, figsize = (16,8))
ax.flat[0].plot(x,y, color = 'g')
ax.flat[1].scatter(x,y)

r = np.random.normal(size = 1000)
ax.flat[2].hist(r, normed=True, bins=10)
plt.ylabel('Probability')

ax.flat[3].bar(y, height = q)
plt.show()



import pandas as pd
import seaborn as sns

train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")

# plot the important features #

train.ix[:10,:10].columns
train.ix[:, 10:].columns
train['X10'].unique()
print(np.sort(train['X10'].unique()).tolist())

for col in train.ix[:, 10:].columns:
    print(col, np.sort(train[col].unique()).tolist())

print(train.columns.values)    
plt.figure(figsize=(30,6))

ax = plt.subplot(1,3,1)
var_name = "X0"
col_order = np.sort(train[var_name].unique()).tolist()
print(col_order)
ax = sns.stripplot(x=var_name, y='y', data=train, order=col_order)

ax = plt.subplot(1,3,2)
var_name = "X1"
col_order = np.sort(train[var_name].unique()).tolist()
ax = sns.boxplot(x=var_name, y='y', data=train, order=col_order)


ax = plt.subplot(1,3,3)
var_name = "X2"
col_order = np.sort(train[var_name].unique()).tolist()
ax = sns.violinplot(x=var_name, y='y', data=train, order=col_order)

plt.show()    




#Examples
#--------

#Initialize a 2x2 grid of facets using the tips dataset:


import seaborn as sns; sns.set(style="ticks", color_codes=True)
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time", row="smoker")

#Draw a univariate plot on each facet:


import matplotlib.pyplot as plt
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")
print(tips.total_bill)


#(Note that it's not necessary to re-catch the returned variable; it's
#the same object, but doing so in the examples makes dealing with the
#doctests somewhat less annoying).

#Pass additional keyword arguments to the mapped function:


import numpy as np
bins = np.arange(0, 65, 5)
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=bins, color="r")

print(tips.columns.values)
print(tips.groupby(['sex','smoker'])['total_bill'].aggregate('sum').reset_index())
print(tips.groupby(['time','smoker','day'])['total_bill'].aggregate('sum').reset_index().sort('day', ascending = False))


#Plot a bivariate function on each facet:


g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

#Assign one of the variables to the color of the plot elements:


g = sns.FacetGrid(tips, col="time",  hue="smoker")
g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
      .add_legend())

#Change the size and aspect ratio of each facet:


g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
g = g.map(sns.boxplot, "time", "total_bill")

#Specify the order for plot elements:


g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
g = g.map(plt.hist, "total_bill", bins=bins, color="m")

#Use a different color palette:


kws = dict(s=50, linewidth=.5, edgecolor="w")
print(kws)

g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
                  hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#Use a dictionary mapping hue levels to colors:


pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
                  hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#Additionally use a different marker for the hue levels:


g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
                  hue_order=["Dinner", "Lunch"],
                  hue_kws=dict(marker=["^", "v"]))
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#"Wrap" a column variable with many levels into the rows:


attend = sns.load_dataset("attention")
g = sns.FacetGrid(attend, col="subject", col_wrap=5,
                  size=1.5, ylim=(0, 10))
g = g.map(sns.pointplot, "solutions", "score", scale=.7)

#Define a custom bivariate function to map onto the grid:


from scipy import stats
def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)
g = sns.FacetGrid(tips, col="smoker", hue="sex")
g = (g.map(qqplot, "total_bill", "tip", **kws)
      .add_legend())

#Define a custom function that uses a ``DataFrame`` object and accepts
#column names as positional variables:


import pandas as pd
df = pd.DataFrame(
    data=np.random.randn(90, 4),
    columns=pd.Series(list("ABCD"), name="walk"),
    index=pd.date_range("2015-01-01", "2015-03-31",
                        name="date"))

print(df)
a = pd.Series(list("ABCD"), name="walk")
print(a)




df = df.cumsum(axis=0).stack().reset_index(name="val")
def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
g = sns.FacetGrid(df, col="walk", col_wrap=2, size=3.5)
g = g.map_dataframe(dateplot, "date", "val")

#Use different axes labels after plotting:


g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="g", **kws)
      .set_axis_labels("Total bill (US Dollars)", "Tip"))

#Set other attributes that are shared across the facetes:


g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="r", **kws)
      .set(xlim=(0, 60), ylim=(0, 12),
           xticks=[10, 30, 50], yticks=[2, 6, 10]))

#Use a different template for the facet titles:


g = sns.FacetGrid(tips, col="size", col_wrap=3)
g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c")
      .set_titles("{col_name} diners"))

#Tighten the facets:


g = sns.FacetGrid(tips, col="smoker", row="sex",
                  margin_titles=True)
g = (g.map(plt.scatter, "total_bill", "tip", color="m", **kws)
      .set(xlim=(0, 60), ylim=(0, 12),
           xticks=[10, 30, 50], yticks=[2, 6, 10])
      .fig.subplots_adjust(wspace=.05, hspace=.05))




      
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import pandas as pd


ts1 = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts1.cumsum()
print(ts)
print(ts1)
ts.plot()      

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
print(df)
plt.figure(); df.plot();

df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
print(df3[:10])
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y='B')

plt.figure();
df.iloc[5].plot(kind='bar'); plt.axhline(0, color='k')

print(df.iloc[5])

df = pd.DataFrame()
df.plot#.<TAB>
#df.plot.area     df.plot.barh     df.plot.density  df.plot.hist     df.plot.line     df.plot.scatter
#df.plot.bar      df.plot.box      df.plot.hexbin   df.plot.kde      df.plot.pie

plt.figure();
df.iloc[5].plot.bar(); plt.axhline(0, color='k')


df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
print(df2)
df2.plot.bar();

df2.plot.bar(stacked=True);
df2.plot.barh(stacked=True);


df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

plt.figure();
df4.plot.hist(alpha=0.5)

plt.figure();
df4.plot.hist(stacked=True, bins=20)

#For further graphs refer: https://pandas.pydata.org/pandas-docs/stable/visualization.html
df_temp = pd.DataFrame(np.random.rand(10,3), columns = list('abc'))
print(df_temp)

np.random.seed(12345)
z = list(np.squeeze((np.random.rand(9,1)*10).astype(int)))
print(z)
dict_temp = {'a':z, 'b':list(np.arange(1,10,1)), 'c':list(np.arange(1,10,1))}
print(dict_temp)
df_temp = pd.DataFrame(dict_temp, columns = list('abc')).reset_index()
df_temp.index = df_temp['b']
print(df_temp)

z = list(df_temp.a) + list(df_temp.b)
print(z, type(z))

z = df_temp.a + df_temp.b
print(z, type(z))

z = np.array(list(df_temp.a) + list(df_temp.b))
print(z, type(z))

print(np.percentile(z, 0.5))

df = df_temp.loc[(df_temp['a'] > df_temp['b']), ['c', 'b']]
print(df)

df1 = df_temp.loc[5:, ['c', 'a']]
print(df1)

df2 = df_temp[(df_temp['a'] > df_temp['b'])]
print(df2)

df_temp['new col'] = df_temp['a'].apply(lambda x: x**2)
print(df_temp)




### MATPLOTLIB END

"""

TENSORFLOW START

"""


import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)


result = tf.multiply(x1,x2)
print(result)

sess = tf.Session()
print(sess.run(result))

sess.close()

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)    

### TENSORFLOW END




# Generate a sound
import numpy as np
framerate = 44100
t = np.linspace(0,5,framerate*5)
data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
Audio(data,rate=framerate)

# Can also do stereo or more channels
dataleft = np.sin(2*np.pi*220*t)
dataright = np.sin(2*np.pi*224*t)
Audio([dataleft, dataright],rate=framerate)

Audio("http://www.nch.com.au/acm/8k16bitpcm.wav")  # From URL
Audio(url="http://www.w3schools.com/html/horse.ogg")

Audio('/path/to/sound.wav')  # From file
Audio(filename='/path/to/sound.ogg')

Audio(b'RAW_WAV_DATA..)  # From bytes
Audio(data=b'RAW_WAV_DATA..)





from __future__ import print_function

import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
%matplotlib inline


# Load the example track
y, sr = librosa.load(librosa.util.example_audio_file())

import sounddevice as sd
sd.play(y, sr)


# Play it back!
IPython.display.Audio(data=y, rate=sr)


# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)
sd.play(y_h, sr)


# Play the harmonic component
IPython.display.Audio(data=y_h, rate=sr)


# Play the percussive component
IPython.display.Audio(data=y_p, rate=sr)
sd.play(y_p, sr)



# Pitch shifting?  Let's gear-shift by a major third (4 semitones)
y_shift = librosa.effects.pitch_shift(y, sr, 7)
sd.play(y_shift, sr)



IPython.display.Audio(data=y_shift, rate=sr)

# Or time-stretching?  Let's slow it down
y_slow = librosa.effects.time_stretch(y, 0.5)
sd.play(y_slow, sr)



IPython.display.Audio(data=y_slow, rate=sr)

# How about something more advanced?  Let's decompose a spectrogram with NMF, and then resynthesize an individual component
D = librosa.stft(y)

# Separate the magnitude and phase
S, phase = librosa.magphase(D)

# Decompose by nmf
components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)



'''
E2E Data Science

'''

import os
import tarfile
from six.moves import urllib
import pandas as pd
from subprocess import check_output
# test os.walk

cwd = os.getcwd()
print(cwd)


check_output(["ls", cwd]).decode("utf8")
os.listdir(cwd)

p = os.walk("C:\\Ericsson\\Data")

z = [x for x in p]
print(len(z))
print(z[0])
print(z[3][:])
test = next(p)
print(test)
test = next(p)[1]
print(test)
test = next(p)[1]
print(test)
test = next(p)[1]
print(test)


# end test os.walk


CW_DIR = os.getcwd()
print(CW_DIR) 
tmp_dir = os.path.join(os.path.dirname(CW_DIR), 'test', 'test')
tmp_dir = os.path.join(CW_DIR, 'test', 'test')
print(tmp_dir)
files = os.listdir(CW_DIR)
print(files)
#convert the list into comma separated string
files1 = pd.DataFrame(files).apply(lambda x: ", ".join(x))
files2 = pd.DataFrame(files)
files3 = pd.DataFrame(files).apply(lambda x: 't')
print(files2)
print(type(files2))
print(files1)
print(type(files1))
print(type(files1[0].split(',')))

#print(files3)



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_PATH_1 = "datasets\\housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH_1):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        print(housing_path)
        print(tgz_path)
    

    
fetch_housing_data()
    
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH_1):
    csv_path = os.path.join(housing_path, "housing.csv")
    print(csv_path)
    return pd.read_csv(csv_path)    

    
housing = load_housing_data()    

print(housing.head())

housing.describe()
housing.info()

housing['median_house_value'].value_counts()
housing['ocean_proximity'].value_counts()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    print(type(train_indices), test_indices)
    return data.iloc[train_indices], data.iloc[test_indices]


a = np.random.permutation(10)
print(a)

arr = np.arange(9).reshape((3, 3))
arr1 = np.random.permutation(arr)
print(arr)
print(arr1)


train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set), "train +", len(test_set), "test")

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

np.ceil([2.15,3.04, 5.89])
np.floor([2.15,3.04, 5.89])
np.round([2.15,3.04, 5.89],1)


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(type(split))

for train_index, test_index in split.split(housing, housing["income_cat"]):
    print(train_index, test_index)
    print(type(train_index), type(test_index))
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    print(len(strat_train_set), len(strat_test_set))
    
housing["income_cat"].value_counts() / len(housing)    
strat_train_set["income_cat"].value_counts() / len(strat_train_set)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
    
housing = strat_train_set.copy()    

housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    ) 

plt.legend()

corr_matrix = housing.corr()
print(corr_matrix)
print(type(corr_matrix))
corr_matrix["median_house_value"].sort_values(ascending=False)

import seaborn as sns
fig = plt.figure()
sns.heatmap(corr_matrix)



from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

z = housing["rooms_per_household"].to_frame().sort_values(by = 'rooms_per_household')
print(z.tail())
print(type(housing["rooms_per_household"].to_frame()))
print(type(housing["rooms_per_household"]))
print(type(list(housing["rooms_per_household"].to_frame())))
print(type(list(housing["rooms_per_household"])))
z = housing['ocean_proximity'].unique()
print(z)


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

strat_train_set.shape
housing.shape
z = housing.dropna(subset=["total_bedrooms"]) # option 1
z.shape
housing.shape
w = housing.drop("total_bedrooms", axis=1) # option 2
w.shape

na_index = housing.loc[np.isnan(housing.total_bedrooms)].index
print(list(na_index))
df_tmp = housing.loc[na_index, 'total_bedrooms']
print(df_tmp.shape)
print(len(list(na_index)))


median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median, inplace=True) # option 3

na_index = housing.loc[np.isnan(housing.total_bedrooms)]
print(na_index)
print(list(na_index))


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median", verbose = 1)

housing_num = housing.drop("ocean_proximity", axis=1)

print(housing["ocean_proximity"])
print(housing_num)
imputer.fit(housing_num)

imputer.statistics_
imputer.statistics_.shape
housing_num.median().values

na_index = housing_num.loc[np.isnan(housing_num.total_bedrooms)]
print(na_index)


X = imputer.transform(housing_num)
print(sum(np.isnan(X)))

housing_tr = pd.DataFrame(X, columns=housing_num.columns)
print(housing_tr)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


print(type(housing_cat_encoded))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer


housing_cat = housing["ocean_proximity"]
housing_to_scale = housing["households"]
housing.head()
print(type(housing_cat.to_frame()))
housing_cat_pd = housing_cat.to_frame()


housing_scale_pipe = Pipeline([#('LabelEncoder', LabelEncoder()),
                             ('Scaler', StandardScaler()),])

housing_cat_pipe = Pipeline([#('selector', DataFrameSelector(housing_cat_attrib)),
                             ('LabelBinarizer', LabelBinarizer()),])


housing_cat_pipe = Pipeline([
                             ('selector', DataFrameSelector(["ocean_proximity"])),
                             ('LabelEncoder', LabelBinarizer()),])



print([item for item in dir(housing_cat_pipe) if item not in "_"])
print([item for item in dir(LabelBinarizer) if item not in "_"])
#print(housing_cat_pipe.classes_)


#housing_cat_encoded = housing_cat_pipe.fit_transform(housing_cat)
X = housing_scale_pipe.fit(np.array(housing_to_scale).reshape(-1,1))
Y = housing_scale_pipe.transform(np.array(housing_to_scale).reshape(-1,1))
print(Y[:10])
Z = housing_scale_pipe.fit_transform(np.array(housing_to_scale).reshape(-1,1))
print(type(Y))
print(Z)
print(type(Z))
print(Z.shape)
print(Y.shape)
print(Z.squeeze())
print(type(Z.squeeze()))
print(Z.squeeze().shape)
print(Y)


print(housing["households"])

lblBin = LabelBinarizer()
f = lblBin.fit(housing_cat)
print(housing_cat.values)
print(f)
f = lblBin.transform(housing_cat)
print(f)


lblEnc = LabelEncoder()
g = lblEnc.fit(housing_cat)
print(housing_cat.values)
print(g)
housing_categorical = lblEnc.transform(housing_cat)
print(housing_categorical)




housing_cat_encoded = None
housing_cat_encoded = housing_cat_pipe.fit(housing_cat)
#housing_cat_encoded = housing_cat_pipe.transform(housing_cat)
housing_cat_encoded = housing_cat_pipe.fit_transform(housing_cat)
housing_cat_encoded


print(housing_cat_pipe.classes_)


from sklearn.pipeline import FeatureUnion

housing_cat_attrib = ["ocean_proximity"]
housing_to_scale_attrib = ["households"]

housing_scale_pipe_selector = Pipeline([#('LabelEncoder', LabelEncoder()),
                             ('selector', DataFrameSelector(housing_to_scale_attrib)),
                             ('Scaler', StandardScaler()),])

#housing_cat_pipe_selector = Pipeline([
#                             ('selector', DataFrameSelector(housing_cat_attrib)),
#                             ('LabelEncoder', LabelBinarizer()),])


housing_cat_pipe_selector = Pipeline([
                             ('selector', DataFrameSelector(housing_cat_attrib)),
                             ('LabelEncoder', LabelEncoder()),])



fullpipe = FeatureUnion(transformer_list=[
                                          ('scale_pipe', housing_scale_pipe_selector), 
                                          ('cat_pipe', housing_cat_pipe_selector),
                                          ])

print(housing.columns)
Q = fullpipe.fit_transform(housing)
print(Q)
print(type(Q))


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_categorical.reshape(-1,1))
print(housing_categorical)
print(housing_cat_1hot)

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_categorical.reshape(-1,1))
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_categorical)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())



from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
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

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

print(housing_extra_attribs.shape)
print(housing.values)
print(housing.shape)
print(type(housing))
print(type(housing_extra_attribs))
print(housing_extra_attribs)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
                ('imputer', Imputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
                ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr[:10])

print(list(housing_num))


from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

print([item for item in dir(BaseEstimator) if item not in "_"])
print([item for item in dir(TransformerMixin) if item not in "_"])
print([item for item in dir(Pipeline) if item not in "_"])


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
                        ('selector', DataFrameSelector(num_attribs)),
                        ('imputer', Imputer(strategy="median")),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler()),
                        ])
cat_pipeline = Pipeline([
                        ('selector', DataFrameSelector(cat_attribs)),
                        ('label_binarizer', LabelEncoder()),
                        ])
full_pipeline = FeatureUnion(transformer_list=[
                            ("num_pipeline", num_pipeline),
                            ("cat_pipeline", cat_pipeline),
                            ])

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared
housing_prepared.shape


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))


from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
print(housing_predictions.shape)
print(housing_labels.shape)
print(type(housing_predictions))
print(type(housing_labels))
print(type(housing_labels.values))
print(housing_predictions)
print(housing_labels.values)
print(np.round(housing_predictions))

# never use below accuracy for non classification type of problems
housing_labels_np = housing_labels.values
acc = accuracy_score(housing_labels_np, np.round(housing_predictions))
print(acc*100)


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
#housing_predictions = forest_reg.predict(housing_prepared)

forest_tree_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_tree_scores)

display_scores(forest_rmse_scores)




from sklearn.model_selection import GridSearchCV
param_grid = [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
             ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

print(housing_prepared.shape[0]/5)

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_
grid_search.best_estimator

cvres = grid_search.cv_results_

print(cvres.keys())
print(cvres['split1_train_score'])
print(cvres['params'])

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
feature_importances = grid_search.best_estimator_.feature_importances_
[x for x in dir(grid_search.best_estimator_) if x not in "_"]

type(housing_prepared)
print(list(housing_num))
print(type(housing_num))
type(housing)
list(housing.columns.values)
print(type(housing.columns.values))
print(type(housing.columns))

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6
print(final_rmse)



''''
CLASSIFICATION

''''

from sklearn.datasets import fetch_mldata
from keras.datasets import mnist
import numpy as np
import pandas as pd

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X, y = mnist["data"], mnist["target"]

print(X_train.shape)
print(X_train[:2,:5, :5])

a = X_train[:2,:5, :5]
print(a)
c = np.reshape(a, [2,-1])
print(c)

d = [[1],[2],[3]]
print(np.squeeze(d))

e = [[1,2],[2,3],[3,4]]
np.array(e).shape
print(len(e))
print(np.squeeze(e))
print(e)
print(type(np.squeeze(e)))
print(type(e))
print(np.squeeze(e).shape)


x = pd.DataFrame(X_train.reshape([len(X_train),-1] ))
print(x.shape)
w = np.array(x).reshape([len(x), 28,28])
print(w.shape)
print(X_train.shape)

print(type(X_train))
x_train = pd.DataFrame(X_train.reshape([len(X_train), -1]))
x_test = pd.DataFrame(X_test.reshape([len(X_test), -1]))

print(x_train.columns.values)

y_train_pd = pd.DataFrame(y_train.reshape([len(y_train), -1]))
y_test_pd = pd.DataFrame(y_test.reshape([len(y_test), -1]))



X = pd.concat([x_train, x_test], axis = 0)
y = np.squeeze(pd.concat([y_train_pd, y_test_pd], axis = 0))
print(y[:10])

#y1 = pd.concat([y_train_pd, y_test_pd], axis = 0)
#print(y1[:10])



print(X.shape)
print(y.shape)

X = np.array(X)
y = np.array(y)

import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = 'binary', interpolation="nearest")
plt.axis("off")
plt.show()

y[36000]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


shuffle_index = np.random.permutation(60000)
print(shuffle_index)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

print(y_train_5.shape)
print(y_test_5.shape)



print(y_train_5[y_train_5 == True].shape)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])



from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
    

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")        

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print(y_train_pred.shape)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)


threshold = 200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

print(y_scores.shape)
print(y_scores[:10])

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

print(precisions.shape, recalls.shape, thresholds.shape)

print(precisions[:-1].shape)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()


roc_auc_score(y_train_5, y_scores_forest)


some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
np.argmax(some_digit_scores)
sgd_clf.classes_
sgd_clf.classes_[True]

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

forest_clf.predict_proba([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
sns.heatmap(np.round(conf_mx,2))


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
print(row_sums)
print(np.round(norm_conf_mx,2))

import seaborn as sns
sns.heatmap(np.round(norm_conf_mx,2))


np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()




### CV2 THRESHOLDING START



import numpy as np
import cv2
# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)
print(img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('messigray.png',img)


import numpy as np
import cv2
img = cv2.imread('messi5.jpg',cv2.IMREAD_UNCHANGED)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()

    
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg',cv2.IMREAD_COLOR)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()    



import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('img', img)

img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.imshow('img', img)

img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
cv2.imshow('img', img)

img = cv2.ellipse(img,(256,256),(100,50),0,0,360,(100,200,100),-1)
cv2.imshow('img', img)


pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
cv2.imshow('img', img)


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('img', img)

import cv2
import numpy as np
img = cv2.imread('messi5.jpg')

cv2.imshow('img', img)


px = img[200,210]
print (px)
#[157 166 200]
# accessing only blue pixel
blue = img[100,100,0]
print (blue)
#157

img[100,100] = [255,255,255]
print (img[100,100])
#[255 255 255]


print (img.shape)
print (img.size)
print (img.dtype)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

cv2.imshow('img1', img)





import cv2
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv2.imread('opencv_logo.png')
replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()




# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('img', mask)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow('img', mask_inv)


# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('img', img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow('img', img2_fg)


# Put logo in ROI and modify the main image
#dst = cv2.add(img1_bg,img2_fg)
dst = cv2.add(img1,img1_bg)

img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)



import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()








import cv2
import numpy as np

imgEllipse = cv2.imread("Ellipse.jpg", cv2.IMREAD_COLOR)
imgRectangle = cv2.imread("Rectangle.jpg", cv2.IMREAD_COLOR)
imgLine = cv2.imread("Line.jpg", cv2.IMREAD_COLOR)

cv2.imshow('img_e', imgEllipse)
cv2.imshow('img_r', imgRectangle)
cv2.imshow('img_l', imgLine)

imgEllipseGray = cv2.cvtColor(imgEllipse, cv2.COLOR_BGR2GRAY)
imgRectangleGray = cv2.cvtColor(imgRectangle, cv2.COLOR_BGR2GRAY)
imgLineGray = cv2.cvtColor(imgLine, cv2.COLOR_BGR2GRAY)

cv2.imshow('img_eg', imgEllipseGray)
cv2.imshow('img_rg', imgRectangleGray)
cv2.imshow('img_lg', imgLineGray)


ret, mask_ell = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img_em', mask_ell)


ret, mask_rect = cv2.threshold(imgRectangleGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img_rm', mask_rect)


mask_ell_not = cv2.bitwise_not(mask_ell)
cv2.imshow('img_em_not', mask_ell_not)


mask_rect_not = cv2.bitwise_not(mask_rect)
cv2.imshow('img_rm_not', mask_rect_not)


mask_and = cv2.bitwise_and(mask_ell, mask_rect)
cv2.imshow('img_and', mask_and)


mask_or = cv2.bitwise_or(mask_ell, mask_rect)
cv2.imshow('img_or', mask_or)


mask_or_of_nots = cv2.bitwise_or(mask_ell_not, mask_rect_not)
cv2.imshow('img_or_of_nots', mask_or_of_nots)

mask_and_of_nots = cv2.bitwise_and(mask_ell_not, mask_rect_not)
cv2.imshow('img_and_of_nots', mask_and_of_nots)


mask_xor_of_nots = cv2.bitwise_xor(mask_ell_not, mask_rect_not)
cv2.imshow('img_xor_of_nots', mask_xor_of_nots)



ell_masked_and = cv2.bitwise_and(imgEllipse, imgEllipse, mask = mask_and_of_nots)
cv2.imshow('ell_masked_and', ell_masked_and)

ell_masked_or = cv2.bitwise_or(imgEllipse, imgEllipse, mask = mask_or_of_nots)
cv2.imshow('ell_masked_or', ell_masked_or)



print(imgEllipseGray < 255)
cv2.imshow('img3', imgEllipseGray)


print(imgEllipseGray[200,200])
print(imgEllipseGray[250,200])



ret, mask1 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img1', mask1)

ret, mask2 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_MASK)
cv2.imshow('img2', mask2)


ret, mask3 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('img3', mask3)

ret, mask4 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_TRIANGLE)
cv2.imshow('img4', mask4)



#cv2.pixel(imgEllipseGray, [10,10])

mask_inv = cv2.bitwise_not(imgEllipse)
cv2.imshow('img5', mask_inv)

mask_inv = cv2.bitwise_and(mask1, imgEllipse)
cv2.imshow('img5_1', mask_inv)





img1_and = cv2.bitwise_and(imgEllipse,imgRectangle, mask = mask_inv)
cv2.imshow('img6', img1_and)

img1_or = cv2.bitwise_or(imgEllipse,imgRectangle)
cv2.imshow('img7', img1_or)



import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('circle.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.png')

print(img1.shape)
print(img2.shape)

img1 = img1[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]]
img2 = img2[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]]

print(img1.shape)
print(img2.shape)

dst = cv2.addWeighted(img1,0.2,img2,0.7,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()







import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bw.png',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()





import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('noisy6.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)


# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(th2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian Blurred Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()







import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bimodal_hsv_noise.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()





from skimage import data , io, filter
image = data.coins() # or any NumPy a r r a y !
edges = filter.sobel(image)
io.imshow(edges)



import numpy as np
import matplotlib.pyplot as plt
# Load a sma l l s e c t i o n o f t h e image .
image = data.coins()[0:95, 70:370]
fig, axes = plt.subplots(ncols=2, nrows=3,
figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat
ax0.imshow(image , cmap=plt.cm.gray)
ax0.set_title('Original', fontsize=24)
ax0.axis('off')


# Hi s togram .
values , bins = np.histogram(image ,bins=np.arange(256))
ax1.plot(bins[:-1], values , lw=2, c='k')
ax1.set_xlim(xmax=256)
ax1.set_yticks([0, 400])
ax1.set_aspect(.2)
ax1.set_title('Histogram', fontsize=24)


# Appl y t h r e s h o l d .
from skimage.filter import threshold_adaptive
bw = threshold_adaptive(image , 95, offset=−15)
ax2.imshow(bw, cmap=plt.cm.gray)
ax2.set_title(’Adaptive threshold’, fontsize=24)
ax2.axis(’off’)


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
coins = data.coins()
histo, bins = np.histogram(coins, bins=np.arange(0, 256))
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(221)
ax.imshow(coins)

ax = fig.add_subplot(222)

ax.plot(bins[:-1], histo, lw=2, c='k')



from skimage.feature import canny
edges = canny(coins/255.)



from skimage.feature import canny
edges = canny(coins/255.)

ax = fig.add_subplot(223)
ax.imshow(edges)



from skimage import data , io, filter
image = data.coins() # or any NumPy a r r a y !
edges = filter.sobel(image)
io.imshow(edges)


import numpy as np
import matplotlib.pyplot as plt
from skimage import data
# Load a sma l l s e c t i o n o f t h e image .
image = data.coins()[0:95, 70:370]
fig, axes = plt.subplots(ncols=2, nrows=3,
figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat
ax0.imshow(image , cmap=plt.cm.gray)
ax0.set_title('Original', fontsize=24)
ax0.axis('off')


# Hi s togram .
values , bins = np.histogram(image, bins=np.arange(256))
ax1.plot(bins[:-1], values , lw=2, c='k')
ax1.set_xlim(xmax=256)
ax1.set_yticks([0, 400])
ax1.set_aspect(.2)
ax1.set_title('Histogram', fontsize=24)


from skimage.filter import threshold_adaptive
bw = threshold_adaptive(image , 95, offset=-15)
ax2.imshow(bw, cmap=plt.cm.gray)
ax2.set_title('Adaptive threshold', fontsize=24)
ax2.axis('off')


# Find maxima .
from skimage.feature import peak_local_max
coordinates = peak_local_max(image , min_distance=20)
print(coordinates)
ax3.imshow(image , cmap=plt.cm.gray)
ax3.autoscale(False)
#ax3.plot(coordinates[:, 1], coordinates[:, 0], c='r.')
ax3.scatter(coordinates[:, 1], coordinates[:, 0], c='r')
ax3.set_title('Peak local maxima', fontsize=24)
ax3.axis('off')


# De t e c t edge s .
from skimage import filter
edges = filter.canny(image , sigma=3, low_threshold=10, high_threshold=80)
ax4.imshow(edges , cmap=plt.cm.gray)
ax4.set_title('Edges', fontsize=24)
ax4.axis('off')


# Labe l image r e g i o n s .
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label
label_image = label(edges)
ax5.imshow(image , cmap=plt.cm.gray)
ax5.set_title('Labeled items', fontsize=24)
ax5.axis('off')

print(type(label_image))

for region in regionprops(label_image):
# Draw r e c t a n g l e around s egment ed c o i n s .
    print(region.bbox)
    minr , minc , maxr , maxc = region.bbox
    rect = mpatches.Rectangle((minc , minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax5.add_patch(rect)
plt.tight_layout()
plt.show()



from skimage import io
ic = io.ImageCollection('C:\\Users\\Public\\Pictures\\Sample Pictures\\*')

from skimage.color import rgb2gray
from skimage import transform
image0 = rgb2gray(ic[0][:, 500:500+1987, :])
image1 = rgb2gray(ic[1][:, 500:500+1987, :])
image0 = transform.rescale(image0 , 0.25)
image1 = transform.rescale(image1 , 0.25)


'''
fig, ax = plt.subplots(2,2, figsize=(8,8))
axes = ax.flat
axes[0].imshow(image0)
axes[1].imshow(image1)
'''


from skimage.feature import ORB, match_descriptors
orb = ORB(n_keypoints=1000, fast_threshold =0.05)
orb.detect_and_extract(image0)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors
orb.detect_and_extract(image1)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors
matches12 = match_descriptors(descriptors1 ,descriptors2 ,cross_check=True)

from skimage.measure import ransac
# S e l e c t k e y p o i n t s f rom t h e s o u r c e ( image t o be
# r e g i s t e r e d ) and t a r g e t ( r e f e r e n c e image ) .
src = keypoints2[matches12[:, 1]][:, ::-1]
dst = keypoints1[matches12[:, 0]][:, ::-1]
model_robust , inliers = ransac((src, dst), ProjectiveTransform ,
                                min_samples=4, residual_threshold=2)



r, c = image1.shape[:2]
# Note t h a t t r a n s f o rma t i o n s t a k e c o o r d i n a t e s i n
# ( x , y ) format , n o t ( row , column ) , i n o r d e r t o be
# c o n s i s t e n t wi t h mos t l i t e r a t u r e .
corners = np.array([[0, 0],[0, r],[c, 0],[c, r]])
# Warp t h e image c o r n e r s t o t h e i r new p o s i t i o n s .
warped_corners = model_robust(corners)
# Find t h e e x t e n t s o f both t h e r e f e r e n c e image and
# t h e warped t a r g e t image .
all_corners = np.vstack((warped_corners , corners))
corner_min = np.min(all_corners , axis=0)
corner_max = np.max(all_corners , axis=0)
output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])


from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
offset = SimilarityTransform(translation=-corner_min)
image0_ = warp(image0 , offset.inverse , output_shape=output_shape , cval=-1)
image1_ = warp(image1 , (model_robust + offset).inverse , output_shape=output_shape, cval=-1)



def add_alpha(image , background=-1):
    # Add an alpha l a y e r t o t h e image .
    #The alpha l a y e r i s s e t t o 1 f o r f o r e g r o u n d
    #and 0 f o r bac kground .
    
    rgb = gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))

image0_alpha = add_alpha(image0_)
image1_alpha = add_alpha(image1_)
merged = (image0_alpha + image1_alpha)
alpha = merged[..., 3]





### CV2 THRESHOLDING END


from bokeh.io import output_file, show, output_notebook



##########################

## CountVectorizer example with ngrams

##########################

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(ngram_range=(1, 3))
print(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)
