# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
iris = load_iris()
type(iris)

print (iris.data[0:5])

print (iris.data.shape)
print (iris.data)
print (iris.feature_names)

print (iris.target)
print (iris.target.shape)

print(iris.target_names)

X = iris.data
print (X.shape)

#X.head

y=iris.target

print(y.shape)

#print(y.type)

print (iris.data.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

print (knn)

knn.fit(X,y)

knn.predict([3,5,4,2])

print(y)

X_new=([3,5,4,2],[5,4,3,2])
knn.predict(X_new)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

print (logreg)

logreg.fit(X,y)

logreg.predict([3,5,4,2])

print(y)

X_new=([3,5,4,2],[5,4,3,2])
y_pred = logreg.predict(X)

print (len(y_pred))
print(y_pred)


from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))

print(type(X_new))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
print (knn)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

import matplotlib.pyplot as plt

plt.scatter(y_pred, y, color = 'red', marker = 'o', linewidths = 10)


print (X.shape)
print (y.shape)


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)

print (X_train.shape)
print (y_train.shape)

print (X_test.shape)
print (y_test.shape)

knn.fit (X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))


k_range = range(1, 26)
print([x for x in k_range])
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit (X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append (metrics.accuracy_score(y_test, y_pred))
#    print (scores)
    
print (scores)

import matplotlib.pyplot as plt

plt.plot(k_range, scores)    
plt.xlabel('value of k in knn')
plt.ylabel('testing accuracy')

import pandas as pd


data = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Advertising.csv', index_col=0)
#data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

#print(pd.read_csv)

data.head()

print(type(data))

print(data.shape)

data.tail()


import seaborn as sns

sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars=['Sales'], size=7, aspect=0.7, kind='reg')


feature_cols = ['TV','Radio','Newspaper']

X = data[feature_cols]

print(X.head())

y = data['Sales']

print(y.head())

print(X.shape)

print(y.shape)

print(type(y))

from sklearn.cross_validation import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)


y_pred = linreg.predict(X_test)

from sklearn import metrics

import numpy as np

print(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))

zip(feature_cols, linreg.coef_)

print(linreg.coef_)

from sklearn.cross_validation import KFold
kf = KFold(25, n_folds = 5, shuffle=False)

print(type(kf))

print('{}{:^61}{}'.format('Iteration','Training Set Observations','Test Set Observations'))

print(data)

for iteration, data in enumerate(kf, start=1):
    #print(data[0])
    #print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))
    print('{:^9}{}{:}'.format(iteration, data[0], data[1]))


    
    
    
from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

print(y)
print(type(X_train))
X_train1 = np.array(X_train).astype(int)
y_train1 = np.array(y_train).astype(int)
print(X_train1)

print(y.head())

scores = cross_val_score(knn, X_train1, y_train1, cv=10, scoring='accuracy')

print(scores)
print(scores.mean())

k_range = range(1,31)
k_scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())
    
print (k_scores)

import matplotlib.pyplot as plt

plt.plot(k_range, k_scores)
plt.xlabel('value of k for knn')
plt.ylabel('cross val accuracy')

knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn, X, y, cv=10, scoring = 'accuracy')
print (scores.mean())
print (scores)

logreg = LogisticRegression()
scores = cross_val_score(logreg, X, y, cv=10, scoring = 'mean_squared_error')
print (scores.mean())
print (scores)

"""
Cross Val Score - some reason linreg is not working below
"""


from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as LinearRegression

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

feature_cols = ['TV','Radio','Newspaper']

X = data[feature_cols]

y = data.Sales

print(X.head())
print(y.head())

linreg = LinearRegression.LinearRegression()
scores = None
scores = cross_val_score(linreg, X, y, cv = 10) #, scoring = 'mean_squared_error')
print(scores)

mse_scores = -scores
print(mse_scores)
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores.mean())


feature_cols=['TV','Radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(linreg, X, y, cv = 10, scoring = 'mean_squared_error')).mean())


"""
Showcase of Cross Val Score
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
scores1 = cross_val_predict(knn, X, y, cv = 10)
print (scores)
print (scores1.shape)
print (X.shape)
print (scores.mean())

k_range = range(1,31)
k_scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    print(scores)
    k_scores.append(scores.mean())
    
print (k_scores)

plt.plot(k_range, k_scores)
plt.xlabel('no. of neighbors')
plt.ylabel('accuracy')

print(type(scores))
print(type(k_scores))
print(type(k_range))

from sklearn.grid_search import GridSearchCV

k_range = list(range(1,31))

z = range(1,10,2)

print(z)

for i in k_range:
    print(i)

print(k_range)
print (type(k_range))
weight_options1 = ['uniform','distance']
param_grid = dict(n_neighbors = k_range, weights = weight_options1)

print(param_grid)
print(type(param_grid))

grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'accuracy')

print(type(grid))

grid.fit(X, y)

print(grid.grid_scores_)

print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

print(grid_mean_scores)

plt.plot(k_range*2, grid_mean_scores)
plt.xlabel('no. of neighbors')
plt.ylabel('grid mean score')

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

grid.predict([3,4,5,6])

"""

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
predict probability of occurrence of diabetes in below program

"""

import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedegree', 'age', 'label']
pima = pd.read_csv(url, header = None, names = col_names)

print (pima.shape)

pima.head()
pima.tail()

col_names1=['pregnant','insulin', 'bmi', 'age']
X = pima[col_names1]
X.head()

y=pima.label

y.head()

from sklearn.cross_validation import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train.head()
y_train.head()

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred=logreg.predict(X_test)

print(y_pred)

print(y_pred.shape)

score = metrics.accuracy_score(y_test, y_pred)
print(score)

print(type(y_test))

print(y_test.value_counts())

y_test.mean()

1-y_test.mean()

print(y_pred[0:25])
print(list(y_test[0:25]))


print(metrics.confusion_matrix(y_test, y_pred))

confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)

logreg.predict(X_test)[0:10]
logreg.predict_proba(X_test)[0:10]

logreg.predict_proba(X_test)[0:10,0]

logreg.predict_proba(X_test)[0:10,1]

y_pred_prob=logreg.predict_proba(X_test)[:,1]

print(y_pred_prob)
y_pred_prob.shape
import matplotlib.pyplot as plt

plt.rcParams['font.size'] =14


plt.hist(y_pred_prob,bins=8)

from sklearn.preprocessing import binarize

y_pred_class = binarize(y_pred_prob, 0.3)[0]
print (y_pred_prob[:10])
print (y_pred_class[:10])

print (confusion)
print(metrics.confusion_matrix(y_test, y_pred_class))
y_pred_class.shape
