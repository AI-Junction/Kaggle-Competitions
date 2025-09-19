# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:36:30 2017

@author: Chandrakant Pattekar
"""

#%%
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import preprocessing, cross_validation
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import seaborn as sns

#%%

# Read CSV files to fetch train, test data
df1 = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\train.csv')
df2 = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\test.csv')
print(df1.head())
print(df2.head())
#print(df.columns.values)



#%%

#print(set(df['Survived'].values.tolist()))

#print(df.head())
df1.convert_objects(convert_numeric=True)
df1.fillna(0, inplace = True)
print(df1.head())


df2.convert_objects(convert_numeric=True)
df2.fillna(0, inplace = True)

#print(df.head())

#%%

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
            
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            unique_elements = set(df[column].values.tolist())
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x +=1
            
            df[column] = list(map(convert_to_int, df[column])) 
            
            
    
    return df


#%%

print(df1.columns.values)

df1a = handle_non_numerical_data(df1)
df2a = handle_non_numerical_data(df2)
#print(df2.columns.values)
#print (np.array(df2).shape)

#print(df2.head())
#print(df.head())

# drop 'Survived' column from training data
# drop 'PassengerId' column as it has low covariance in predicting survivors

X_train = np.array(df1a.drop(['Survived'], 1).astype(float))
X_train = np.array(df1a.drop(['PassengerId'], 1).astype(float))
X_train = np.array(df1a.drop(['Ticket'], 1).astype(float))
X_train = np.array(df1a.drop(['Name'], 1).astype(float))


# process training data to scale
X_train = preprocessing.scale(X_train)

# create training validation data for training
y_train = np.array(df1a['Survived'].astype(int))


# use KMeans for training
#clf = KMeans(n_clusters = 2)
#clf.fit(X_train)

# use LogisticRegression for training

clf = LogisticRegression()
clf.fit(X_train, y_train)


# use SVM for training
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=10)
#clf = clf.fit(X_train, y_train)

X_test = np.array(df2a.drop(['PassengerId'], 1).astype(float))
X_test = np.array(df2a.drop(['Ticket'], 1).astype(float))
X_test = np.array(df2a.drop(['Name'], 1).astype(float))

X_test = np.array(df2a.astype(float))
X_test = preprocessing.scale(X_test)

#X_test = np.array(df2.drop(['Survived'], 1).astype(float))

# declare an empty list for capturing predicted results
predict_test = []

i = 0
for i, test_features in enumerate(X_test):
    # reshape to list of list for prediction
    test_features_reshaped = test_features.reshape(-1,len(test_features))
    z = clf.predict(test_features_reshaped)
    print(z)
    predict_test.append(z)

    
# count no. of predicted survivors vs age using dictionaries
count = 0
AgeVsSurvived = {}
NoOfPassengersOfGivenAge = {}
AgeUniqueVals = set(df2['Age'].values.tolist())
print(max(AgeUniqueVals))


# populate indexes for the two dictionaries with same set of keys (ages)
for i in range(int(max(AgeUniqueVals))+1):
    if i not in NoOfPassengersOfGivenAge:
        AgeVsSurvived[int(i)] = 0
        NoOfPassengersOfGivenAge[int(i)] = 0
        
for i in df2['Age']:
    NoOfPassengersOfGivenAge[int(i)] += 1

                             
                             
print(NoOfPassengersOfGivenAge)                             

# check that we go no. of passengers right in our dictionary
sumofpassengers = 0
for i in NoOfPassengersOfGivenAge:
    sumofpassengers += NoOfPassengersOfGivenAge[i]


i = 0    
age = []

# capture the predicted survivors against the 'age' index in the AgeVsSurvived dict
for i,j in enumerate(predict_test):
    AgeVsSurvived[int(df2['Age'][i])] += int(predict_test[i])

print(sum(predict_test))    
print(sumofpassengers)

# Plot age vs suvivors as line chart
AgeList = []
PredictedSurvivorsByAge = []
SurvivorsAgeList = []
ListofNoOfPassengersbyAge=[]
for i in AgeVsSurvived:
    AgeList.append(i)
    PredictedSurvivorsByAge.append(AgeVsSurvived[i])
    ListofNoOfPassengersbyAge.append(NoOfPassengersOfGivenAge[i])


print(NoOfPassengersOfGivenAge)    

print(AgeVsSurvived)    
    
width = 0.50       # the width of the bars
fig = plt.figure()
ax = plt.subplot2grid((1,10), (0,0), rowspan = 1, colspan = 10)
#rects1 = ax.bar(AgeList, SurvivorsByAgeList, width, color='r')
line = ax.plot(AgeList, PredictedSurvivorsByAge, color='r', label = 'Survived')
line2 = ax.plot(AgeList, ListofNoOfPassengersbyAge, color='g', label = 'Total Passengers')


# add some text for labels, title and axes ticks
ax.set_ylabel('No of people')
ax.set_xlabel('Age')
ax.set_title('Predicted Survivors by age')
ax.set_xticks(AgeList)
ax.set_xticklabels(AgeList)
plt.subplots_adjust(left=0.09, bottom= 0.18, right = 0.94, top = 0.85, wspace = 0.2, hspace = 0)
plt.legend()
plt.show()

print(predict_test)
predicted_result = []
for i,j in enumerate(predict_test):
    predicted_result.append([int(df2['PassengerId'][i]), int(j)])
    

print(type(predicted_result))        
print(np.array(predicted_result)[:,(0)])    

result = pd.DataFrame({
        "PassengerId": np.array(predicted_result)[:,(0)],
        "Survived": np.array(predicted_result)[:,(1)]
    })
result.to_csv('titanic_result.csv', index=False)



#%%

# Creating some trend graphs using training data itself

count_trn = 0
AgeVsSurvived_trn = {}
NoOfPassengersOfGivenAge_trn = {}
AgeUniqueVals_trn = set(df1['Age'].values.tolist())
print(max(AgeUniqueVals_trn))

print(len(X_train))

# populate indexes for the two dictionaries with same set of keys (ages)
for i in range(int(max(AgeUniqueVals_trn))+1):
    if i not in NoOfPassengersOfGivenAge_trn:
        AgeVsSurvived_trn[int(i)] = 0
        NoOfPassengersOfGivenAge_trn[int(i)] = 0
        
print(NoOfPassengersOfGivenAge_trn)
                                     
for i in df1['Age'].astype(int):
    NoOfPassengersOfGivenAge_trn[int(i)] += 1

                                 
# check that we go no. of passengers right in our dictionary
sumofpassengers_trn = 0
for i in NoOfPassengersOfGivenAge_trn:
    sumofpassengers_trn += NoOfPassengersOfGivenAge_trn[i]


print(sumofpassengers_trn)

i = 0    
age_trn = []

# capture the no. of survivors against the 'age' index in the AgeVsSurvived dict
for i,j in enumerate(df1['Age'].astype(int)):
    AgeVsSurvived_trn[j] += int(df1['Survived'][i])

print(AgeVsSurvived_trn)

# Plot age vs suvivors as line chart
AgeList_trn = []
SurvivorsByAge_trn = []
SurvivorsAgeList_trn = []
ListofNoOfPassengersbyAge_trn=[]
for i in AgeVsSurvived_trn:
    AgeList_trn.append(i)
    SurvivorsByAge_trn.append(AgeVsSurvived_trn[i])
    ListofNoOfPassengersbyAge_trn.append(NoOfPassengersOfGivenAge_trn[i])

print(ListofNoOfPassengersbyAge_trn)

width = 0.50       # the width of the bars
fig_trn = plt.figure()
ax_trn = plt.subplot2grid((1,10), (0,0), rowspan = 1, colspan = 10)
#rects1 = ax.bar(AgeList, SurvivorsByAgeList, width, color='r')
line1_trn = ax_trn.plot(AgeList_trn, SurvivorsByAge_trn, color='r', label = 'Survived')
line2_trn = ax_trn.plot(AgeList_trn, ListofNoOfPassengersbyAge_trn, color='g', label = 'Total Passengers')


# add some text for labels, title and axes ticks
ax_trn.set_ylabel('No of people')
ax_trn.set_xlabel('Age')
ax_trn.set_title('Survivors by age in training set')
ax_trn.set_xticks(AgeList_trn)
ax_trn.set_xticklabels(AgeList_trn)
plt.subplots_adjust(left=0.09, bottom= 0.18, right = 0.94, top = 0.85, wspace = 0.2, hspace = 0)
plt.legend()
plt.show()
    


#%%

# Test prediction accuracy on the training set itself
       
correct = 0

print(range(len(X_train)))

for i in range(len(X_train)):
    predict_me = np.array(X_train[i]).astype(float)
    #print('now preparing predict_me i = ' + str(i))
    #print(predict_me)
    predict_me = predict_me.reshape(-1, len(predict_me))

    #print('now reshaping predict_me i = ' + str(i))
    #print(predict_me)

    prediction = clf.predict(predict_me)
    
    #print('now check y[i] = ' )
    #print(y[i])
    print('prediction', prediction, df1['Survived'][i])
    
    if prediction==df1['Survived'][i]:
        correct += 1
        
print ('Accuracy of prediction over the training set = ', correct/len(X_train), correct)


correlation = df1.corr()
print(correlation)
fig , ax = plt.subplots( figsize =( 12 , 12) )
cmap = sns.diverging_palette( 255 , 0 , as_cmap = True )
fig = sns.heatmap(correlation,cmap = cmap, square=True,cbar_kws={ 'shrink' : .9 },ax=ax,annot = True,annot_kws = { 'fontsize' : 10 })

pclass = pd.get_dummies( df1.Pclass , prefix='Pclass' )
pclass.head()

embarked = pd.get_dummies( df1.Embarked , prefix='Embarked' )
embarked.head()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    
plot_distribution( df1 , var = 'Age' , target = 'Survived' , row = 'Sex' )

row = 'Age'
col = None
facet = sns.FacetGrid( df1 , hue='Survived' , aspect=4 , row = row , col = col )
facet.map( sns.kdeplot , 'Age' , shade= True )
facet.set( xlim=( 0 , df1[ 'Age' ].max() ) )
facet.add_legend()



#############################################
##                                         ##
##      Titanic Kernel Showcase            ##
##                                         ##
#############################################

#%%
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

#%%

# 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


# Load in the train and test datasets
train = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\train.csv')
test = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\test.csv')

#%%
print(train.columns.values)
train.head()

# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# Continous: Age, Fare. Discrete: SibSp, Parch.
# Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

train.tail()
train.shape
type(train)
type(np.array(train))



'''
Which features contain blank, null or empty values?

These will require correcting.

    Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
    Cabin > Age are incomplete in case of test dataset.

    
What are the data types for various features?

Helping us during converting goal.

    Seven features are integer or floats. Six in case of test dataset.
    Five features are strings (object).

    
'''

print(type(train.dtypes))

df_dtype = train.dtypes.to_frame()
type(df_dtype)
df_dtype.columns = ['type']
print(type(df_dtype))
print(df_dtype)
df_dtype_grp = df_dtype.groupby(['type'])['type'].aggregate('count')
print(df_dtype_grp )


train.info()
print('_'*40)
test.info()

train.describe()
train.describe(include=['O'])


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)['Survived'].aggregate('mean')

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();



# grid = sns.FacetGrid(train, col='Embarked')
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()



# Store our passenger ID for easy access
PassengerId = test['PassengerId']

train.head(3)

# Feature Engineering


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

np.isnan(train['Cabin'][0])



# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print(train['CategoricalAge'][:5])

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    
dataset['Fare']
#print(index)

    
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


g = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked',
       'FamilySize', 'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


'''

Model, predict and solve

Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:

    Logistic Regression
    KNN or k-Nearest Neighbors
    Support Vector Machines
    Naive Bayes classifier
    Decision Tree
    Random Forrest
    Perceptron
    Artificial neural network
    RVM or Relevance Vector Machine

'''







# Take Reusable components from below


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import sys
import glob
import os as os
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

print(os.getcwd(), "---", os.curdir)
print(check_output(["ls", os.getcwd()]).decode("utf8"))

train = pd.read_csv(os.path.join(os.getcwd(), "AllTitanicData", "train.csv"))
test = pd.read_csv(os.path.join(os.getcwd(), "AllTitanicData", "test.csv"))

train.head()
test.head()

#train = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\train.csv')
#test = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\test.csv')

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
                
def univariateplots(df, attribs, typeofplot):
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
    plt.show()


def jointplots(df, attribpairs, plottypes):
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
    plt.show()                
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
    sorted_columns_dict = {}
    for col in columnstosort:
        sorted_columns_dict[col] = df[col].sort_values(ascending = True)
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
    
print(train.shape, "\t", test.shape)
print(train.dtypes, "\n\n", test.dtypes)
print(train.head(), "\n\n", test.head())

sorted_cols_dict = sortcolumns(train, ['Fare', 'Age'])
print(sorted_cols_dict['Fare'][:10])
print(pd.DataFrame(sorted_cols_dict['Fare']).reset_index()[:10])

#print(type(np.sort(train['Fare'])), "\t\t", type(train['Fare'].sort_values(ascending=True)))
#df_train_sorted = df_train['logerror'].sort_values(ascending=True)

t = remove_outliers(train, 'Fare', 99, 2)
print(np.max(train.Fare), np.min(train.Fare))

df_coltype_counts1, df_coltype_counts2 = df_get_coltypes(train)
df_coltype_counts1[df_coltype_counts1.type == 'object']


uniquecountsdict = getuniquecounts(train, train.columns)
print(uniquecountsdict)
df_cols = train.dtypes.reset_index()
df_cols.columns = ['colname','type']
uniquecntsforobj = {k:v for k,v in uniquecountsdict.items() if k in list(df_cols.colname[df_cols.type == 'object'])}
print(uniquecntsforobj)

# Dataframe of value counts for given column
train['Embarked'].value_counts().reset_index()
# No. of unique values in given column
print("Embarked unique counts", len(train['Embarked'].value_counts()), "\t\t", len(pd.unique(train['Embarked'])))


dictcountofNaNs = getcountofNaNs(train)
dictcountofNaNs['float64']
dictcountofNaNs['object']
dictcountofNaNs

'''                     
fig = plt.figure(figsize = [12,12])
plt.bar(list(uniquecountsdict.keys()), list(uniquecountsdict.values()))
plt.xlabel('colname')
plt.ylabel('counts')
plt.xticks(rotation=90)
plt.show()
'''


train.columns

impute_attribs = ['Fare','Age'] 
attribs_toBinarize = ['Pclass', 'Sex']
attribs_toEncode = ['Embarked', 'Cabin']
dict_prop_piped = pipeliner(train, impute_attribs, attribs_toBinarize, attribs_toEncode)



jointplots(train, [['Age','Pclass']], ['hex','reg','kde','kde_special'])
jointplots(train, [['Age','Fare']], ['hex','reg','kde','kde_special'])
dualvariateplots(train, [['Age','Fare'], ['Age','Sex'], ['Pclass','Sex']], ['violinplot','swarmplot','boxplot', 'stripplot'])
univariateplots(train, ['Age', 'Fare', 'Pclass','SibSp', 'Parch'], 'violin')

dict_prop_piped
