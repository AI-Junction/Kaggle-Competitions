# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:04:08 2017

@author: echtpar
"""



####################################


#Adarsh Chavakula
#How to cross validate properly
#Adarsh Chavakula
#Mercedes-Benz Greener Manufacturing
#voters
#last run 3 days ago · Python script · 467 views
#using data from Mercedes-Benz Greener Manufacturing ·
#Public

####################################

'''
This script is to illustrate a solid cross validation process for this competition.
We use 10 fold out-of-bag overall cross validation instead of averaging over folds. 
The entire process is repeated 5 times and then averaged.

You would notice that the CV value obtained by this method would be lower than the
usual procedure of averaging over folds. It also tends to have very low deviation.

Any scikit learn model can be validated using this. Models like XGBoost and 
Keras Neural Networks can also be validated using their respective scikit learn APIs.
XGBoost is illustrated here along with Ridge regression.
'''


import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import xgboost as xgb

def R2(ypred, ytrue):
    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg)**2)
    SS_res = np.sum((ytrue - ypred)**2)
    r2 = 1 - (SS_res/SS_tot)
    return r2

def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    ypred = np.zeros((len(y),repeats))
    score = np.zeros(repeats)
    x = np.array(x)
    for r in range(repeats):
        i=0
        print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
        x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
        kf = KFold(n_splits=folds,random_state=i+1000) #random split, different each time
        for train_ind,test_ind in kf.split(x):
            print('Fold', i+1, 'out of',folds)
            xtrain,ytrain = x[train_ind,:],y[train_ind]
            xtest,ytest = x[test_ind,:],y[test_ind]
            model.fit(xtrain, ytrain)
            ypred[test_ind,r]=model.predict(xtest)
            i+=1
        score[r] = R2(ypred[:,r],y)
    print('\nOverall R2:',str(score))
    print('Mean:',str(np.mean(score)))
    print('Deviation:',str(np.std(score)))
    pass

#def main():

train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")
#test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\test.csv")
#train = pd.read_csv('../input/train.csv')
y = np.array(train['y'])
train = train.drop(['ID','y','X0','X1','X2','X3','X4','X5','X6','X8'], axis=1)
ridge_model = Ridge(alpha=1)
xgb_model = xgb.XGBRegressor(max_depth=2, learning_rate=0.01, n_estimators=10, silent=True,
                            objective='reg:linear', nthread=-1, base_score=100, seed=4635,
                            missing=None)
cross_validate(ridge_model, np.array(train), y, folds=10, repeats=5) #validate ridge regression
cross_validate(xgb_model, np.array(train), y, folds=10, repeats=5) #validate xgboost

#pass

#if __name__ == '__main__':
#	main()




################################

#LinuX18 - kernel 0.5686？？？
#LinuX18
#Mercedes-Benz Greener Manufacturing
#last run 23 days ago · Python script · 1542 views
#using data from Mercedes-Benz Greener Manufacturing ·
#Public
################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

# read datasets

train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\test.csv")


#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')


# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
#n_comp = 12
#n_comp = 50
n_comp = 10

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

train_pca = pd.DataFrame()
test_pca =  pd.DataFrame()

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    train['grp_' + str(i)] = grp_results_train[:,i-1]
    test['grp_' + str(i)] = grp_results_test[:, i-1]

    train['srp_' + str(i)] = srp_results_train[:,i-1]
    test['srp_' + str(i)] = srp_results_test[:, i-1]

          
#for i in range(1, n_comp+1):
#    train_pca['pca_' + str(i)] = pca2_results_train[:,i-1]
#    test_pca['pca_' + str(i)] = pca2_results_test[:, i-1]
#
#    train_pca['ica_' + str(i)] = ica2_results_train[:,i-1]
#    test_pca['ica_' + str(i)] = ica2_results_test[:, i-1]
#
#    train_pca['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
#    test_pca['tsvd_' + str(i)] = tsvd_results_test[:, i-1]
#
#    train_pca['grp_' + str(i)] = grp_results_train[:,i-1]
#    test_pca['grp_' + str(i)] = grp_results_test[:, i-1]
#
#    train_pca['srp_' + str(i)] = srp_results_train[:,i-1]
#    test_pca['srp_' + str(i)] = srp_results_test[:, i-1]

          
y_train = train["y"]
y_mean = np.mean(y_train)



### Regressor
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

#dtrain = xgb.DMatrix(train_pca, y_train)
#dtest = xgb.DMatrix(test_pca)


num_boost_rounds = 1350
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score
print(r2_score(dtrain.get_label(),model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('2017-07-08-LinuXV4.csv', index=False)    












###############################
#Utkarsh
#A Strategy for Feature Engine
#Utkarsh
#Mercedes-Benz Greener Manufacturing
#voters
#last run a month ago · Python notebook · 1657 views
#using data from Mercedes-Benz Greener Manufacturing ·
#Public
###############################




#Feature engineering in this competition is quite challenging as the column names have been masked. 
#I have tried many approaches of adding, multiplying etc. which all have failed. 
#The number of combinations that we can make using the given columns is too huge. To tackle this and to reduce the number of combinations I have created the following strategy for feature engineering. I hope it helps.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

#sample_submission.csv
#test.csv
#train.csv

#/opt/conda/lib/python3.6/site-packages/sklearn/cross_validation.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
#  "This module will be removed in 0.20.", DeprecationWarning)

    
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\test.csv")
    
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

# Remove the outlier
train=train[train.y<250]

#The strategy First we split the data into two parts. Using the cross validated results try finding out which range of values have max error. In this example I take 100.

# Check no. of rows greater than equal to 100
len(train['y'][(train.y>=100)])

#2005

# Check no. of rows less than 100
len(train['y'][(train.y<100)])

#2203

#Now we convert the training set into a classification problem. Create a new field for class.

train['y_class'] = train.y.apply(lambda x: 0 if x<100  else 1 )

# Concat the datasets
data = pd.concat([train,test])

# Removing object type vars as I am more interested in binary ones
data = data.drop(data.select_dtypes(include = ['object']).columns,axis=1)

feat = list(data.drop(['y','y_class'],axis=1).columns.values)

train_df = (data[:train.shape[0]])
test_df = (data[train.shape[0]:])

train_df.drop(['ID'], axis=1, inplace=True)
test_df.drop(['ID'], axis=1, inplace=True)

# I have not removed zero valued columns for now
len(feat)

#369

# Remove ID as we want some honest features :)
feat.remove('ID')

from sklearn.metrics import f1_score as f1

# Calculating CV score
def cv_score(model):
    return cross_val_score(model,train_df[feat],train_df['y_class'],cv=10,scoring = 'f1').mean()

#Now, the interesting part.Decision trees are the basic entities that make up complex algos like XGB. But to understand the rules on the which splitting happens is not possible in XGB. Here, we build decision trees to understand the rules and build features for the same. Lets go!

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

model = DTC(max_depth = 5,min_samples_split=200) # We don't want to overfit

# Its important to notice that there is no sense in keeping high depth values. Since we need strong features, the rules should have considerably large sample size in the leaves. For small sample size values, the feature may not be that strong.

cv_score(model) 

#0.86590019119911388

# F1 looks good! But sometimes it may not. Doesn't matter, as we want the branches with good gini scores and sample size

model.fit(train_df[feat],train_df.y_class)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            #min_impurity_decrease=0.0, #min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=200,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

#> To visualize the tree we use graphviz

# Graphviz is used to build decision trees
from sklearn.tree import export_graphviz
from sklearn import tree

# This statement builds a dot file.
tree.export_graphviz(model, out_file='tree.dot',feature_names  = feat)  

#After building the tree dot file you can convert it into a png using :
#    dot -Tpng tree.dot -o tree.png
#in command line (first cd to dot directory)

# This will bring the image to the notebook (or you can view it locally)
from IPython.display import Image
#Image("tree.png") # Uncomment if you are trying this on local
# Can't read the image in kernal. Anyone know how? Will try and add image in comments
#EDIT : Here is the link to the image - Tree
#    Now, if you start traversing the tree. We see that when, X314 = 0 and X315 = 0 and X47=0 then the gini score is .2183 with good sample size!! This can be a good feature as it gives us a good separation between the classes.
#Experiment more with this notebook and if you are generous enough, keep adding good features in comments below ;D
#Thanks! :)

print(test.columns)    
print(test['ID'][:10])    
sub = pd.DataFrame()
sub['ID'] = test['ID']    
#test.drop(['ID'], axis=1, inplace=True)    
print(np.any(np.isnan(test_df)))
z=test_df.dtypes
z.columns = ['col', 'type']
z.columns
print(z[:10])
print(type(z))
z=pd.DataFrame(z, index=False)
k = z.groupby('type').aggregate('count').reset_index()

y_pred = model.predict(test_df)    
sub['y'] = y_pred
sub.to_csv('2017-07-06-stacked-modelsV1.csv', index=False)



    
################################

#bytestorm
#stacked then averaged models [~ 0.5697]
#L
#forked from stacked then averaged models [~ 0.5697] by Hakeem (+0/-0/~0)
#bytestorm
#Mercedes-Benz Greener Manufacturing
#voters
#last run 11 days ago · Python script · 325 views
#using data from Mercedes-Benz Greener Manufacturing ·
#Public



################################




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score



class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\test.csv")
        
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('2017-07-06-stacked-modelsV1.csv', index=False)


# Any results you write to the current directory are saved as output.    



