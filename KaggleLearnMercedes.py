# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:54:35 2017

@author: echtpar
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.tree import DecisionTree

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

from sklearn.decomposition import PCA
import lightgbm as lgb


from subprocess import check_output
import datetime




#print(check_output(["ls", '../input']).decode("utf8"))

train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\test.csv")

# lets see the distribution plot


f = plt.figure(figsize = (12,6))
plt.imshow(sns.distplot(np.log1p(train['y'].values)))
plt.show()

f = plt.figure(figsize = (12,6))
plt.imshow(sns.distplot(train['y'].values))
plt.show()
        
# above plot shows the data is more nornally distributed than log of the data values

# lets see the outliers        
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train['y'].values))      
plt.xlabel('Count', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Count of y values')
plt.show()


train_cols = train.columns.values
test_cols = test.columns.values
match_cols = []
for i, col in enumerate(train_cols):
    if col in test_cols:
        match_cols.append("True" + " " + str(i) + " " + col)
    else:
        match_cols.append("Column not present in test: col no. and name:" + " " + str(i) + " " + col)

non_matching_cols = [item for item in match_cols if "Column" in item]
print(non_matching_cols)
# so we have only column 'y' which is not present in test data. All other columns match.

# check if there are any null values in the train set
df_null = train.isnull().sum().reset_index()
df_null.columns = ["Col", "Count"]
print(df_null.loc[df_null['Count']>0])

# check if any values in train set are nan = no values are nan
#print(np.any(np.isnan(train)))

# check the dtypes of columns in train set
df_dtypes = train.dtypes.reset_index()
df_dtypes.columns = ['Col', 'Dtype']
df_dtypes_grouped = df_dtypes.groupby('Dtype').aggregate('count').reset_index()
print(df_dtypes_grouped)



#print(df_all['y'][len_train-10:len_train])
#print(df_all['y'][len_train:len_train+10]) # just to check for fun that y col has NaN values for the test section

len_train = train.shape[0]
print(len_train)

y_train = train['y']
test_id = test['ID'].astype(np.int32)
print(y_train.shape)
#train.drop(['y'], axis=1, inplace=True)

df_all=None
df_all = pd.concat([train,test])

df_all.drop(["y"], axis = 1, inplace=True)

for col in df_all.columns:
    if train[col].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        df_all[col] = lbl.fit_transform(df_all[col])

        
#remove the columns with low no. of 1s to avoid overfitting        
print(df_all.shape)
df_sum=df_all.sum(axis=0).reset_index()
df_sum.columns = ['col', 'sum']
to_remove = df_sum[df_sum['sum'] < 10].index
df_all.drop(df_sum.loc[to_remove, "col"], inplace=True, axis=1)
print(df_all.shape)

print(df_all.head())
print(np.isnan(df_all).sum(axis=0).reset_index())

x_train = df_all[:len_train]
x_test = df_all[len_train:]

print(len(df_all), len(train), len(test), len(x_train), len(x_test))



#for col in train.columns:
#    if train[col].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        train[col] = lbl.fit_transform(train[col])
#
#
#for col in test.columns:
#    if test[col].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        test[col] = lbl.fit_transform(test[col])


        
        
#for col in df_all.columns:
#    if df_all[col].dtype == 'object':
#        df_dummies = pd.get_dummies(df_all[col], prefix = col)
#        df_all = pd.concat([df_all, df_dummies], axis=1)
#        df_all.drop([col], axis=1, inplace = True)




##Add decomposed components: PCA / ICA etc.
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

#n_comp = 12
n_comp = 50

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(x_train)
tsvd_results_test = tsvd.transform(x_test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(x_train)
pca2_results_test = pca.transform(x_test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(x_train)
ica2_results_test = ica.transform(x_test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(x_train)
grp_results_test = grp.transform(x_test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(x_train)
srp_results_test = srp.transform(x_test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    x_train['pca_' + str(i)] = pca2_results_train[:,i-1]
    x_test['pca_' + str(i)] = pca2_results_test[:, i-1]

    x_train['ica_' + str(i)] = ica2_results_train[:,i-1]
    x_test['ica_' + str(i)] = ica2_results_test[:, i-1]

    x_train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    x_test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    x_train['grp_' + str(i)] = grp_results_train[:,i-1]
    x_test['grp_' + str(i)] = grp_results_test[:, i-1]

    x_train['srp_' + str(i)] = srp_results_train[:,i-1]
    x_test['srp_' + str(i)] = srp_results_test[:, i-1]

## there are 8 columns that are objects. Use LabelBinarize to convert Objects into int
#for col in train.columns.values:
#    if train[col].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(train[col].values)
#        train[col] = lbl.transform(list(train[col].values))
#
#for col in test.columns.values:
#    if test[col].dtype == 'object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(test[col].values)
#        test[col] = lbl.transform(list(test[col].values))
        



# lets view a slide of the train data
temp_train = x_train.ix[:10,:10]
print(temp_train.shape)
print(temp_train)



#y should be between 85 and 130 to avoid outliers
#nonoutliers_index = x_train[(x_train['y'] <150)].index
#x_train = x_train.loc[nonoutliers_index]



print(x_train.shape)

#y_train = x_train['y']
#x_train.drop(["y"], axis = 1, inplace=True)
#x_test.drop(["y"], axis = 1, inplace=True)


# lets see the correlation map
correlation = x_train.corr()
print(correlation.shape)
cmap = sns.diverging_palette( 255 , 0 , as_cmap = True )
sns.heatmap(x_train.ix[:100,:100].astype(float).corr(),linewidths=0, square=True, cmap="viridis", xticklabels=False, yticklabels= False, annot=True)



# prepare dict of params for xgboost to run with
y_mean = np.mean(y_train)


#xgb_params = {
#    'n_trees': 800, 
#    'eta': 0.005,
#    'max_depth': 6,
#    'subsample': 0.95,
#    'objective': 'reg:linear',
##    'objective': 'multi:softprob',
##    'eval_metric': 'rmse',
#    'eval_metric': 'mae',
#    'base_score': y_mean, # base prediction = mean(target)
#    'silent': 1
#}
#
## form DMatrices for Xgboost training
#dtrain = xgb.DMatrix(x_train, y_train)
#dtest = xgb.DMatrix(x_test)
#
## xgboost, cross-validation
#cv_result = xgb.cv(xgb_params, 
#                   dtrain, 
#                   num_boost_round=700, # increase to have better results (~700)
#                   early_stopping_rounds=50,
#                   verbose_eval=50, 
#                   show_stdv=False
#                  )
#
#num_boost_rounds = len(cv_result)
#print(num_boost_rounds)
#
## train model
#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#
#predict = model.predict(dtest)
#print(type(predict))
#result = pd.concat([test["ID"], pd.DataFrame(predict)], axis=1)
#result.columns = ['ID', 'y']
#print(result[:10])
#result.to_csv("2017-07-05-Merc-SubmitV2.csv")



######################


# prepare dict of params for xgboost to run with
xgb_params = {
#    'n_trees': 520,
    'n_trees': 800,
    'eta': 0.0045,
    'max_depth': 6,
    'subsample': 0.98,
    'objective': 'reg:linear',
#    'eval_metric': 'rmse',
    'eval_metric': 'mae',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}




# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1350, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)



#num_boost_rounds = 1350
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# check f2-score (to get higher score - increase num_boost_round in previous cell)
#from sklearn.metrics import r2_score
print(r2_score(dtrain.get_label(),model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test_id, 'y': y_pred})
output.to_csv('2017-07-07-Merc-SubmitV7.csv', index=False)    




######################






from tpot import TPOTRegressor
auto_classifier = TPOTRegressor(generations=2, population_size=8, verbosity=2)
from sklearn.model_selection import train_test_split

# Split training data to train and validate
X_train, X_valid, y_train, y_valid = train_test_split(finaltrainset, y_train,
                                                    train_size=0.75, test_size=0.25)

auto_classifier.fit(X_train, y_train)

print("The cross-validation accuracy")
print(auto_classifier.score(X_valid, y_valid))

test_result = auto_classifier.predict(finaltestset)
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = test_result

sub.to_csv('MB_TpotModels.csv', index=False)


sub.head()















xgb_params = {
              'eta': 0.05,
              'max_depth': 6,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': 1
              }
              
              
dtrain = xgb.DMatrix(x_train, y_train, feature_names = x_train.columns.values)
dtest = xgb.DMatrix(x_test)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=700)

print(len(set(train["ID"].values)))
print(len(set(train["X0"].values)))

predict = model.predict(dtest)
print(type(predict))
result = pd.concat([test["ID"], pd.DataFrame(predict)], axis=1)
result.columns = ['ID', 'y']
print(result[:10])
result.to_csv("2017-07-02-Merc-SubmitV4.csv")




RS=1
np.random.seed(RS)
ROUNDS = 1500 # 1300,1400 all works fine
params = {
    'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.01 , #small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 2 ** 5,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': RS,
        'feature_fraction': 0.7,
        'feature_fraction_seed': RS,
        'max_bin': 100,
        'max_depth': 7,
        'num_rounds': ROUNDS,
    }


train_lgb=lgb.Dataset(x_train,y_train)
model=lgb.train(params,train_lgb,num_boost_round=ROUNDS)
predict=model.predict(x_test)
    
result = pd.concat([test["ID"], pd.DataFrame(predict)], axis=1)
result.columns = ['ID', 'y']
print(result[:10])
result.to_csv("2017-07-01-Merc-SubmitV2.csv")




# another model by Fred Navruzov Baselines-To-Start-With(LB=0.56+)
y_mean = np.mean(y_train)
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

predict = model.predict(dtest)
result = pd.concat([test["ID"], pd.DataFrame(predict)], axis=1)
result.columns = ['ID', 'y']
print(result[:10])
result.to_csv("2017-07-01-Merc-SubmitV3.csv")








# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

train.ix[:10,:10].columns
train.ix[:, 10:].columns
train['X10'].unique()
print(np.sort(train['X10'].unique()).tolist())

for col in train.ix[:, 10:].columns:
    print(col, np.sort(train[col].unique()).tolist())

plt.figure(figsize=(30,6))

ax = plt.subplot(1,3,1)
var_name = "X0"
col_order = np.sort(train[var_name].unique()).tolist()
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
    

from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(x_train, y_train)
feat_names = train.columns.values

## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


pca = PCA(300)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

print(x_train_pca.shape)        
print(x_test_pca.shape)        

print(type(x_train_pca))        
print(type(x_test_pca))        


xgb_params = {
              'eta': 0.05,
              'max_depth': 6,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': 1
              }
              
              
dtrain_pca = xgb.DMatrix(x_train_pca, y_train)
dtest = xgb.DMatrix(x_test_pca)
model = xgb.train(dict(xgb_params, silent=1), dtrain_pca, num_boost_round=100)

print(len(set(train["ID"].values)))
print(len(set(train["X0"].values)))

predict = model.predict(dtest)
print(type(predict))
result = pd.concat([test["ID"], pd.DataFrame(predict)], axis=1)
result.columns = ['ID', 'y']
print(result[:10])
result.to_csv("2017-07-02-Merc-Submit.csv")
