# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:40:06 2017

@author: echtpar
"""

"""

Chandrakant Pattekar
Instacart - LB 0.38 running on Kaggle - F1 vs Size
L
forked from Instacart - LB 0.38 running on Kaggle - F1 vs Size by Rudolph (+0/–0)
Chandrakant Pattekar
Instacart Market Basket Analysis
voters
last run 2 hours ago · Python notebook
using data from Instacart Market Basket Analysis ·
PrivateMake Public


"""


import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#myfolder = '../input/'
print('loading files ...')

path_orders = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\orders.csv"
path_products = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\products.csv"
path_order_products__prior = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__prior.csv"
path_departments = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\departments.csv"
path_aisles = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\aisles.csv"
path_order_products__train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__train.csv"



#prior = pd.read_csv(myfolder + 'order_products__prior.csv', dtype={'order_id': np.uint32,
#           'product_id': np.uint16, 'reordered': np.uint8, 'add_to_cart_order': np.uint8})
#
#train_orders = pd.read_csv(myfolder + 'order_products__train.csv', dtype={'order_id': np.uint32,
#           'product_id': np.uint16, 'reordered': np.int8, 'add_to_cart_order': np.uint8 })
#
#orders = pd.read_csv(myfolder + 'orders.csv', dtype={'order_hour_of_day': np.uint8,
#           'order_number': np.uint8, 'order_id': np.uint32, 'user_id': np.uint32,
#           'order_dow': np.uint8, 'days_since_prior_order': np.float16})
#
#orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.uint8)
#orders.days_since_prior_order = orders.days_since_prior_order.fillna(30).astype(np.uint8)
#
#products = pd.read_csv(myfolder + 'products.csv', dtype={'product_id': np.uint16,
#            'aisle_id': np.uint8, 'department_id': np.uint8},
#             usecols=['product_id', 'aisle_id', 'department_id'])


prior = pd.read_csv(path_order_products__prior, dtype={'order_id': np.uint32,
           'product_id': np.uint16, 'reordered': np.uint8, 'add_to_cart_order': np.uint8})

train_orders = pd.read_csv(path_order_products__train, dtype={'order_id': np.uint32,
           'product_id': np.uint16, 'reordered': np.int8, 'add_to_cart_order': np.uint8 })

orders = pd.read_csv(path_orders, dtype={'order_hour_of_day': np.uint8,
           'order_number': np.uint8, 'order_id': np.uint32, 'user_id': np.uint32,
           'order_dow': np.uint8, 'days_since_prior_order': np.float16})

print(orders.columns)

orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.uint8)

print(set(orders.eval_set))

orders.days_since_prior_order = orders.days_since_prior_order.fillna(30).astype(np.uint8)

#print(orders.days_since_prior_order.value_counts())

products = pd.read_csv(path_products, dtype={'product_id': np.uint16,
            'aisle_id': np.uint8, 'department_id': np.uint8},
             usecols=['product_id', 'aisle_id', 'department_id'])



print(products.columns)


print('done loading')





print('merge prior and orders and keep train separate ...')

orders_products = orders.merge(prior, how = 'inner', on = 'order_id')

orders_products.columns.values
orders.columns.values
prior.columns.values

print(prior.reordered.value_counts())

orders_products.shape
orders.shape
prior.shape

print(train_orders[:10])

train_orders = train_orders.merge(orders[['user_id','order_id']], left_on = 'order_id', right_on = 'order_id', how = 'inner')

print(train_orders[:10])
print(orders_products[:5])

print(train_orders.columns)
print(orders_products.columns)

del prior
gc.collect()



print('Creating features I ...')

# sort orders and products to get the rank or the reorder frequency
prdss = orders_products.sort_values(['user_id', 'order_number', 'product_id'], ascending=True)
print(prdss.columns)
prdss['product_time'] = prdss.groupby(['user_id', 'product_id']).cumcount()+1

print(prdss.loc[:10,['order_id', 'user_id', 'product_id', 'product_time']])

print(prdss.columns)
print(orders_products.columns)
print(prdss.product_time.unique()[:10])
print(prdss.product_time.value_counts()[:10])

# getting products ordered first and second times to calculate probability later
sub1 = prdss[prdss['product_time'] == 1].groupby('product_id').size().to_frame('prod_first_orders')
sub2 = prdss[prdss['product_time'] == 2].groupby('product_id').size().to_frame('prod_second_orders')

print(sub1[:10])
print(sub2[:10])


sub1['prod_orders'] = prdss.groupby('product_id')['product_id'].size()
sub1['prod_reorders'] = prdss.groupby('product_id')['reordered'].sum()
sub2 = sub2.reset_index().merge(sub1.reset_index())
sub2['prod_reorder_probability'] = sub2['prod_second_orders']/sub2['prod_first_orders']
sub2['prod_reorder_ratio'] = sub2['prod_reorders']/sub2['prod_orders']
prd = sub2[['product_id', 'prod_orders','prod_reorder_probability', 'prod_reorder_ratio']]

del sub1, sub2, prdss
gc.collect()


print(prd.columns)

print('Creating features II ...')

# extracting prior information (features) by user
users = orders[orders['eval_set'] == 0].groupby(['user_id'])['order_number'].max().to_frame('user_orders')
users['user_period'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].sum()
users['user_mean_days_since_prior'] = orders[orders['eval_set'] == 0].groupby(['user_id'])['days_since_prior_order'].mean()

print(users.user_mean_days_since_prior.value_counts())
users.drop(['user_mean_days_since_prior'], axis = 1)

print(users.columns)


# merging features about users and orders into one dataset
us = orders_products.groupby('user_id').size().to_frame('user_total_products')
us['eq_1'] = orders_products[orders_products['reordered'] == 1].groupby('user_id')['product_id'].size()
us['gt_1'] = orders_products[orders_products['order_number'] > 1].groupby('user_id')['product_id'].size()
us['user_reorder_ratio'] = us['eq_1'] / us['gt_1']
us.drop(['eq_1', 'gt_1'], axis = 1, inplace = True)
us['user_distinct_products'] = orders_products.groupby(['user_id'])['product_id'].nunique()

print(us.columns)

# the average basket size of the user
users = users.reset_index().merge(us.reset_index())
users['user_average_basket'] = users['user_total_products'] / users['user_orders']

us = orders[orders['eval_set'] != 0]
us = us[['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
users = users.merge(us)

print(users.columns)

del us
gc.collect()





print('Finalizing features and the main data file  ...')
# merging orders and products and grouping by user and product and calculating features for the user/product combination
data = orders_products.groupby(['user_id', 'product_id']).size().to_frame('up_orders')
data['up_first_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].min()
data['up_last_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].max()
data['up_average_cart_position'] = orders_products.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
data = data.reset_index()

#merging previous data with users
data = data.merge(prd, on = 'product_id')
data = data.merge(users, on = 'user_id')

#user/product combination features about the particular order
data['up_order_rate'] = data['up_orders'] / data['user_orders']
data['up_orders_since_last_order'] = data['user_orders'] - data['up_last_order']
data = data.merge(train_orders[['user_id', 'product_id', 'reordered']], 
                  how = 'left', on = ['user_id', 'product_id'])
data = data.merge(products, on = 'product_id')

print(data.columns)

del orders_products     #, orders, train_orders
gc.collect()





print(' Training data for later use in F1 vs cart size only  ...')

#save the actual reordered products of the train set in a list format and then delete the original frames
train_orders = train_orders[train_orders['reordered']==1].drop('reordered',axis=1)
orders.set_index('order_id', drop=False, inplace=True)
train1=orders[['order_id','user_id']].loc[orders['eval_set']==1]
train1['actual'] = train_orders.groupby('order_id').aggregate({'product_id':lambda x: list(x)})
train1['actual']=train1['actual'].fillna('')
n_actual = train1['actual'].apply(lambda x: len(x)).mean()   # this is the average cart size

del orders, train_orders
gc.collect()






print('setting dtypes for data ...')

#reduce the size by setting data types
data = data.astype(dtype= {'user_id' : np.uint32, 'product_id'  : np.uint16,
            'up_orders'  : np.uint8, 'up_first_order' : np.uint8, 'up_last_order' : np.uint8,
            'up_average_cart_position' : np.uint8, 'prod_orders' : np.uint16, 
            'prod_reorder_probability' : np.float16,   
            'prod_reorder_ratio' : np.float16, 'user_orders' : np.uint8,
            'user_period' : np.uint8, 'user_mean_days_since_prior' : np.uint8,
            'user_total_products' : np.uint8, 'user_reorder_ratio' : np.float16, 
            'user_distinct_products' : np.uint8, 'user_average_basket' : np.uint8,
            'order_id'  : np.uint32, 'eval_set' : np.uint8, 
            'days_since_prior_order' : np.uint8, 'up_order_rate' : np.float16, 
            'up_orders_since_last_order':np.uint8,
            'aisle_id': np.uint8, 'department_id': np.uint8})

data['reordered'].fillna(0, inplace=True)  # replace NaN with zeros (not reordered) 
data['reordered']=data['reordered'].astype(np.uint8)

gc.collect()


print(data.columns)


print('Preparing Train and Test sets ...')

# filter by eval_set (train=1, test=2) and dropp the id's columns (not part of training features) 
# but keep prod_id and user_id in test

train = data[data['eval_set'] == 1].drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis = 1)
test =  data[data['eval_set'] == 2].drop(['eval_set', 'user_id', 'reordered'], axis = 1)

#del data
gc.collect()





print('preparing X,y for LightGBM ...')

# for preliminary runs sample a fraction of the data by (un)commenting the next two lines
#print('sampling train data ...')
#train = train.sample(frac=0.25)

# Splitting the training set to train and validation set
X_train, X_eval, y_train, y_eval = train_test_split(
    train[train.columns.difference(['reordered'])], train['reordered'], test_size=0.33)

del train
gc.collect()



print(X_train.columns)
print(y_train[:10])


print('formatting and training LightGBM ...')

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference = lgb_train)

# there is some room to change the parameters and improve - I have not done it systematically
# for example change num_boost_round to 200

params = {'task': 'train', 'boosting_type': 'gbdt',   'objective': 'binary', 'metric': {'binary_logloss', 'auc'},
    'num_iterations' : 1000, 'max_bin' : 100, 'num_leaves': 512, 'feature_fraction': 0.6, 'learning_rate' : 0.05}

lgb_model = lgb.train(params, lgb_train, num_boost_round = 75, valid_sets = lgb_eval, early_stopping_rounds=10)

del lgb_train, X_train, y_train
gc.collect()

print('applying model to test data ...')
test['reordered'] = lgb_model.predict(test[test.columns.difference(
    ['order_id', 'product_id'])], num_iteration = lgb_model.best_iteration)

gc.collect()





print('formatting and writing to submission file ...')

#set the threshold z (should be optimized for F1) 
z=0.22
prd_bag = dict()
for row in test.itertuples():
    if row.reordered > z:   
        try:
            prd_bag[row.order_id] += ' ' + str(row.product_id)
        except:
            prd_bag[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in prd_bag:
        prd_bag[order] = 'None'

submit = pd.DataFrame.from_dict(prd_bag, orient='index')
submit.reset_index(inplace=True)
submit.columns = ['order_id', 'products']
submit.to_csv('LightGBM_submit22.csv', index=False)
print(submit['products'].apply(lambda x: len(x.split())).mean())




# check feature importance
lgb.plot_importance(lgb_model, figsize=(7,9))
plt.show()




print(' F1 vs cart size analysis ...')

check =  data[data['eval_set'] == 1].drop(['eval_set', 'user_id', 'reordered'], axis = 1)

check['reordered'] = lgb_model.predict(check[check.columns.difference(
    ['order_id', 'product_id'])], num_iteration = lgb_model.best_iteration)

print(' running ...')

def f1_score_single(x):                 #from LiLi
    y_true = set(x.actual)
    y_pred = set(x.list_prod)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)

F1array=np.array([])
narray=np.array([])
zarray=np.array([])
actualarray=np.array([])

# change the range to 0.12 - 0.30 in order to produce data for the chart in the introduction
# otherwise there may not be enough time for it to run on Kaggle
#for z in np.arange(0.12,0.31,0.01):
for z in np.arange(0.22,0.23,0.01):    

    prd_bag = dict()
    for row in check.itertuples():
        if row.reordered > z:   
            try:
                prd_bag[row.order_id] += ' ' + str(row.product_id)
            except:
                prd_bag[row.order_id] = str(row.product_id)

    for order in check.order_id:
        if order not in prd_bag:
            prd_bag[order] = ' '

    submit2 = pd.DataFrame.from_dict(prd_bag, orient='index')
    submit2.reset_index(inplace=True)
    submit2.columns = ['order_id', 'products']
    submit2['list_prod']=submit2['products'].apply(lambda x: list(map(int, x.split())))
    n = submit2['products'].apply(lambda x: len(x.split())).mean()
        
    predact=pd.merge(train1,submit2,on='order_id',how='inner')
    predact['f1']=predact.apply(f1_score_single,axis=1)
    F1 = predact['f1'].mean()
    
    F1array=np.append(F1array,F1)
    narray=np.append(narray,n)
    zarray=np.append(zarray,z)
    actualarray=np.append(actualarray,n_actual)
    
    print(' F1, n, z, n_actual :  ', F1,n,z,n_actual)
    
print(' done ')





#I saved one of the previous runs so that it is not timed out on Kaggle
Y1 =[0.368,0.373,0.377,0.3801,0.382,0.3828,0.3832,0.3825,0.3815,0.3796,
         0.3771,0.3744,0.3713,0.3677,0.3636,0.3591,0.3542,0.349,0.3438]
X=np.arange(0.12,0.31,0.01)
Y2 = np.empty(19)
Y2.fill(6.31)
Y3=[15.5,14.3,13.3,12.3,11.5,10.8,10.1,9.53,8.97,8.46,7.99,7.55,7.15,6.77,6.41,6.08,5.78,5.4,5.2]
#replace X,Y1,Y2,Y3 with zarray,F1array,actualarray,narray to update

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(X, Y2, '-', label = 'Actual')
lns2 = ax.plot(X, Y3, '-', label = 'Predicted')
ax2 = ax.twinx()
lns3 = ax2.plot(X, Y1, '-r', label = 'F1')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mean Cart Size')
ax2.set_ylabel('F1')
plt.suptitle('F1 vs Mean Cart Size', size=12)
plt.savefig('F1_vs_mean_cart_size.jpg')
plt.show()