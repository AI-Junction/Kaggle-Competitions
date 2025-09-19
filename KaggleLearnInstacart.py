# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:37:03 2017

@author: echtpar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from subprocess import check_output
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as gbm
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer

#print(check_output("ls", "../input"))


orders = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\orders.csv")
products = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\products.csv")
order_products__prior = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__prior.csv")
departments = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\departments.csv")
aisles = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\aisles.csv")
order_products__train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__train.csv")

orders_columns_dtype = orders.dtypes.reset_index()
orders_columns_dtype.columns = ['col', 'type']
orders_columns_dtype_grouped = orders_columns_dtype.groupby('type').aggregate('count').reset_index()
print(orders_columns_dtype_grouped)

df_cat=None
print(orders.eval_set.unique())
for col in orders.columns:
    if orders[col].dtype == 'object':
        df_cat = pd.get_dummies(orders[col].values, prefix = col)
        orders.drop([col], inplace=True, axis=1)
pd.concat([orders, df_cat], axis = 1)


products_columns_dtype = products.dtypes.reset_index()
products_columns_dtype.columns = ['col', 'type']
products_columns_dtype_grouped = products_columns_dtype.groupby('type').aggregate('count').reset_index()
print(products_columns_dtype_grouped)


print(products.columns)
df_dummies = None
print(products.product_name.head())
#products.map(products.apply(lambda x: pd.get_dummies(x, prefix='col')))

df_split_products = pd.DataFrame()
df_split_products['split_tags'] = products['product_name'].map(lambda row: row.split(" "))
#y = df_split_products['split_tags'].str.get_dummies(sep=' ')

print(df_split_products.head())
print(products.product_name.head())
print(type(df_split_products.split_tags[0]))

labels = np.array(products['product_name'])
print(labels[-10:])
vect = CountVectorizer()
vect.fit(labels)
z = vect.get_feature_names()
print(z[:1000])
labels_dtm = vect.transform(labels)
print(labels_dtm[:10])

df_labels = pd.DataFrame(labels_dtm.toarray(), columns = vect.get_feature_names())

print(df_labels.shape)
print(df_labels.head())


#pd.get_dummies(df_split_products['split_tags']).astype(int)

print(df_split_products.head())

from sklearn.preprocessing import LabelBinarizer
labelbin = LabelBinarizer(sparse_output=True)
print(labelbin)
labels_dtm = labelbin.fit_transform(labels)
labelbin.classes_
df_labels2 = pd.DataFrame(labels_dtm.toarray(), columns = labelbin.classes_)




####### checking some concepts
label1 = ["my first_name is Chandrakant and my last_name is Pattekar"]
label2 = ["this is my address"]
dfnew = pd.DataFrame()
label1s = label1.split(' ')
label2s = label2.split(' ')
print(type(label1))

df_new = pd.DataFrame()
label = label1+label2
print(label)
dict1 = {}
dict1['label'] = label
print(dict1)
df_new = pd.DataFrame(dict1)
print(df_new)

pd.concat([dfnew['label'],label2], axis = 0)



print(dfnew)
print(label1s)
print('is' in label1s)


q = label1.get_dummies(sep=' ')

y = dfnew['label1'].str.get_dummies(sep=' ')
y = pd.get_dummies(dfnew['label1']).astype(int)
y

dfnew2 = pd.DataFrame({'tag1': label1s})
dfnew2

listalllabels = []

####### end concepts check




for item in products.product_name.values:
    dummies = item.apply(lambda x: pd.get_dummies(x, prefix='col'))
    df_dummies = df_dummies.append(dummies)
    

df_cat=None
for col in products.columns:
    if products[col].dtype == 'object':
        print(products[col].unique())
        df_cat = pd.get_dummies(products[col].values, prefix = col)
        products.drop([col], inplace=True, axis=1)
pd.concat([products, df_cat], axis = 1)



#####################################

"""
Instacart - ACME Notebook
L
forked from ACME Notebook by the1owl (+0/–0)
Chandrakant Pattekar
Instacart Market Basket Analysis
voters
last run 20 minutes ago · Python notebook
using data from Instacart Market Basket Analysis ·
PrivateMake Public
"""

#####################################

from IPython.display import display
import pandas as pd
import numpy as np
from sklearn import * 
import sklearn
#import xgboost as xgb
#import lightgbm as lgb
import gc
import random

pd.options.mode.chained_assignment = None
random.seed(0)
np.random.seed(0)


path_orders = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\orders.csv"
path_products = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\products.csv"
path_order_products__prior = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__prior.csv"
path_departments = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\departments.csv"
path_aisles = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\aisles.csv"
path_order_products__train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__train.csv"



#add your features here, I borrowed many from other Kernels
prior = pd.read_csv(path_order_products__prior).fillna(0)
orders = pd.read_csv(path_orders).fillna(0)
orders.order_dow = orders.order_dow.astype(np.int8)
orders.order_hour_of_day = orders.order_hour_of_day.astype(np.int8)
orders.order_number = orders.order_number.astype(np.int16)
orders.order_id = orders.order_id.astype(np.int32)
orders.user_id = orders.user_id.astype(np.int32)
orders.days_since_prior_order = orders.days_since_prior_order.astype(np.float32)
print(orders.values.shape)
display(orders.head())
gc.collect()

products = pd.read_csv(path_products).fillna(0)
products.drop(['product_name'], axis=1, inplace=True)
products.aisle_id = products.aisle_id.astype(np.int8)
products.department_id = products.department_id.astype(np.int8)
products.product_id = products.product_id.astype(np.int32)
prods = pd.DataFrame()
prods['orders'] = prior.groupby(prior.product_id).size().astype(np.float32)
prods['reorders'] = prior['reordered'].groupby(prior.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)


print(prods['reorders'].unique())
print(prods.reorders[:10])
print(prior.groupby([prior.product_id, prior.reordered])['reordered'].count().astype(np.float32))



products = products.join(prods, on='product_id').fillna(0)
print(products.values.shape)
display(products.head())
gc.collect()

#partial prior
prior = orders[orders['eval_set']=='prior']
prior = pd.merge(orders, pd.read_csv(path_order_products__prior).fillna(0), how='inner', on='order_id')
prior = pd.merge(prior, pd.read_csv(path_products).fillna(0), how='inner', on='product_id')
prior.drop(['product_name'], axis=1, inplace=True)
prior.reordered = prior.reordered.astype(np.int8)
prior.add_to_cart_order = prior.add_to_cart_order.astype(np.int16)

#print(prior.add_to_cart_order[-5:])

usr = orders.groupby('user_id', as_index=False)['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
users = pd.DataFrame()
users['total_items'] = prior.groupby('user_id').size().astype(np.int16)
users['total_distinct_items'] = prior.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.total_distinct_items.map(len)).astype(np.int16)
users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print(users.values.shape)
display(users.head())
gc.collect()

test = orders[orders['eval_set']=='test'].reset_index(drop=True)
test['add_to_cart_order'] = 1
test.add_to_cart_order = test.add_to_cart_order.astype(np.int16)
test = pd.merge(test, users, how='left', on='user_id').fillna(0)
print(test.values.shape)
display(test.head())
gc.collect()

train = orders[orders['eval_set']=='train']
train = pd.merge(train, pd.read_csv(path_order_products__train).fillna(0), how='inner', on='order_id')
train = pd.merge(train, pd.read_csv(path_products).fillna(0), how='inner', on='product_id')
train.drop(['product_name'], axis=1, inplace=True)
train.reordered = train.reordered.astype(np.int8)
train.add_to_cart_order = train.add_to_cart_order.astype(np.int16)
train = pd.merge(train, users, how='left', on='user_id').fillna(0)
print(train.values.shape)
display(train.head())
gc.collect()

prior = prior[prior['user_id'].isin(test.user_id.unique())]
prior = pd.merge(prior, users, how='left', on='user_id').fillna(0)
print(prior.values.shape)
display(prior.head())
gc.collect()

col = [c for c in prior.columns if c not in ['order_id', 'eval_set','reordered']] #'user_id', 
y = prior.reordered.values
test = test.reset_index(drop=True)
print(test.shape)
#test = test[:1000] #limiting to 1000 for Kaggle Kernel - Comment this line out

etc = ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=4, n_jobs=-1, random_state=1) #up the estimators or max_depth here
etc.fit(prior[col], y)

print(prior[col][:10])

#prior[prior['user_id'].isin([test.user_id[u]])].product_id.unique()

""" start trial"""
print(prior[prior['user_id'] == [test.user_id[1]]].product_id)
print(test.user_id[:2])
products[products['product_id'].isin(pprod)]
print([test.columns.values])
""" End trial"""

out = []
for u in range(len(test)):
    pred = 'None'
    pprod = prior[prior['user_id'].isin([test.user_id[u]])].product_id.unique()
    possible = products[products['product_id'].isin(pprod)]
    if len(possible)>1:
        for c in test.columns:
            possible[c] = test[c][u]
        pred = etc.predict_proba(possible[col])[:,1]
        pred = ' '.join([str(j[1]) for j in sorted([[pred[i],pprod[i]] for i in range(len(pprod)) if pred[i]>.21], reverse=True)][:20])
    if len(pred) == 0:
        pred = 'None'
    if u % 1000 == 0:
        print(u)
    out.append(pred)
    gc.collect()

test['products'] = out
test[['order_id','products']].to_csv('submission.csv', index=False)




#########################################

"""
Instacart Market Basket Analysis
"""

#########################################



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

path_orders = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\orders.csv"
path_products = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\products.csv"
path_order_products__prior = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__prior.csv"
path_departments = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\departments.csv"
path_aisles = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\aisles.csv"
path_order_products__train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInstaCartMarketData\\order_products__train.csv"



#order_products_train_df = pd.read_csv("../input/order_products__train.csv")
#order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
#orders_df = pd.read_csv("../input/orders.csv")
#products_df = pd.read_csv("../input/products.csv")
#aisles_df = pd.read_csv("../input/aisles.csv")
#departments_df = pd.read_csv("../input/departments.csv")


order_products_train_df = pd.read_csv(path_order_products__train)
order_products_prior_df = pd.read_csv(path_order_products__prior)
orders_df = pd.read_csv(path_orders)
products_df = pd.read_csv(path_products)
aisles_df = pd.read_csv(path_aisles)
departments_df = pd.read_csv(path_departments)




orders_df.head()

order_products_prior_df.head()

order_products_train_df.head()

cnt_srs = orders_df.eval_set.value_counts()

print(set(order_products_train_df.reordered))

print(cnt_srs.index.values)
print(cnt_srs)

print(set(orders_df.eval_set))

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
print(cnt_srs)



cnt_srs = orders_df.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_srs = cnt_srs.order_number.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


plt.figure(figsize=(12,8))
sns.countplot(x="order_dow", data=orders_df, color=color[0])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by week day", fontsize=15)
plt.show()



plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders_df, color=color[1])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()


grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

print(grouped_df.head())

plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


plt.figure(figsize=(12,8))
sns.countplot(x="days_since_prior_order", data=orders_df, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()

print(orders_df.days_since_prior_order.value_counts())


# percentage of re-orders in prior set #
order_products_prior_df.reordered.sum() / order_products_prior_df.shape[0]


# percentage of re-orders in train set #
order_products_train_df.reordered.sum() / order_products_train_df.shape[0]

grouped_df = order_products_prior_df.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_df["reordered"].ix[grouped_df["reordered"]>1] = 1
grouped_df.reordered.value_counts() / grouped_df.shape[0]

grouped_df = order_products_train_df.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_df["reordered"].ix[grouped_df["reordered"]>1] = 1
grouped_df.reordered.value_counts() / grouped_df.shape[0]


grouped_df = order_products_train_df.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_df.add_to_cart_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of products in the given order', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


products_df.head()

aisles_df.head()

departments_df.head()

order_products_prior_df.head()

order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')
order_products_prior_df.head()

cnt_srs = order_products_prior_df['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
cnt_srs


cnt_srs = order_products_prior_df['aisle'].value_counts().head(20)
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
print(cnt_srs)

plt.figure(figsize=(10,10))
temp_series = order_products_prior_df['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()


grouped_df = order_products_prior_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


grouped_df = order_products_prior_df.groupby(["department_id", "aisle"])["reordered"].aggregate("mean").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_df.reordered.values, grouped_df.department_id.values)
for i, txt in enumerate(grouped_df.aisle.values):
    ax.annotate(txt, (grouped_df.reordered.values[i], grouped_df.department_id.values[i]), rotation=45, ha='center', va='center', color='green')
plt.xlabel('Reorder Ratio')
plt.ylabel('department_id')
plt.title("Reorder ratio of different aisles", fontsize=15)
plt.show()

#print(grouped_df)


order_products_prior_df["add_to_cart_order_mod"] = order_products_prior_df["add_to_cart_order"].copy()
order_products_prior_df["add_to_cart_order_mod"].ix[order_products_prior_df["add_to_cart_order_mod"]>70] = 70
grouped_df = order_products_prior_df.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()



order_products_train_df = pd.merge(order_products_train_df, orders_df, on='order_id', how='left')
grouped_df = order_products_train_df.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df['order_dow'].values, grouped_df['reordered'].values, alpha=0.8, color=color[3])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Reorder ratio across day of week", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()

grouped_df = order_products_train_df.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df['order_hour_of_day'].values, grouped_df['reordered'].values, alpha=0.8, color=color[4])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Reorder ratio across hour of day", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()


grouped_df = order_products_train_df.groupby(["order_dow", "order_hour_of_day"])["reordered"].aggregate("mean").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'reordered')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("Reorder ratio of Day of week Vs Hour of day")
plt.show()

print(grouped_df)








