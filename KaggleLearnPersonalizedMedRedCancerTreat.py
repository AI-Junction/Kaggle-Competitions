# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:05:43 2018

@author: echtpar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn
import sklearn.pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import feature_extraction
from sklearn import decomposition


input_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllPersonalizedMedicineData\\"

train_file = "training_variants"
test_file = "test_variants"
trainx_file = "training_text"
testx_file = "test_text"


'''
text_file = open(input_path + train_file, "r", encoding = "utf8")
lines = text_file.readlines()
print (lines[:10])
print(type(lines))
print (len(lines))
text_file.close()

text_file = open(input_path + trainx_file, "r", encoding = "utf8")
lines = text_file.read().splitlines()
lines = text_file.readlines()
print (lines[:10])
print(type(lines))
print (len(lines))
text_file.close()
'''

train = pd.read_csv(input_path + train_file)
test = pd.read_csv(input_path + test_file)
trainx = pd.read_csv(input_path + trainx_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv(input_path + testx_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

print(train.columns.values)
print(test.columns.values)
print(test.columns)
print(type(test.columns.values))
print(type(test.columns))

print(trainx.columns.values)
print(testx.columns.values)

print(train.shape)
print(test.shape)
print(trainx.shape)
print(testx.shape)



'''
print(trainx['Text'][:10].apply(lambda x: str (x[:20])))
print(trainx.count())
print(len(trainx))
print(*trainx)
print(*trainx.shape)
print(trainx.columns.values)
print(trainx.index)
print(type(trainx.index.values))
print(set(trainx.index.values))
print(np.random.randn(*trainx.shape)[:10])
'''

'''
print(type(train))
df_cols = train.dtypes.reset_index()
df_cols.columns = ['col', 'type']
print(df_cols.columns.values)
df_cols_grouped = df_cols.groupby(['type']).aggregate('count').reset_index()
print(df_cols_grouped)
print(train.columns.values)
print(df_cols)
print(len(set(train['Gene'])))
'''

'''
df_train_group = train.groupby(['Gene'])['ID'].aggregate('count').reset_index()
print(df_train_group)
z = train.loc[train['Gene'] == 'RET', ['Gene', 'ID']]
print(z)
w = train.iloc[:10, :]
print(w)
for rec in enumerate(np.array(w)):
#    print(len(rec))
#    print(type(rec))
    print(np.array(rec)[1][2])
#    print(list(rec))

'''
'''
df_train_filled = train.fillna('WOW')
df_trainx_filled = trainx.fillna('WOW')
print(df_trainx_filled.columns.values)
print(df_trainx_filled.loc[df_trainx_filled['Text'] == 'WOW', ['Text']][:100])
'''

#train_merged.plot.hist()

#train_merged.plot.hist()

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
print(y[:10])
print(type(train.values))
print(train.values.shape)
print(type(train))
print(train.shape)

train = train.drop(['Class'], axis = 1)
print(train.columns.values)


test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
print(pid[:10])




df_all = pd.concat((train, test), axis=0, ignore_index=True)

'''
#tmp1 = df_all.loc[:,['Text', 'ID']][1].split(' ')
txt = df_all['Text'][1].split(' ')
gene = df_all['Gene'][1].split(' ')
#print(gene)
df_all['Text'][1]
df_all['Gene'][1]
print(txt)
gene = 'c-CBL'
print(txt)
gene = gene.split(' ')
print([w for w in txt if w in gene])
print(tmp)
'''

print(df_all['Gene'][10:])
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
#df_all.loc[df_all.Gene_Share==0]['Gene_Share']
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

#df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
#df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)


'''
print(df_all.columns.values)
print(df_all['Variation_Share'][:10])
print(max(df_all['Variation_Share']))
print(max(df_all['Gene_Share']))



df_all['Gene'].map(lambda x: x[:-1])
df_all['Gene']
print(df_all['test'][:10])

df_all['Gene'].apply(lambda x: x[:-1])

'''

'''
df_all['Gene'][:10].apply(lambda x: x[:-1])
print(df_all['Gene'].apply(lambda x: x[:-1]).unique())

print(type(df_all['Gene'].apply(lambda x: x[:-1]).unique()))

print(set(df_all['Gene'].apply(lambda x: x[:-1])))
print(type(set(df_all['Gene'].apply(lambda x: x[:-1]))))

print(list(set(df_all['Gene'].apply(lambda x: x[:-1]))))

#print(df_all.values)

df_all['Gene']

np.max(df_all['Gene'].apply(lambda x: len(x)))
np.max(df_all['Variation'].apply(lambda x: len(x)))
'''
'''
for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'][:10].apply(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'][:10].apply(lambda x: str(x[i]) if len(x)>i else '')


tmp = df_all.loc[not np.isnan(df_all.Gene_9)]
print(np.isnan(df_all['Gene_9'][17]))


print(df_all.Gene_9)        
df_all.loc[pd.isnull(df_all['Gene_9']) == False]
'''

#commented for Kaggle Limits
for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].apply(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].apply(lambda x: str(x[i]) if len(x)>i else '')

'''
print(df_all.columns.values)

z = df_all.loc[df_all.Gene_Share == 1 ].index
w = df_all.loc[df_all.Gene_Share == 1 ]
q = df_all.loc[df_all.Gene_Share == 0 ]

print(q.shape)
print(z.shape)
print(w.shape)
print(df_all.shape)
#z = np.any(np.isnan(df_all)) #.sum(axis=0).reset_index()

print(z.values)
df_all[z]
'''
'''
a = list(train.Gene.unique())
b = set(train.Gene.unique())
c = set(train.Gene)
d = train.Gene.unique()

print(len(a))
print(type(a))
print(len(b))
print(type(b))
print(len(c))
print(type(c))
print(len(d))
print(type(d))

e = [1,2,3,4,5]
f = [6,7,8,9,10]

print(e+f)

g = ["a b c d e"]
print(len(g))
print(len(str(g).split(' ')))
'''
'''
h = "a b"
k = h.split(' ')
print(k)
print(h.count('b'))
'''

print(df_all.dtypes)
print(type(df_all.dtypes))
pd_dtype = df_all.dtypes.to_frame().reset_index()
pd_dtype = df_all.dtypes.reset_index()
print(type(pd_dtype))
pd_dtype.columns = ['col', 'type']
print(pd_dtype)
pd_groupby = pd_dtype.groupby(['type'])['type'].aggregate('count')
pd_groupby_df = pd_dtype.groupby(['type'])['type'].aggregate('count').to_frame()
print(pd_groupby)

pd_groupby_df = pd_groupby_df.reset_index()
pd_groupby_df.columns = ['type', 'count']


print(pd_groupby_df)

print(type(pd_groupby))
print(type(pd_groupby_df))


z = df_all['Gene'].apply(str)
print(type(z[0]))
print(type(df_all['Gene'][0]))



gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0
#commented for Kaggle Limits
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1

print(type(df_all.columns.values))
print(type(df_all.columns))
print(df_all.columns.values.shape)    
print(df_all.columns.shape)    
    
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')

#fp = pipeline.Pipeline([
#    ('union', pipeline.FeatureUnion(
#        n_jobs = -1,
#        transformer_list = [
#            ('standard', cust_regression_vals()),
#            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
#            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
#            #commented for Kaggle Limits
#            #('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
#        ])
#    )])



tmp = cust_regression_vals()

tmp = cust_regression_vals(train)

print(tmp)
fp = Pipeline([
    ('union', FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            #('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
        ])
    )])


pipe = ('pi1', Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))]))

tmp = cust_regression_vals()
tmp1 = tmp.fit_transform(df_all)
print(tmp1.shape)
print(type(tmp1))



l = cust_txt_col('Gene').fit_transform(df_all)
print(type(l[1]))
print(l[:])
print(type(df_all['Gene'][1]))
m = feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))
n = m.fit_transform(l)
q = m.get_feature_names()
print(q)
print(len(q))
print(type(q))
print(np.sort(q))
print(len(pd.unique(q)))

r = feature_extraction.text['a,b,c,']
s = decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12)
t = s.fit(n)
print(t)
u = s.transform(n)
print(u.shape)
print(n.shape)
'''


train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

y = y - 1 #fix for zero bound array

denom = 0
fold = 1 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)    



import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model); plt.show()