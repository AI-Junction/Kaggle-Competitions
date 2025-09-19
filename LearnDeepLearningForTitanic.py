# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:17:54 2018

@author: echtpar
"""

import numpy as np 
import pandas as pd 
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


train = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\train.csv')
test = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\test.csv')

train.shape
test.shape

train.columns


#train = pd.read_csv("../input/train.csv")
#test = pd.read_csv("../input/test.csv")
# start with some EDA
train.head()
train.columns
print(train['Embarked'].unique(), train['Pclass'].unique())
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for df in [train, test]:
    df.drop(labels=["PassengerId", "Cabin", "Name", "Ticket"], axis=1, inplace=True)
    
    
# fill missing value
# another way is to use sklearn imputation to fill na
for df in [train, test]:
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(np.mean(df[col]))
        
        
# data normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
for df in [train, test]:
    for col in ["Age", "Fare"]:
        x = df[[col]].values.astype(float)
        df[col] = min_max_scaler.fit_transform(x)



# transform string
for df in [train, test]:
    # remove duplicate features to avoid overfitting
    df['is_male'] = np.where(df['Sex']=="male", 1, 0)
    df['is_female'] = np.where(df['Sex']=="female", 1, 0)
    df['EmbarkedS'] = np.where(df['Embarked']=="S", 1, 0)
    df['EmbarkedC'] = np.where(df['Embarked']=="C", 1, 0)
    df['EmbarkedQ'] = np.where(df['Embarked']=="Q", 1, 0)
    df['Pclass1'] = np.where(df['Pclass']==1, 1, 0)
    df['Pclass2'] = np.where(df['Pclass']==2, 1, 0)
    df['Pclass3'] = np.where(df['Pclass']==3, 1, 0)
    df['is_single'] = np.where(np.logical_and(df['SibSp']==0, df['Parch']==0), 1, 0)




# then remove transformed columns
for df in [train, test]:
    df.drop(labels=["Sex", "Embarked", 'Pclass'], axis=1, inplace=True)



# load up train/validation set! 
train_size = int(train.shape[0] * 0.85)

train_dataset = train[:train_size]
val_dataset = train[train_size:]

X_train = train_dataset.drop(labels=["Survived"], axis=1).values
Y_train = train_dataset["Survived"].values

X_val = val_dataset.drop(labels=["Survived"], axis=1).values
Y_val = val_dataset["Survived"].values

input_size = len(train_dataset.columns) - 1  # number of final features 

X_train.shape
train.head()



from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import np_utils
from keras import optimizers, metrics #,losses, 

model = Sequential()
k_init = 'glorot_uniform'
optimizer = optimizers.Adam() #'rmsprop' #optimizers.Adam()

model.add(Dense(64,input_dim=input_size, init=k_init))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(64, init=k_init))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(1, init=k_init))
model.add(Activation("sigmoid"))  # or softmax with category_crossentropy

model.compile(loss="binary_crossentropy", optimizer=optimizer, 
              metrics=[metrics.binary_accuracy]) 

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    nb_epoch=200, batch_size=100, 
                    verbose=0)
# loss_and_metrics = model.evaluate(X_val, Y_val_cat)
# print(loss_and_metrics)

# pre-selected paramters
# best_epochs = 200
# best_batch_size = 5
# best_init = 'glorot_uniform'
# best_optimizer = 'rmsprop'



acc = history.history['binary_accuracy']
print(acc)
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig = plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss,'b', label='Validation loss')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
 
#plt.clf()

epochs = range(1, len(acc) + 1)
fig = plt.figure()
plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc,'b', label='Validation Acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()


y_final = model.predict_classes(test.values).reshape(-1) # ravel()
y_final

#df_test = pd.read_csv("../input/test.csv")

df_test = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTitanicData\\test.csv')

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
surv_num = sum(output["Survived"] != 0) / len(output)
#print(f"Survive ratio: {surv_num}")


output.to_csv('prediction.csv', index=False)
output

        