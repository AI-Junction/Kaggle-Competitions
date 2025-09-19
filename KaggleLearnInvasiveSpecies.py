# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:33:45 2017

@author: echtpar
"""

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
from tqdm import tqdm

path_master = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\train_labels.csv"
#master = pd.read_csv("../input/train_labels.csv")
master = pd.read_csv(path_master)
master.head()



#img_path = "../input/train/"

img_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\train\\train\\"
print(len(master))
print(master.ix[2][1])

y = []
file_paths = []
for i in range(len(master)):
    file_paths.append( img_path + str(master.ix[i][0]) +'.jpg' )
    y.append(master.ix[i][1])
y = np.array(y)

print(y[:5])


#image reseize & centering & crop 

def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized


x = []


for i, file_path in tqdm(enumerate(file_paths), miniters = 100):
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    x.append(img)

print(x[0].shape)
x = np.array(x)

path_submission = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\sample_submission.csv(1)\\sample_submission.csv"

sample_submission = pd.read_csv(path_submission)

img_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\test-1\\test\\"
#img_path = "../input/test/"

test_names = []
file_paths = []

#print(sample_submission.loc[1][0])

for i in range(len(sample_submission)):
    test_names.append(sample_submission.ix[i][0])
    file_paths.append( img_path + str(int(sample_submission.ix[i][0])) +'.jpg' )
    

test_names = np.array(test_names)

test_images = []
for file_path in tqdm(file_paths,miniters = 100):
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    test_images.append(img)
    
    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)




#np.savez('224.npz', x=x, y=y, test_images=test_images, test_names=test_names)


data_num = len(y)
print(data_num)
random_index = np.random.permutation(data_num)

print(random_index[:10])

x_shuffle = []
y_shuffle = []
for i in range(data_num):
    x_shuffle.append(x[random_index[i]])
    y_shuffle.append(y[random_index[i]])
    
x = np.array(x_shuffle) 
y = np.array(y_shuffle)


val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

#print(x_shuffle[:10])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
print(base_model.output_shape[1:])

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(input=base_model.input, output=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 5

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen.fit(x_train)


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
#    samples_per_epoch=x_train.shape[0] // batch_size,
    samples_per_epoch=x_train.shape[0],
    nb_epoch=epochs,
    validation_data=(x_test, y_test),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
)



test_images = test_images.astype('float32')
test_images /= 255



predictions = model.predict(test_images)


#sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission = pd.read_csv(path_submission)


for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("submit.csv", index=False)





######################
"""
AlGiLa
Inception v3 and k-fold in Python (0.98996)
AlGiLa
Invasive Species Monitoring
voters
last run 7 days ago · Python script · 156 views
using data from Invasive Species Monitoring ·
Public
"""
######################








# kaggle kernel of AlGiLa
# I need to thanks the following contributors inspiring me on this kernel

# the function implementing k-fold come from the kernel provided by Finlay Liu 
# The idea to use Inception v3 come from the kernel provided by Ogurtsov it was in R, I wrote it in python

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn import model_selection
from sklearn import metrics


import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras import applications
from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

#train_set = pd.read_csv('../input/train_labels.csv')
#test_set = pd.read_csv('../input/sample_submission.csv')


path_test = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\test-1\\test\\"
path_train_set = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\train_labels.csv"
path_train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\train\\train\\"
path_test_set = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllInvasiveSpeciesMonitoringData\\sample_submission.csv(1)\\sample_submission.csv"

train_set = pd.read_csv(path_train_set)
test_set = pd.read_csv(path_test_set)



def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  #needed for Inception v3
    return img

print(train_set[:10])

for img_path1 in tqdm(train_set['name']):
    print(img_path1)
    
train_img, test_img = [], []
for img_path in tqdm(train_set['name'].iloc[: ]):
    train_img.append(read_img(path_train + str(img_path) + '.jpg'))
for img_path in tqdm(test_set['name'].iloc[: ]):
    test_img.append(read_img(path_test + str(img_path) + '.jpg'))

    
    
train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255

#Transfer learning da Inception V3 traino solo gli ultimi fully connected layers
img_rows, img_cols, img_channel = 224, 224, 3
base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(img_rows, img_cols, img_channel))
print(base_model.summary())


#Adding custom Layers
add_model = Sequential()
add_model.add(Dense(1024, activation='relu',input_shape=base_model.output_shape[1:]))
add_model.add(Dropout(0.60))
add_model.add(Dense(1, activation='sigmoid'))
print(add_model.summary())

# creating the final model
model = Model(input=base_model.input, output=add_model(base_model.output))

# compile the model
opt = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = optimizers.SGD(lr = 0.0001, momentum = 0.8, nesterov = True)
# meglio la funzione sotto che lega il learning rate decay al monitor che si vuole
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              patience=5,
                              verbose=1,
                              factor=0.1,
                              cooldown=10,
                              min_lr=0.00001)  # funzione di kesar to reduce learning rate (new_lr = lr*factor)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
print(model.summary())

n_fold = 5
kf = model_selection.KFold(n_splits = n_fold, shuffle = True)
eval_fun = metrics.roc_auc_score

def run_oof(tr_x, tr_y, te_x, kf):
    preds_train = np.zeros(len(tr_x), dtype = np.float)
    preds_test = np.zeros(len(te_x), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    for train_index, test_index in tqdm(kf.split(tr_x)):
        x_tr = tr_x[train_index]; x_te = tr_x[test_index]
        y_tr = tr_y[train_index]; y_te = tr_y[test_index]

        datagen = ImageDataGenerator(
            # featurewise_center = True,
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            # zca_whitening = True,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)

#        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 64),
#            validation_data = (x_te, y_te), callbacks=[reduce_lr],
#            steps_per_epoch = len(train_img) / 64, epochs = 45, verbose = 2)

        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 64),
            validation_data = (x_te, y_te), callbacks=[reduce_lr],
            samples_per_epoch = len(x_tr), nb_epoch = 5, verbose = 2)


        train_loss.append(eval_fun(y_tr, model.predict(x_tr)[:, 0]))
        test_loss.append(eval_fun(y_te, model.predict(x_te)[:, 0]))

        preds_train[test_index] = model.predict(x_te)[:, 0]
        preds_test += model.predict(te_x)[:, 0]

        print('{0}: Train {1:0.5f} Val {2:0.5f}'.format(i, train_loss[-1], test_loss[-1]))
        i += 1

    print('Train: ', train_loss)
    print('Val: ', test_loss)
    print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(test_loss)))
    preds_test /= n_fold
    return preds_train, preds_test

train_pred, test_pred = run_oof(train_img, train_label, test_img, kf)

test_set['invasive'] = test_pred
test_set.to_csv('../submit.csv', index = None)

