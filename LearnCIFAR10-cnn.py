'''Train a simple deep CNN on the CIFAR10 small images dataset.


It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

# plot import matplotlib
#import matplotlib
import matplotlib.pyplot as plt


batch_size = 128
nb_classes = 10
#nb_epoch = 50
nb_epoch = 1
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(y_test[:10])
print(y_test.shape)
print(X_train.shape)


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(Y_test[:10])
print(Y_test.shape)


i=6600

#plt.imshow(x_train[i,0], interpolation='nearest')
#plt.imshow(x_train[i,0], extent=[1,1,1,1], aspect='auto')
plt.imshow(X_train[i,:,:,:], interpolation='nearest')
print (X_train.shape[3])
print('y_label', Y_train[i,:])
print(type(y_test))
print(np.unique(y_test))


fig, axes = plt.subplots(12,12, figsize = (32,32))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

print(axes.shape)


# Plot the impages starting from i = 1
for i, ax in enumerate(axes.flat):
    a = i + 300
    im = X_train[a,:,:]
    ax.imshow(im, cmap = 'binary')
    ax.text(0.95, 0.05, 'n={0}'.format(i+1), ha='right', 
            transform = ax.transAxes, color = 'green')
    
    ax.set_xticks([])
    ax.set_yticks([])


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), shuffle=True)

score = model.evaluate(X_test, Y_test, verbose=0)
#print("Test score:", score[0])
print("Test accuracy:", score[1])
#print(model.predict_classes(X_test[1:150]))
z = model.predict_classes(X_test[0:149])
print(type(z))
m=np.reshape(z, (149,-1))
n = zip(m, y_test[0:149])
for i, ii in n:
    print (i, ii)


fig, axes = plt.subplots(12,12, figsize = (32,32))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

print(axes.shape)


# Plot the impages starting from i = 1
for i, ax in enumerate(axes.flat):
    a = i
    im = X_test[a,:,:]
    ax.imshow(im, cmap = 'binary')
    ax.text(0.95, 0.05, 'n={0}{1}'.format(y_test[i], m[i]), ha='right', 
            transform = ax.transAxes, color = 'yellow')
    
    ax.set_xticks([])
    ax.set_yticks([])

    

#if not data_augmentation:
#    print('Not using data augmentation.')
#    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), shuffle=True)
#else:
#    print('Using real-time data augmentation.')
#    # This will do preprocessing and realtime data augmentation:
##    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
##        samplewise_center=False,  # set each sample mean to 0
##        featurewise_std_normalization=False,  # divide inputs by std of the dataset
##        samplewise_std_normalization=False,  # divide each input by its std
##        zca_whitening=False,  # apply ZCA whitening
##        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
##        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
##        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
##        horizontal_flip=True,  # randomly flip images
##        vertical_flip=False)  # randomly flip images
#
#    datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,  featurewise_std_normalization=False,  samplewise_std_normalization=False,  zca_whitening=False,  rotation_range=0,  width_shift_range=0.1,  height_shift_range=0.1,  horizontal_flip=True,  vertical_flip=False)  # randomly flip images
#
#
#
#
#    # Compute quantities required for featurewise normalization
#    # (std, mean, and principal components if ZCA whitening is applied).
#    datagen.fit(X_train)
#
#    # Fit the model on the batches generated by datagen.flow().
#    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), samples_per_epoch=X_train.shape[0], nb_epoch=nb_epoch, validation_data=(X_test, Y_test))
#
## -*- coding: utf-8 -*-
#


#GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
#    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
