# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 06:53:57 2017

@author: echtpar
"""


#from keras.datasets import mnist
#from sklearn.decomposition import PCA
#
#
## the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
##print(len(X_train[0][0])*len(X_train[0][0]))
#
#X_train.reshape(len(X_train), 784)
##X_train.flatten('A')
#
#print(shape(X))
#print(len(X_train[0][0]))
#
#
#pca = PCA(n_components = 64)
#pca.fit(X)
#print(pca.explained_variance_)
#print(pca.components_)


from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.html.widgets import interact


digits = load_digits()     
X = digits.data
y = digits.target

print(X.shape)

pca = PCA(2)
Xproj = pca.fit_transform(X)

print(X.shape)
print(Xproj.shape)
print(Xproj[0])
print(np.unique(y))

plt.scatter(Xproj[:,0], Xproj[:,1],c=y, edgecolor = 'none', alpha = 0.5, 
            cmap = plt.cm.get_cmap('nipy_spectral', 10 ))

plt.colorbar()
plt.show()

pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.show()
print(X[20:22])
print(y[21])


fig, axes = plt.subplots(8,8, figsize = (8,8))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

for i, ax in enumerate(axes.flat):
    pca = PCA(i+1).fit(X)
    
    im = pca.inverse_transform(pca.transform(X[20:21]))
    print(pca.transform(X[20:21]).shape)
    ax.imshow(im.reshape((8,8)), cmap = 'binary')
    
    ax.text(0.95, 0.05, 'n={0}'.format(i+1), ha='right', 
            transform = ax.transAxes, color = 'green')
    
    ax.set_xticks([])
    ax.set_yticks([])

#
#fig, axes = plt.subplots(8,8, figsize = (8,8))
#fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
#
#pca = PCA(64).fit(X)
#eigenfaces = pca.components_.reshape((64, 8, 8))
#plt.imshow(eigenfaces[9].reshape((8, 8)), cmap=plt.cm.gray)
#print(eigenfaces[9].shape)
#
#im_inv_transform = pca.inverse_transform(pca.transform(X[20]))
#print(im_inv_transform)
#plt.imshow(X[20].reshape((8, 8)), cmap=plt.cm.gray)
#plt.imshow(im_inv_transform.reshape((8, 8)), cmap=plt.cm.gray)



#im = pca.transform(X)
#print(pca.transform(X).shape)
#ax.imshow(im)
#
#ax.text(0.95, 0.05, 'n={0}'.format(i+1), ha='right', 
#        transform = ax.transAxes, color = 'green')
#
#ax.set_xticks([])
#ax.set_yticks([])
#
#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)
#    