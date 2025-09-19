# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:39:23 2017

@author: echtpar
"""

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.html.widgets import interact


digits = load_digits()     
X = digits.data
y = digits.target

    
def plot_digits(n_components):
    plt.figure(figsize = (8,8))     
    nside = 10
    
    pca = PCA(n_components).fit(X)
    Xproj = pca.inverse_transform(pca.transform(X[:nside**2]))
    Xproj = np.reshape(Xproj, (nside, nside, 8, 8))
    total_var = pca.explained_variance_ratio_.sum()
    
    im = np.vstack([np.hstack([Xproj[i,j] for j in range(nside)])
            for i in range(nside)])
    
    plt.show(im)
    plt.grid(False)
    plt.title("n={0}, variance = (1:.2f)".format(n_components, total_var, size = 18))
    plt.clim(0,16)

interact(plot_digits, n_components = [1, 64], nside = [1,8])
    