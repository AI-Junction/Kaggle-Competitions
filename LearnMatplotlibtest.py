# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:52:16 2017

@author: echtpar
"""

import matplotlib.pyplot as plt
import numpy as np


x = [1,2,3,4,5,6,7,8,9,9,9,10,10,10.5,10.8]
y = [11,12,13,14,15,17,19,20,25,27,28,29,20,31,32]


X = np.array(x)
Y = np.array(y)

print(X.shape)
print(Y.shape)

print(y)

z = zip(x, y)
print(z)

a = np.array([x,y])

print(a)

plt.plot(x,y)
plt.scatter(x,y)

plt.show()
