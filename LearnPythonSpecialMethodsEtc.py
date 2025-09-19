# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:44:45 2017

@author: echtpar
"""
import numpy as np

x = (i for i in range(5))

next (x)
next (x)

x.__next__()
x.__next__()

dir(x)
dir(range)

dir(np.array)
