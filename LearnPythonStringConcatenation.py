# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:55:01 2017

@author: echtpar
"""

names = ['Jeff', 'Gary', 'Jill', 'Samantha']

for name in names:
    print('Hello there, ' + name)
    print (' '.join(['Hello there,', name]))    

print(', '.join(names))   

import os

location_of_file = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial' 

file_name = 'PythonStringConcat.txt'

file_location = location_of_file + '\\' + file_name

print(location_of_file + '\\' + file_name)
print(file_location)

with open(os.path.join(location_of_file, file_name)) as f:
    print (f.read())
    
who = 'Gary'
how_many = 12

#sentence = Gary bought 12 apples today

#sentence = 

print(who, 'bought', how_many, 'apples today!')

print('{0} bought {1} apples today!'.format(who, how_many))
print('{1} bought {0} apples today!'.format(who, how_many))

NewDict = {}
NewDict.index.add[1]
