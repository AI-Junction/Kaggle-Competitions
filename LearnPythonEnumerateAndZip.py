# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:51:58 2017

@author: Chandrakant Pattekae
"""

example = ['left', 'right', 'up', 'down']

for i in range(len(example)):
    print(i, example[i])
    
for i, j in enumerate(example):
    print(i,j)
    
new_dict = dict(enumerate(example))

print(new_dict)

[print(i,j) for i, j in enumerate(new_dict)]

[print(i,new_dict[j]) for i, j in enumerate(new_dict)]

print(new_dict[1]) 

x = [1,2,3,4]
y = [7,8,9,10]
z = ['a','b','c','d']

for a, b, c in zip(x,y,z):
    print(a,b,c)
    
for i in zip(x,y,z):
    print(i)
    
j = list(zip(x,y,z))    
print(j)

k = dict(zip(x,z))    
print(k)


print (len(x))
