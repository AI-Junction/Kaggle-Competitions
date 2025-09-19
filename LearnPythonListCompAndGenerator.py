# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 06:36:35 2017

@author: echtpar
"""

#for i in range(5):
    
#abc = [i for i in range(500000)] 
#print(abc)

#abc_gen = (i for i in range(500000))       
#print(abc_gen)

#for i in abc_gen:
#    print(i)


#xyz = []       
#for i in range(5):
#    xyz.append(i)
#
#print (xyz)    
#    

import random
import numpy as np

input_list = [5,6,2,10,15,20,5,2,1,3]

def div_by_five(num):
    if num%5==0:
        return True
    else:
        return False

xyz = [i for i in input_list if div_by_five(i)]
print(xyz)

for i in input_list:
    print (i)
        
xyz = (i for i in input_list if div_by_five(i))
print(xyz)

for i in xyz:
    print(i)

#xyz = []
#
#for i in input_list:
#    if div_by_five(i):
#        xyz.append(i)
        
#for i in xyz:
#    print(i)
    
abc = [i for i in (i for i in input_list if div_by_five(i))]
print(abc)
       
#print(xyz)

xyz = [i for i in input_list if div_by_five(i)]
print(xyz)       

[print(i) for i in xyz]

i=0
ii =0
[[print (i, ii) for ii in range(5)] for i in range(5)]

for i in range(5):
    for ii in range(5):
        print(i, ii)

i=0
ii =0

xyz = [[ii in range(5)] for i in range(5)]

#print(xyz.shape)
print(xyz)


for i in xyz:
    for ii in i:
        print(i)
 
a=0
b=[]        

b = (i for i in range(5))

for z in b:
    print (z)
    

c = [i for i in range(5) for i in range(5)]
print (c)     


c = (i for i in range(5) for i in range(5))
for k in c:
    print (k)     


    
c = [[i for i in range(5)] for i in range(5)]
print (c)     

c = ([i for i in range(5)] for i in range(5))

for m in c:
    print(m)
   
print (m)    

d = ((i for i in range(5)) for i in range(5))
print (d)

[print([[n for n in m] for m in d])]



for m in d:
#    print(m)
    for n in m:
        print (n)
print (m)    

#q = [[[4,5], [6,7], [8,9], [10,11], [12,13]], [[14,15], [16,17]], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

#define a list of lists
q = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

#print the list items
for i in q:
    print (i)

#print all elements in the list of list    
for i in q:
    for ii in i:
        print (ii)


#print by specifying index of the element
print(q[0])    
print(q[0][1])    
print(q[0][1][1])    

#convert list to numpy array
p = np.array(q)

#print by specifying index of the element in the numpy array
print(p.shape)
print(p[0])    
print(p[0][1])    
print(p[0][1][1])    

#reshape the numpy array into one row
r = np.reshape(p, [1,-1])
print(r)
print(r.shape)

#reshape the numpy array into one column
s = np.reshape(p, [-1,1])
print(s)
print(s.shape)


#flatten, convert numpy array to list, 
t = p.flatten()
w = t.tolist()
print(type(w))
print(t)
print(p)
print(w)
print(t.shape)

#for appending an element to numpy array - we need to convert it into a list
w.append(100)
print(len(w))


u = np.reshape(t,[3,5])
print(u)

v = np.reshape(t,(5,3))
print(v)
print(type(v))

print(p)
