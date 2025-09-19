# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:12:03 2017

@author: echtpar
"""
import timeit

input_list = range(100)
print([input_list])

def div_by_five(num):
    if num%5==0:
        return True
    else:
        return False

xyz = [i for i in input_list if div_by_five(i)]

xyz = (i for i in input_list if div_by_five(i))
       
print(timeit.timeit(''' 
input_list = range(100)

def div_by_five(num):
    if num%5==0:
        return True
    else:
        return False

xyz = [i for i in input_list if div_by_five(i)]
''', number = 5000))
