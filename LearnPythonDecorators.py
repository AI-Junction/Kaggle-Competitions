# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:00:53 2017

@author: echtpar
"""

from functools import wraps
    
def add_wrapping_with_style(style):
    print(style)
    def add_wrapping(item):
    #    print ('entered add wrapping', item)
#        @wraps(item)
        def wrapped_item():
    #        print ('entered wrapped item')
            return 'a {} wrapped box of {}'.format(style, str(item()))
    #    print('before calling wrapped_item()')
        return wrapped_item
    return add_wrapping
    
#@add_wrapping        
#@add_wrapping    
#@add_wrapping
def new_bicycle():
#    print ('in new bicycle')
    return 'a new bicycle'

@add_wrapping_with_style('beautifully')
def new_gpu():
#    print ('in new gpu')
    return 'a new Tesla P100 GPU'

#@add_wrapping
def new_laptop():
#    print ('in new laptop')
    return 'a new apple mac'

    
#print('begin')
print(new_gpu())
print(new_bicycle())   
print(new_gpu.__name__)
    