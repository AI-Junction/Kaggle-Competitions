# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 23:29:03 2017

@author: echtpar
"""

import quandl
import pandas as pd

api_key = open('quandlapikey.txt', 'r').read()
#df = quandl.get('FMAC/HPI_AK', authtoken=api_key)
#print(df.head())

fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

#This is a list
print(fiddy_states)

#This is a dataframe
print(fiddy_states[0])

#this is a column
print(fiddy_states[0][2])

print(fiddy_states[0][0][1:])

for abbv in fiddy_states[0][0][1:]:
    print('FMAC/HPI_'+abbv)
    

