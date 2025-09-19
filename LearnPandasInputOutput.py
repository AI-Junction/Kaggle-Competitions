# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:31:58 2017

@author: echtpar
"""

import pandas as pd


df = pd.read_csv('ZILL-Z77006_MPC.csv')    

print(df.head())

df.set_index('Date', inplace = True)
print(df.head())
df.to_csv('newcsv.csv')

df2 = pd.read_csv('newcsv.csv', index_col=0)

print(df2.head())

df.columns = ['Austin_HPI']

print(df.head())

df.to_csv('newcsv3.csv')
df.to_csv('newcsv4.csv', header = False)

df = pd.read_csv('newcsv4.csv', names = ['Dates','Austin_HPI'], index_col=0)
print(df.head())


df.to_html('example.html')
df.rename(columns={'Austin_HPI':'77006_HPI'}, inplace=True)
print(df.head())
