# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:02:10 2017

@author: echtpar
"""

import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
'''
#uncomment this section for basic section
style.use('ggplot')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 1, 1)

df = web.DataReader("XOM", "yahoo", start, end)
print(df.head())

df['Adj Close'].plot()

plt.show()
'''

web_stats = {'Day':[1,2,3,4,5,6],
             'visitors':[43,53,34,45,64,34],
            'Bounce_Rate':[65, 72, 62, 64, 54, 66]}

#df = pd.DataFrame(np.array(pd.DataFrame(web_stats)))
df = pd.DataFrame(web_stats)
print(df)


#print(df)
#print(df.head(3))
#print(df.tail(2))
#
#print(df.set_index('Day'))
#print(df.head())
#
#df = df.set_index('Day')
#print(df.head())
#
#df2 = df.set_index('Day')
#print(df2)
#
#df.set_index('Day', inplace = True)
#print(df.head())

print(df['visitors'])
print(df.visitors)

print(df.shape)

print(df.columns.values)


print(df[['Bounce_Rate','visitors']])

print(df.visitors.tolist())

print(np.array(df[['Bounce_Rate','visitors']]))

df2 = pd.DataFrame(np.array(df[['Bounce_Rate','visitors']]))
print(df2)
print(df2[0])
