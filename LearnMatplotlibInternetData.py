# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 06:41:28 2017

@author: echtpar
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import urllib
import matplotlib.dates as mdates

def bytespdate2num(fmt, encoding = 'utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter
    
    
        

def graph_data(stock):
    
    fig = plt.figure()
#    style.use('fivethirtyeight')
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    
    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + stock + '/chartdata;type=quote;range=10y/csv' 
    source_code = urllib.request.urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')
    
    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6:
            if 'values' not in line and 'labels' not in line:
                stock_data.append(line)
                
    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data, 
                                                                 delimiter = ',',
                                                                 unpack = True,
                                                                 # %Y = Full Year 2015
                                                                 # %y = Partial Year 15
                                                                 # %m = number month
                                                                 # %d = number day
                                                                 # %H = hours
                                                                 # %M = minutes
                                                                 # %S = seconds
                                                                 # 2016-06-30: %Y-%m-%d 
                                                                 converters = {0:bytespdate2num('%Y%m%d')})

#    plt.style('ggplot')
#    plt.plot_date(date, closep, '-', label='Price')

    ax1.plot_date(date, closep, '-', label='Price')
    ax1.grid(True, color = 'g', linestyle = '-', linewidth = 1 )
    
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph \nCheck It Out', color = 'r')
    plt.subplots_adjust(left=0.09, bottom= 0.18, right = 0.94, top = 0.85, wspace = 0.2, hspace = 0)
#    plt.legend(color = 'r')
    plt.legend()

graph_data('TSLA')