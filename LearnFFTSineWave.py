# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:24:42 2019

@author: echtpar
"""

import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
import glob
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

Fs = 500.0;  # sampling rate
Ts = 0.5/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector


sample_rate = 1024
dt = 1.0/sample_rate
t = np.arange(sample_rate)*dt  # 1 second of samples
freq = 5
amp = 1.0
sine1 = amp*np.sin(2*np.pi*freq*t)
sine2 = .5*np.sin(2*np.pi*15*t)

ff = 5;   # frequency of the signal
y1 = 0 #+ np.sin(2*np.pi*ff*t)
y2 = np.sin(2*np.pi*10*ff*t)
y3 = np.sin(2.5*np.pi*8*ff*t)
y4 = np.sin(1.5*np.pi*7*ff*t)
y5 = np.sin(0.5*np.pi*6*ff*t)
y6 = np.sin(2*np.pi*5*ff*t)
y7 = np.sin(2*np.pi*4*ff*t)
y8 = np.sin(2*np.pi*3*ff*t)
y9 = np.sin(-10*np.pi*3*ff*t)

#jpgFiles = glob.glob("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\FFTImages\\Image2\\Slide12.jpg")



#y = y1+y2+y3+y4+y5+y6+y7+y8+y9
y = sine1+sine2
n = len(y) # length of the signal
#y=jpgFiles
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int((n/2)))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

#plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')
plt.show()