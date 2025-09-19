# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:14:03 2019

@author: echtpar
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt



fs = 500 # sample rate 
f = 50 # the frequency of the signal

x = np.arange(fs) # the points on the x axis for plotting
print(np.pi*f*(x/fs))
print(np.sin(np.pi*f*(x/fs)))
print(np.sin(1399.57952717))
print(x)
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*(np.pi+np.pi/2)*f * ((x)/fs)) 
y2 = np.sin(2*np.pi*f*2 * (x/fs)) 
y3 = y+y2

z = np.column_stack((x, y))

print(z)

"""

x = np.zeros([200,200])
x[:200,:200] = 0

#x[:10, :10] = 1
#x[10:150, 10:150] = 1
#x[900:1300, 900:1300] = 1
#x[10:20, 10:20] = 1


for i in range(399):
    a = i+1
    b = int(np.abs((160000 - a*a)**(1/2)))
    print(a, b)
    x[:a,:b] = 1
    

for i in range(399):
    a = i+20
    b = int(np.abs((160000 - a*a)**(1/2)))
    print(a+400, b)
    x[:a+400,:b] = 1
    
#x[1000, 1000] = 1
x[160:170, 160:170] = 1
#x[:, 0:50] = 1
#x[:, 100:150] = 1
#x[:, 200:250] = 1
#x[:, 300:350] = 1
#x[:, 400:450] = 1
#x[:, 500:550] = 1
#x[:, 600:650] = 1
#x[:, 700:750] = 1
#x[:, 800:850] = 1
#x[:, 900:950] = 1
#x[:, 1000:1050] = 1
#
#x[:, 1100:1150] = 1
#x[:, 1200:1250] = 1
#x[:, 1300:1350] = 1
#x[:, 1400:1450] = 1
#x[:, 1500:1550] = 1
#
#x[:, 1600:1650] = 1
#x[:, 1700:1750] = 1
#x[:, 1800:1850] = 1
#x[:, 1900:1950] = 1




#x[115, :] = 1
#x[120, :] = 1
#x[125, :] = 1
#x[130, :] = 1
"""
cv2.namedWindow('original-y', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original-y', 400,400)
cv2.imshow('original-y', y)

cv2.namedWindow('original-y2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original-y2', 400,400)
cv2.imshow('original-y2', y2)

cv2.namedWindow('original-y3', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original-y3', 400,400)
cv2.imshow('original-y3', y3)


#plt.stem(x,y, 'r', )
#plt.plot(x,y)


#X = np.fft.fft2(x);
#X = np.fft.fftshift(x);
X = np.fft.fft(y);
X2 = np.fft.fft(y2);
X3 = np.fft.fft(y3);

#print(X.shape)
#print(X[:3,:3])
#print(X[295:300,295:300])
#print(X[495:500,495:500])
"""
z = 0
for i in range(200):
    for j in range (200):
        if X[i,j] != 0:
            z = z+1
            print(X[i,j], z)

"""

cv2.namedWindow('fft1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('fft1', 400,400)
#cv2.imshow('fft', np.log(abs(X) + 1))
cv2.imshow('fft1', np.log(abs(X) + 1))
#cv2.imshow('image', np.fft.ifftshift(np.log(abs(X) + 1),None))


cv2.namedWindow('fft2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('fft2', 400,400)
#cv2.imshow('fft', np.log(abs(X) + 1))
cv2.imshow('fft2', np.log(abs(X2) + 1))


cv2.namedWindow('fft3', cv2.WINDOW_NORMAL)
cv2.resizeWindow('fft3', 400,400)
#cv2.imshow('fft', np.log(abs(X) + 1))
cv2.imshow('fft3', np.log(abs(X3) + 1))



#f_ishift = np.fft.ifftshift(fshift)
#X[3:198,3:198] = (-8.08420622494-0.381241487783j)
#X[0:100,0:100] = (1.08420622494+1.381241487783j)
#X[101:200,101:200] = (500.08420622494+500.381241487783j)
"""
X[:200,:200] = 0
X[0:10,0:10] = (5+5j)
X[11:20,11:20] = (7+7j)
X[21:30,21:30] = (0-9j)
X[31:40,31:40] = (9+9j)
X[41:50,41:50] = (1-9j)
X[51:60,51:60] = (3.2+9j)
X[61:70,61:70] = (4-9j)
X[71:80,71:80] = (9+9j)
X[81:90,81:90] = (9-9j)
X[91:100,91:100] = (9+9j)
X[101:30,21:30] = (9-9j)
X[101:150,101:150] = (15+15j)
X[151:200,151:200] = (100+100j)
cv2.imshow('fft2', np.log(abs(X) + 1))
print(type(X))


#print(X[199,199])
img_back = np.fft.ifft2(X)
img_back = np.log(np.abs(img_back))

cv2.namedWindow('image-recovered', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image-recovered', 400,400)
cv2.imshow('image-recovered', img_back)
"""

