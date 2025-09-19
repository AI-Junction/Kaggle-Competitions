# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:57:17 2019

@author: echtpar
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\FFTImages\\Image2\\Slide1.jpg',0)
#img2 = cv2.imread('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\FFTImages\\ZigZag.jpg',0)
#img = img2-img1
#img = img-img


"""
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    out.write(frame)
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    

    
    #plt.subplot(121),plt.imshow(img, cmap = 'gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.show()


#    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#    plt.show()

    
#    magnitude_spectrum.dtype = "uint8"
#    #fshift.dtype = "uint8"
    cv2.imshow('gray', magnitude_spectrum/255)

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    cv2.imshow('inverse', img_back/255)

    
    #maskimg.astype('uint8') * 255
    k = ord('q')    
    if cv2.waitKey(1) & 0xFF == k:
        break


#print(cv2.waitKey(1))
#print(0xFF)
#print(cv2.waitKey(1)&0xFF==k)


print (k)
    
cap.release()
#out.release()
cv2.destroyAllWindows()

print(fshift[:20, :20])
print(f_ishift[:20, :20])


"""
"""
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
print(magnitude_spectrum[:10])
print("next")
print(magnitude_spectrum[:10]/5000)
#magnitude_spectrum = fshift
#print(type(magnitude_spectrum))
#print(magnitude_spectrum[:20, :20])

fig = plt.figure()

#cv2.imshow('magnitude_spectrum', magnitude_spectrum/255)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'hot')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig = plt.figure()

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()

"""


import glob
jpgFiles = glob.glob("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\FFTImages\\Image2\\Slide1*.jpg")
NoOfFiles = len(jpgFiles)
print(jpgFiles[0])

fig, axes = plt.subplots(NoOfFiles,2, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(160,160))
fftImageList = {}
#fig.subplots_adjust(hspace = 0, wspace = 0)

plt.subplots_adjust(wspace=0, hspace=0)
z = axes.flat
nextR = 0
images = {}
fshiftList = {}
print(axes.shape)
for i in range(NoOfFiles):
    img2 = cv2.imread(jpgFiles[i],0)
    img = (img2-img1)/255
    #img = img2
    images[i] = img
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshiftList[i] = fshift
    magnitude_spectrum = np.log(np.abs(fshift))
    #magnitude_spectrum = np.log(np.abs(fshift))
#    print(magnitude_spectrum[:10])
#    print("next")
#    print(magnitude_spectrum[:10]/5000)

    axes.flat[i*2+0].imshow(img, cmap = 'gray')
    axes.flat[i*2+1].imshow(magnitude_spectrum, cmap = 'hot')
    fftImageList[i] = magnitude_spectrum
#    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'hot')
    axes.flat[i*2+0].set_yticklabels([])
    axes.flat[i*2+0].set_xticklabels([])
    axes.flat[i*2+1].set_yticklabels([])
    axes.flat[i*2+1].set_xticklabels([])
#    axes.flat[nextR+0].axis('scaled')
#    axes.flat[nextR+1].axis('scaled')
    axes.flat[i*2+0].autoscale(enable=False) 
    axes.flat[i*2+1].autoscale(enable=False) 
#    axes[nextR+0].set_xticklabels([])
#    axes[nextR+0].set_yticklabels([])
#    axes[nextR+1].set_xticklabels([])
#    axes[nextR+1].set_yticklabels([])
plt.tight_layout()
plt.show()

print(len(fftImageList))

"""
fig, axes = plt.subplots(NoOfFiles,1, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(160,160))
plt.subplots_adjust(wspace=0, hspace=0)
z = axes.flat

for i in range(NoOfFiles):
    axes.flat[i].imshow(fftImageList[i], cmap = 'hot')
    axes.flat[i].set_yticklabels([])
    axes.flat[i].set_xticklabels([])
    axes.flat[i].autoscale(enable=False) 
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(NoOfFiles,1, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(160,160))
plt.subplots_adjust(wspace=0, hspace=0)
z = axes.flat
j=0
for i in range(NoOfFiles):
    if j == 0:
        j = i
    else:
        j=i-1
    axes.flat[i].imshow(fftImageList[i]- fftImageList[j], cmap = 'hot')
    axes.flat[i].set_yticklabels([])
    axes.flat[i].set_xticklabels([])
    axes.flat[i].autoscale(enable=False) 
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(4,2, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(160,160))
plt.subplots_adjust(wspace=0, hspace=0)
z = axes.flat
NoOfImages = len(images)
plt.subplots_adjust(wspace=0, hspace=0)
#DiffFft = fftImageList[10]-fftImageList[2]
#axes.flat[0].imshow(DiffFft, cmap = 'hot')


axes.flat[0].imshow(images[NoOfImages-7])
axes.flat[1].imshow(images[NoOfImages-6])

axes.flat[2].imshow(fftImageList[NoOfImages-7], cmap = 'hot')
axes.flat[3].imshow(fftImageList[NoOfImages-6], cmap = 'hot')



#f_ishift = np.fft.ifftshift(DiffFft)
#diffImg = np.fft.ifft2(f_ishift)
#diffImg = np.abs(diffImg)
#axes.flat[1].imshow(diffImg, cmap = 'gray')

diffImg = np.fft.ifft2(fshiftList[NoOfImages-7] - fshiftList[NoOfImages-6])
diffImg = np.abs(diffImg)
axes.flat[4].imshow(diffImg, cmap = 'gray')

axes.flat[5].imshow(fftImageList[NoOfImages-7] - fftImageList[NoOfImages-6], cmap = 'hot')


#print("shape of fshiftList[9] =", fshiftList[9].shape)
#print(fshiftList[9][:2,:2])
#axes.flat[6].imshow(fshiftList[9], cmap = 'hot')
#axes.flat[7].imshow(fshiftList[2], cmap = 'hot')



#f_ishift = fftImageList[2]
#diffImg = np.fft.ifft2(f_ishift)
#diffImg = np.abs(diffImg)
#axes.flat[3].imshow(diffImg, cmap = 'gray')


#rows, cols = images[10].shape
#crow,ccol = rows/2 , cols/2
#fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
#f_ishift = np.fft.ifftshift(fshiftList[2])
#img_back = np.fft.ifft2(f_ishift)
#img_back = np.abs(img_back)
#axes.flat[3].imshow(img_back, cmap = 'gray')

plt.show()

"""