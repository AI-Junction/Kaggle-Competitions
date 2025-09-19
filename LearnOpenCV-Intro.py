# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 00:39:40 2017

@author: echtpar
"""

### CV2 START


import cv2
import numpy as np


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
#    out.write(frame)
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
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




import numpy as np
import cv2
# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)
print(img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('messigray.png',img)


import numpy as np
import cv2
img = cv2.imread('messi5.jpg',cv2.IMREAD_UNCHANGED)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()

    
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg',cv2.IMREAD_COLOR)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()    



import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('img', img)

img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.imshow('img', img)

img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
cv2.imshow('img', img)

img = cv2.ellipse(img,(256,256),(100,50),0,0,360,(100,200,100),-1)
cv2.imshow('img', img)


pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
cv2.imshow('img', img)


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('img', img)

import cv2
import numpy as np
img = cv2.imread('messi5.jpg')

cv2.imshow('img', img)


px = img[200,210]
print (px)
#[157 166 200]
# accessing only blue pixel
blue = img[100,100,0]
print (blue)
#157

img[100,100] = [255,255,255]
print (img[100,100])
#[255 255 255]


print (img.shape)
print (img.size)
print (img.dtype)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

cv2.imshow('img1', img)





import cv2
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv2.imread('opencv_logo.png')
replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()




# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('img', mask)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow('img', mask_inv)


# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow('img', img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow('img', img2_fg)


# Put logo in ROI and modify the main image
#dst = cv2.add(img1_bg,img2_fg)
dst = cv2.add(img1,img1_bg)

img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)



import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()








import cv2
import numpy as np

imgEllipse = cv2.imread("Ellipse.jpg", cv2.IMREAD_COLOR)
imgRectangle = cv2.imread("Rectangle.jpg", cv2.IMREAD_COLOR)
imgLine = cv2.imread("Line.jpg", cv2.IMREAD_COLOR)

cv2.imshow('img_e', imgEllipse)
cv2.imshow('img_r', imgRectangle)
cv2.imshow('img_l', imgLine)

imgEllipseGray = cv2.cvtColor(imgEllipse, cv2.COLOR_BGR2GRAY)
imgRectangleGray = cv2.cvtColor(imgRectangle, cv2.COLOR_BGR2GRAY)
imgLineGray = cv2.cvtColor(imgLine, cv2.COLOR_BGR2GRAY)

cv2.imshow('img_eg', imgEllipseGray)
cv2.imshow('img_rg', imgRectangleGray)
cv2.imshow('img_lg', imgLineGray)


ret, mask_ell = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img_em', mask_ell)


ret, mask_rect = cv2.threshold(imgRectangleGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img_rm', mask_rect)


mask_ell_not = cv2.bitwise_not(mask_ell)
cv2.imshow('img_em_not', mask_ell_not)


mask_rect_not = cv2.bitwise_not(mask_rect)
cv2.imshow('img_rm_not', mask_rect_not)


mask_and = cv2.bitwise_and(mask_ell, mask_rect)
cv2.imshow('img_and', mask_and)


mask_or = cv2.bitwise_or(mask_ell, mask_rect)
cv2.imshow('img_or', mask_or)


mask_or_of_nots = cv2.bitwise_or(mask_ell_not, mask_rect_not)
cv2.imshow('img_or_of_nots', mask_or_of_nots)

mask_and_of_nots = cv2.bitwise_and(mask_ell_not, mask_rect_not)
cv2.imshow('img_and_of_nots', mask_and_of_nots)


mask_xor_of_nots = cv2.bitwise_xor(mask_ell_not, mask_rect_not)
cv2.imshow('img_xor_of_nots', mask_xor_of_nots)



ell_masked_and = cv2.bitwise_and(imgEllipse, imgEllipse, mask = mask_and_of_nots)
cv2.imshow('ell_masked_and', ell_masked_and)

ell_masked_or = cv2.bitwise_or(imgEllipse, imgEllipse, mask = mask_or_of_nots)
cv2.imshow('ell_masked_or', ell_masked_or)



print(imgEllipseGray < 255)
cv2.imshow('img3', imgEllipseGray)


print(imgEllipseGray[200,200])
print(imgEllipseGray[250,200])



ret, mask1 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('img1', mask1)

ret, mask2 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_MASK)
cv2.imshow('img2', mask2)


ret, mask3 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('img3', mask3)

ret, mask4 = cv2.threshold(imgEllipseGray, 100, 255, cv2.THRESH_TRIANGLE)
cv2.imshow('img4', mask4)



#cv2.pixel(imgEllipseGray, [10,10])

mask_inv = cv2.bitwise_not(imgEllipse)
cv2.imshow('img5', mask_inv)

mask_inv = cv2.bitwise_and(mask1, imgEllipse)
cv2.imshow('img5_1', mask_inv)





img1_and = cv2.bitwise_and(imgEllipse,imgRectangle, mask = mask_inv)
cv2.imshow('img6', img1_and)

img1_or = cv2.bitwise_or(imgEllipse,imgRectangle)
cv2.imshow('img7', img1_or)



import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('circle.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.png')

print(img1.shape)
print(img2.shape)

img1 = img1[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]]
img2 = img2[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]]

print(img1.shape)
print(img2.shape)

dst = cv2.addWeighted(img1,0.2,img2,0.7,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()








import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bw.png',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()





import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('noisy6.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)


# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(th2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian Blurred Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()







import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bimodal_hsv_noise.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()






img = cv2.imread('noisy2.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)
# find normalized_histogram, and its cumulative distribution function


hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
print(hist_norm.shape)
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1
for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print (thresh,ret)
cv2.imshow('otsu',otsu)



import cv2
import numpy as np
img = cv2.imread('messi5.jpg')
res1 = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('res1', res1)

#OR below gives same resize effect as above

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
cv2.imshow('res', res)


img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('dst', dst)



import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('opencv_logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()



import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('opencv_logo.png')
blur = cv2.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()



blur = cv2.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

import cv2
import numpy as np
img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2)
cv2.imshow('erosion', erosion)

dilation = cv2.dilate(img,kernel,iterations = 2)
cv2.imshow('dilation', dilation)


img = cv2.imread('j-noise.png',0)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)

img = cv2.imread('j-noise-internal.png',0)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)






import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('dave.jpg',0)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


import numpy as np
import cv2
im = cv2.imread('messi5.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('contour image', image)
cv2.imshow('contour image2', img)



img = cv2.imread('messi5.jpg',0)
hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
hist2 = np.bincount(img.ravel(),minlength=256)
print(hist1.shape)
print(hist2.shape)


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()
print(img.ravel().shape)
print(img.size)



import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()




img = cv2.imread('messi5.jpg',0)
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()










### CV2 END








    