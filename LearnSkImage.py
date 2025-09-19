# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:52:23 2018

"""

### SKIMAGE START


#%%
from skimage import data, io, filters
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_adaptive
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label

image = data.coins()
edges = filters.sobel(image)
io.imshow(edges)
io.imshow(image)

#%%
# Load part of the image.
image = data.coins()[0:100, 50:400]
fig, axes = plt.subplots(ncols=3, nrows=3,
figsize=(8, 4))
ax = axes.flat
ax[0].imshow(image , cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

#%%

# Histogram.
values, bins = np.histogram(image ,bins=np.arange(256))
#values, bins = np.histogram(image ,bins=50)
print(values.shape)
print(bins.shape)
print(bins[:-1])
ax[1].plot(bins[:-1], values , lw=2, c='k')
ax[1].set_xlim(xmax=256)
ax[1].set_yticks([0, 400])
ax[1].set_aspect(.2)
ax[1].set_title('Histogram')

#%%

# Apply threshold
bw = threshold_adaptive(image, 95, offset=-15)
ax[2].imshow(bw, cmap=plt.cm.gray)
ax[2].set_title('Adaptive threshold')
ax[2].axis('off')
#%%

edges = canny(image/255.)
ax[3].imshow(edges)

#%%
# Find maxima.
coordinates = peak_local_max(image , min_distance=20)
print(coordinates)
print(coordinates.shape)
ax[4].imshow(image , cmap=plt.cm.gray)
ax[4].autoscale(False)
#ax[3].plot(coordinates[:, 1], coordinates[:, 0], c='r.')
ax[4].scatter(coordinates[:, 1], coordinates[:, 0], c='r')
ax[4].set_title('Peak local maxima')
ax[4].axis('off')

#%%
# Detect edges
edges = canny(image, sigma=3, 
                     low_threshold=10, high_threshold=80)
ax[5].imshow(edges , cmap=plt.cm.gray)
ax[5].set_title('Edges')
ax[5].axis('off')

#%%
# Label image regions
#label_image = label(edges)
label_image = label(bw)
print(label_image.shape)
ax[5].imshow(label_image , cmap=plt.cm.gray)
ax[5].set_title('Labeled items')
ax[5].axis('off')

#%%

print(type(label_image))

for region in regionprops(label_image):
    # Draw rectangle around segmented coins
    print(region.bbox)
    minr , minc , maxr , maxc = region.bbox
    rect = mpatches.Rectangle((minc , minr), 
                              maxc - minc, 
                              maxr - minr, 
                              fill=False, 
                              edgecolor='red', 
                              linewidth=2)
    ax[5].add_patch(rect)
plt.tight_layout()
plt.show()


#%%
label_image = label(bw)
print([x.shape for x in label_image])
ax[6].imshow(bw , cmap=plt.cm.gray)
ax[6].set_title('Labeled items')
ax[6].axis('off')

print(type(label_image))
print(label_image)

rprops = regionprops(label_image)
print(len(rprops))
print([x.area for x in rprops])
print([x.perimeter for x in rprops])

coin_x_pix = [roi['centroid'][1] for roi in rprops if roi['area'] > 20]
coin_y_pix = [roi['centroid'][0] for roi in rprops if roi['area'] > 20]
ax[6].scatter(coin_x_pix, coin_y_pix, c='r')

#%%

#for region in regionprops(label_image):
## Draw r e c t a n g l e around s egment ed c o i n s .
#    if region.area > 10:
#        print(region.bbox)
#        minr , minc , maxr , maxc = region.bbox
##       rect = mpatches.Rectangle((minc , minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
#        ax6.scatter(minc, minr, c='r')


binary_img_with_labels = label2rgb(label_image,
                            image=bw,
                            alpha=0.7,
                            bg_label=0,
                            bg_color=(0, 0, 0),
                            image_alpha=1,
                            kind='overlay'
                            )
ax[7].imshow(binary_img_with_labels, interpolation='Nearest', origin='upper')
ax[7].set_title('Label matrix')
ax[7].axis('off')

plt.tight_layout()
plt.show()

#%%

ic = io.ImageCollection('C:\\Users\\Public\\Pictures\\Sample Pictures\\*')

image0 = rgb2gray(ic[0][:, 500:500+1987, :])
image1 = rgb2gray(ic[1][:, 500:500+1987, :])
image0 = transform.rescale(image0 , 0.25)
image1 = transform.rescale(image1 , 0.25)


'''
fig, ax = plt.subplots(2,2, figsize=(8,8))
axes = ax.flat
axes[0].imshow(image0)
axes[1].imshow(image1)
'''


orb = ORB(n_keypoints=1000, fast_threshold =0.05)
orb.detect_and_extract(image0)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors
orb.detect_and_extract(image1)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors
matches12 = match_descriptors(descriptors1 ,descriptors2 ,cross_check=True)

from skimage.measure import ransac
# Select key points from the source (image to be
# registered) and target (reference image ) .
src = keypoints2[matches12[:, 1]][:, ::-1]
dst = keypoints1[matches12[:, 0]][:, ::-1]
model_robust , inliers = ransac((src, dst), ProjectiveTransform ,
                                min_samples=4, residual_threshold=2)



r, c = image1.shape[:2]
# Note t h a t t r a n s f o rma t i o n s t a k e c o o r d i n a t e s i n
# ( x , y ) format , n o t ( row , column ) , i n o r d e r t o be
# c o n s i s t e n t wi t h mos t l i t e r a t u r e .
corners = np.array([[0, 0],[0, r],[c, 0],[c, r]])
# Warp t h e image c o r n e r s t o t h e i r new p o s i t i o n s .
warped_corners = model_robust(corners)
# Find t h e e x t e n t s o f both t h e r e f e r e n c e image and
# t h e warped t a r g e t image .
all_corners = np.vstack((warped_corners , corners))
corner_min = np.min(all_corners , axis=0)
corner_max = np.max(all_corners , axis=0)
output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])


from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
offset = SimilarityTransform(translation=-corner_min)
image0_ = warp(image0 , offset.inverse , output_shape=output_shape , cval=-1)
image1_ = warp(image1 , (model_robust + offset).inverse , output_shape=output_shape, cval=-1)



def add_alpha(image , background=-1):
    # Add an alpha l a y e r t o t h e image .
    #The alpha l a y e r i s s e t t o 1 f o r f o r e g r o u n d
    #and 0 f o r bac kground .
    
    rgb = gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))

image0_alpha = add_alpha(image0_)
image1_alpha = add_alpha(image1_)
merged = (image0_alpha + image1_alpha)
alpha = merged[..., 3]





### SKIMAGE END