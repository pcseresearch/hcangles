import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
 
grayFile = 'gray.png'
tophatFile = 'tophat.png'
gradXFile = 'gradX.png'
threshFile = 'thresh.png'
contoursFile = 'contours.png'

gray = imread(grayFile)
tophat = imread(tophatFile)
gradX = imread(gradXFile)
thresh = imread(threshFile)
contours = imread(contoursFile)

elevation = 75
azimuth = 135

#mat = mat[:,:,0] # get the first channel
rows, cols = gray.shape
xv, yv = np.meshgrid(range(cols), range(rows)[::-1])

fig1 = plt.figure(figsize=(6,6))
 
ax = fig1.add_subplot(221)
ax.imshow(gray, cmap='gray')
 
ax = fig1.add_subplot(222, projection='3d')
ax.elev= elevation
ax.azim= azimuth
ax.plot_surface(xv, yv, gray)
 
ax = fig1.add_subplot(223)
ax.imshow(tophat, cmap='gray')
 
ax = fig1.add_subplot(224, projection='3d')
ax.elev= elevation
ax.azim= azimuth
ax.plot_surface(xv, yv, tophat)

fig2 = plt.figure(figsize=(6,6))
 
ax2 = fig2.add_subplot(221)
ax2.imshow(gradX, cmap='gray')
 
ax2 = fig2.add_subplot(222, projection='3d')
ax2.elev= elevation
ax2.azim= azimuth
ax2.plot_surface(xv, yv, gradX)
 
ax2 = fig2.add_subplot(223)
ax2.imshow(thresh, cmap='gray')
 
ax2 = fig2.add_subplot(224, projection='3d')
ax2.elev= elevation
ax2.azim= azimuth
ax2.plot_surface(xv, yv, thresh)

fig3 = plt.figure(figsize=(6,3))
 
ax3 = fig3.add_subplot(121)
ax3.imshow(contours, cmap='gray')
 
ax3 = fig3.add_subplot(122, projection='3d')
ax3.elev= elevation
ax3.azim= azimuth
ax3.plot_surface(xv, yv, contours)
 
plt.show()
