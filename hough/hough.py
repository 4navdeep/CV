# import other necessary libaries
from utils import create_line, create_mask
from skimage import feature
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import math

# load the input image
img = imread('road.jpg', as_gray=True)
h,w = img.shape
# run Canny edge detector to find edge points
c_edge = feature.canny(img)
# create a mask for ROI by calling create_mask
mask = create_mask(h,w)
# extract edge points in ROI by multipling edge map with the mask
edges = c_edge*mask
# perform Hough transform

#max length of any edge equals to diagonal
diagonal = int(math.sqrt((h * h) + (w * w)))
#range of angles
thetas = np.deg2rad(np.arange(-80.0, 80.0))
rhos = np.linspace(-diagonal, diagonal, diagonal * 2)
num_thetas = len(thetas)
num_rhos = len(rhos)
# Cache some resuable values
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
num_thetas = len(thetas)
# hough accumulator vote array
accumulator = np.zeros((2 * diagonal, num_thetas), dtype=np.uint64)
y_idxs, x_idxs = np.nonzero(edges)  # (row, col) indexes to edges

for i in range(len(x_idxs)):
    # For each edge point
    x = x_idxs[i]
    y = y_idxs[i]
    for t_idx in range(num_thetas):
      # Calculate rho. diagonal is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diagonal
      accumulator[rho, t_idx] += 1

# find the right lane by finding the peak in hough space
# Easiest peak finding based on max votes
idx = np.argmax(accumulator)
r1 = rhos[round(idx / accumulator.shape[1])]
t1 = thetas[(idx % accumulator.shape[1])]
xs1,ys1 = create_line(r1,t1,img)

accumulator_ = np.copy(accumulator)
peak_row = round(idx / accumulator.shape[1])
peak_col = (idx % accumulator.shape[1])
# zero out the values in accumulator around the neighborhood of the peak
neighborhood_radius = 25
# Define the indices of the neighborhood around the peak
row_min = max(0, peak_row - neighborhood_radius)
row_max = min(accumulator_.shape[0], peak_row + neighborhood_radius)
col_min = max(0, peak_col - neighborhood_radius)
col_max = min(accumulator_.shape[1], peak_col + neighborhood_radius)

# Set the values in the neighborhood to zero
accumulator_[row_min:row_max, col_min:col_max] = 0
# find the left lane by finding the peak in hough space
idx2 = np.argmax(accumulator_)
r2 = rhos[round(idx2 / accumulator_.shape[1])]
t2 = thetas[(idx2 % accumulator_.shape[1])]
xs2,ys2 = create_line(r2,t2,img)
# plot the results
plt.imshow(c_edge,cmap='gray')
plt.show()
plt.imshow(mask,cmap='gray')
plt.show()
plt.imshow(edges,cmap='gray')
plt.show()
plt.imshow(img,cmap='gray')
plt.plot(xs1,ys1,color='blue')
plt.plot(xs2,ys2,color='orange')
plt.show()