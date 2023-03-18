import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d

def main():
    
    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        #subsample image 
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i+1)
        plt.imshow(im_subsample)
        plt.axis('off')

    # subsampling without aliasing, visualize results on 2nd row
    im_subsample = im
    for i in range(N_levels):
        # low-pass filter each color channel of the image before subsampling
        im_filtered = np.zeros_like(im_subsample)
        for c in range(im_subsample.shape[-1]):
            im_filtered[..., c] = filter2d(im_subsample[..., c], gaussian_kernel(5))
        # subsample the filtered image
        im_subsample = im_filtered[::2, ::2, :]
        plt.subplot(2, N_levels, N_levels+i+1)
        plt.imshow(im_subsample)
        plt.axis('off')

    plt.show()
    
if __name__ == "__main__":
    main()
