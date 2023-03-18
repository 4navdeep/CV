import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)
    # Smooth image with Gaussian kernel
    kernel = gaussian_kernel()
    img = filter2d(img,kernel)
    # Compute x and y derivate on smoothed image
    x_derivate = partial_x(img)
    y_derivate = partial_y(img)
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(np.square(x_derivate) + np.square(y_derivate))
    # Visualize results
    plt.imshow(x_derivate)
    plt.show()
    plt.imshow(y_derivate)
    plt.show()
    plt.imshow(grad_magnitude)
    plt.show()
    
if __name__ == "__main__":
    main()

