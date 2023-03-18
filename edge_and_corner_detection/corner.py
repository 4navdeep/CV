import numpy as np
from utils import filter2d, partial_x, partial_y, gaussian_kernel
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """
# Compute partial derivatives
    Ix = partial_x(img)
    Iy = partial_y(img)

    # Compute products of derivatives
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    filter = gaussian_kernel(window_size)
    # Compute sums of the products of derivatives
    Sx2 = filter2d(Ix2, filter)
    Sy2 = filter2d(Iy2, filter)
    Sxy = filter2d(Ixy, filter)

    # Compute determinant and trace
    det = (Sx2 * Sy2) - (Sxy**2)
    trace = Sx2 + Sy2

    # Compute Harris response
    response = det - k * (trace**2)

    return response

def main():
    img = imread('building.jpg', as_gray=True)

    # Compute Harris corner response
    response = harris_corners(img)
    plt.imshow(response)
    plt.show()
    # Threshold on response
    threshold = 0.1 * np.max(response)
    response = (response > threshold) * response
    plt.imshow(response)
    plt.show()
    # Perform non-max suppression by finding peak local maximum
    corners = peak_local_max(response, min_distance=5)
    # Visualize results
    plt.scatter(corners[:, 1], corners[:, 0], marker = 'X',s=10, c='red')
    plt.show()
    
if __name__ == "__main__":
    main()
