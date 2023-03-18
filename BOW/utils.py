from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist
import random
def computeHistogram(img_file, F, textons):
    img = img_as_float(rgb2gray(io.imread(img_file)))
    num_filters = F.shape[2]
    filtered_img = np.zeros((img.shape[0], img.shape[1], num_filters))
    for j in range(num_filters):
        filtered_img[:,:,j] = correlate(img, F[:,:,j])
    filtered_img = np.reshape(filtered_img, (-1, num_filters))
    # calculate the L2 distance between the 48-dimensional vector representation of this pixel and the cluster center
    dists = cdist(filtered_img, textons, 'euclidean')
    assignments = np.argmin(dists, axis=1)
    histogram, _ = np.histogram(assignments, bins=range(textons.shape[0]+1))
    return histogram

def createTextons(F, file_list, K):
    filtered_images = []
    num_filters = F.shape[2]
    for name in file_list:
        img = img_as_float(rgb2gray(io.imread(name)))
        filtered_img = np.zeros((img.shape[0], img.shape[1], num_filters))
        # calculate the response of each filter
        for j in range(num_filters) :
            filtered_img[:,:,j] = correlate(img,F[:,:,j])
        filtered_img = np.reshape(filtered_img, (-1, num_filters))
        # Random sample of 1000 pixels per image
        rows_id = random.sample(range(0,filtered_img.shape[0]-1), 1000)
        filtered_images.append(filtered_img[rows_id,:])
    filtered_images = np.concatenate(filtered_images, axis=0)
    # Run k-mean cluster algorithm
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_init=1).fit(filtered_images)
    textons = kmeans.cluster_centers_
    return textons