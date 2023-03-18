import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import cv2 as cv2
def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs
    I2 = rgb2gray(I2)
    descriptor_extractor = SIFT()

    # extract descriptor for first image
    descriptor_extractor.detect_and_extract(I1)
    locs1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    # extract descriptor for second image
    descriptor_extractor.detect_and_extract(I2)
    locs2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # match the descriptors from both images
    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,cross_check=True)

    return matches, locs1, locs2

def getHomography(pts1,pts2):
    size = pts1.shape[0]
    A = np.zeros((size*2, 9))
    for j in range(size):
        A[j*2,:] = np.array([-pts1[j,1], -pts1[j,0], -1, 0, 0, 0, pts1[j,1]*pts2[j,1], pts1[j,0]*pts2[j,1], pts2[j,1]])
        A[j*2+1,:] = np.array([0, 0, 0, -pts1[j,1], -pts1[j,0], -1, pts1[j,1]*pts2[j,0], pts1[j,0]*pts2[j,0], pts2[j,0]])
    _, _, V = np.linalg.svd(A)
    return V[-1,:].reshape((3, 3))

def computeH_ransac(matches, locs1, locs2):
    # Compute the best fitting homography using RANSAC given a list of matching pairs
    max_iter = 10000
    threshold = 50
    min_inliers = 4

    bestH = None
    inliers = []
    for i in range(max_iter):
        sample = np.random.choice(len(matches), 4, replace=False)
        pts1 = locs1[matches[sample, 0]]
        pts2 = locs2[matches[sample, 1]]

        #Defining Homography matrix
        H = getHomography(pts1,pts2)

        # Compute projection of locs1
        pts1_proj = np.dot(H, np.vstack((locs1.T, np.ones(locs1.shape[0]))))
        pts1_proj = pts1_proj[:2, :] / pts1_proj[2, :]
        pts1_proj = pts1_proj.T

        # Compute distance between projected points and locs2
        dist = np.empty(locs2.shape[0])
        dist.fill(np.inf)
        for j in range(len(matches)):
            if matches[j, 1] < locs2.shape[0]:
                d = np.sqrt(np.sum((pts1_proj[matches[j, 0]] - locs2[matches[j, 1]])**2))
                if d < threshold:
                    dist[matches[j, 1]] = min(d, dist[matches[j, 1]])
        # Count inliers
        current_inliers = np.where(dist < threshold)[0]
        if len(current_inliers) > min_inliers:
            #Re-compute least-squares H estimate on all of the inliers
            pts1_inliers = locs1[matches[matches[:, 1] == current_inliers[0], 0], :]
            for k in range(1, len(current_inliers)):
                pts1_inliers = np.vstack((pts1_inliers, locs1[matches[matches[:, 1] == current_inliers[k], 0], :]))
            pts2_inliers = locs2[current_inliers]
            # Compute homography
            H_ = getHomography(pts1_inliers,pts2_inliers)
            if len(current_inliers) > len(inliers):
                bestH = H_
                inliers = current_inliers
    indexes =[]
    for k in range(len(inliers)):
        indexes.append(np.where(matches[:,1]==inliers[k]))
    indexes = np.asarray(indexes).reshape(len(inliers))
    return bestH, indexes

def compositeH(H, template, img):

    # Create a compositie image after warping the template image on top
    # of the image using homography


    # Get the height and width of the template image
    h, w = template.shape[:2]

    # Create a mask of the same size as the template image
    mask = np.ones((h, w), np.uint8) * 255

    # Warp the mask by the homography matrix
    warped_mask = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))

    # Warp the template image by the homography matrix
    warped_template = cv2.warpPerspective(template, H, (img.shape[1], img.shape[0]))

    # Create a 3-channel mask from the warped mask
    mask_3channel = np.stack([warped_mask] * 3, axis=2)

    # Use the warped mask to combine the warped template and the destination image
    composite_img = np.where(mask_3channel == 255, warped_template, img)

    return composite_img
