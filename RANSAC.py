import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def filtered(match):
    filtered = []
    for i, j in match:
        if 0.77 * j.distance > i.distance: # adds the reliable matches based on a chosen value of phi
            filtered.append(i)
    return filtered

def source_des(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None) # determines the key points and descriptors for image 1
    kp2, des2 = sift.detectAndCompute(image2,None) # determines the key points and descriptors for image 2

    brute_force = cv2.BFMatcher()
    matches = filtered(brute_force.knnMatch(des1, des2, k=2)) # determines the matches between the two images

    source = []
    descriptors = []
    for match in matches:
        source.append(kp1[match.queryIdx].pt)
        descriptors.append(kp2[match.trainIdx].pt) # adds the sources and descriptors to their respective arrays
    source = np.array(source, dtype=np.float32).reshape(-1,1,2)
    descriptors = np.array(descriptors, dtype=np.float32).reshape(-1,1,2)
    return source, descriptors, matches, kp1, kp2, des1, des2 

def ransac_homo(image1, image2, threshold):
    source, descriptors, matches, kp1, kp2, des1, des2 = source_des(image1, image2)
    M, mask = cv2.findHomography(source, descriptors, cv2.RANSAC, threshold) # mask which specifies inliers and outliers
    # print(M)
    height, width = image1.shape[0], image1.shape[1]

    coordinates = np.float32([[0,0],[0,height-1],[width-1,height-1],[width-1,0]]).reshape(-1,1,2) # edge points of the image of interest
    perspect = cv2.perspectiveTransform(coordinates, M) # changes image perspective based on important points
    im2_poly = cv2.polylines(image2,[np.int32(perspect)],True,255,2,cv2.LINE_AA) # draws the polygon around the image of interest
    plt.imshow(im2_poly)
    plt.show()
    im1_match = cv2.drawMatches(image1,kp1,im2_poly,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=(255,0,0)) # determines the matches between object and image
    plt.imshow(im1_match)
    plt.show()

def ransac_affine(image1, image2, s, threshold, p, p_small, k):
    source, descriptors, matches, kp1, kp2, des1, des2 = source_des(image1, image2)
    result = 0 # will be used to store the updated values of p_small
    model = None # will be used to save the best model
    height = image1.shape[0]
    width = image1.shape[0]
    i = 0
    
    while i < s:
        source_len = len(source)
        idx = np.random.choice(source_len,3,replace=False) # chooses 3 random points to fit line to
        affine = cv2.getAffineTransform(source[idx], descriptors[idx]) # determines the affine transformation between the points
        count = 0
        for j in range(source_len):
            distance = np.linalg.norm(np.dot(affine, np.append(source[j][0], [1])) - descriptors[j][0])
            if distance < threshold: # determines the distance between the points and the affine transformation
                count += 1 # if the point is within the threshold, add it to the count of inliers
        p_small = (source_len/count)**-1 # update the value of p_small
        P = 1 - (1-p_small**k) ** i # update the value of P

        if p_small > result: # update result and model if the current iteration is better than the previous best
            result = p_small
            model = affine
        if P > p: # update number of trials based on the new value of P
            s = np.log(1-P) / np.log(1-p_small**k)
        i += 1

    model = np.concatenate((model, np.array([[0.0,0.0,1.0]]))) 
    # print(model)
    coordinates = np.float32([[0,0],[0,height-1],[width-1,height-1],[width-1,0]]).reshape(-1,1,2) # edge points of image of interest
    perspect = cv2.perspectiveTransform(coordinates, model) # changes image perspective based on important points
    im2_poly = cv2.polylines(image2,[np.int32(perspect)],True,255,2,cv2.LINE_AA) # draws a shape around the object
    plt.imshow(im2_poly)
    plt.show()
    im1_match = cv2.drawMatches(image1,kp1,im2_poly,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=(255,0,0)) # determines the matches between object and image
    plt.imshow(im1_match)
    plt.show()

def main():
    im1 = cv2.imread(os.path.join(os.getcwd(), 'Book_cover.png')) # loads the images
    im2 = cv2.imread(os.path.join(os.getcwd(), 'Book_pic.png'))
    threshold = 5
    p = 0.995
    p_small = 0.5
    k = 3
    s = np.log(1-p) / np.log(1-0.5**k) # sets the respective initial values of the variables

    ransac_homo(im1, im2, threshold) # ransac using homography
    ransac_affine(im1, im2, s, threshold, p, p_small, k) # ransac using affine transformations

if __name__ == "__main__":
    main()