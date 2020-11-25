import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
import random

def gaussian(x, stddev):
    return 1 / np.sqrt(2 * np.pi) * stddev * np.exp(-(x ** 2) / (2 * stddev ** 2)) # gaussian function

def compute_kernel(sigma):
    dim = 2 * int(2 * sigma + 0.5) + 1 # determines the size of the kernel using sigma --> ensures the size of the kernel is odd
    k_gaussian = np.linspace(-(dim//2), dim//2, dim) # creates the array that will be used to create the outer product
    for i in range(len(k_gaussian)):
        k_gaussian[i] = gaussian(k_gaussian[i], sigma) # 1D gaussian function
    k_gaussian = np.outer(k_gaussian.T, k_gaussian.T) # creates a 2D gaussian function by taking the outer product of the two 1D gaussian functions
    k_gaussian = k_gaussian / k_gaussian.max() # normalizes the kernel to ensure the maximum value is 1
    return k_gaussian

def convolve(im, kern):
    im_h = im.shape[0]
    im_w = im.shape[1]
    kern_h = kern.shape[0]
    kern_w = kern.shape[1]
    output = np.zeros((im_h, im_w))

    kern_size = kern_h * kern_w

    add = [int((kern_h-1)/2), int((kern_w-1)/2)] # size of the additional padded height and weight when filter is placed at edges of the image
    new_im = np.zeros((im_h + 2 * add[0], im_w + 2 * add[1])) # dimensions of the padded image
    new_im[add[0] : im_h + add[0], add[1] : im_w + add[1]] = im # sets the non-padded pixels of the padded image to the pixels of the image being convolved

    for i in range(im_h):
        for j in range(im_w):
            result = kern * new_im[i : i + kern_h, j : j + kern_w] # elementwise multiplication of kernel and appropriate pixels
            output[i, j] = np.sum(result) # adds the elements of the elementwise multiplication
    output = output / kern_size # reduction of pixel values based on kernel size

    return output

def gradient(gx, gy):
    return np.sqrt(gx**2 + gy**2) # gradient magnitude of image

def time_filter(sigma):
    for dirpath, dirnames, files in os.walk('./Q3_optical_flow/'): # determines all the folders with images
        if dirpath == './Q3_optical_flow/':
            continue
        files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))] # obtains all the files located within a specific folder
        try:
            files.remove('.DS_Store')
        except:
            print("No DS Store file")
        print("Applying 1-D time-filter to %s folder" %(dirpath))
        files = sorted(files) # makes sure the files are in order of the frame time
        image = Image.open(dirpath + '/' + files[0])
        image = image.convert(mode='L')
        data = np.asarray(image)
        frames = np.zeros((8, data.shape[0], data.shape[1])) # frames array which will hold all 8 frames for a specific folder
        
        for i in range(len(files)):
            image = Image.open(dirpath + '/' + files[i])
            image = image.convert(mode='L')
            data = np.asarray(image)
            frames[i,:,:] = data # sets the each layer to be one respective frame
        
        for i in range(frames.shape[1]):
            for j in range(frames.shape[2]):
                frames[:,i,j] = gaussian_filter1d(frames[:,i,j], sigma) # applies a 1d gaussian filter along the time axis

        for i in range(frames.shape[0]):
            data = frames[i,:,:]
            image = Image.fromarray(data)
            if i < 3:
                image.convert('L').save('%s/convert_frame0%s.png' %(dirpath,str(i+7)))
            else:
                image.convert('L').save('%s/convert_frame%s.png' %(dirpath,str(i+7))) # saves each new image in the respective folder

    return True

def iteration(data_0, data_1, grad_0, grad_1, x_val, y_val, w, threshold):
    I_n = data_0[x_val-w:x_val+w, y_val-w:y_val+w] # I_n(x,y)
    I_n1 = data_1[x_val-w:x_val+w, y_val-w:y_val+w] # I_n+1(x,y)
    I_x = grad_0[x_val-w:x_val+w, y_val-w:y_val+w] # partial with respect to x
    I_y = grad_1[x_val-w:x_val+w, y_val-w:y_val+w] # partial with respect to y
    A = np.array([[np.sum(I_x*I_x), np.sum(I_x*I_y)], [np.sum(I_x*I_y), np.sum(I_y*I_y)]]) # creates the A matrix
    b = np.array([[-np.sum((I_n-I_n1)*I_x)], [-np.sum((I_n-I_n1)*I_y)]]) # creates the b vector
    A_tA = np.matmul(A.T, A) # determines A transpose times A
    try:
        A_tA_inv = np.linalg.inv(A_tA) # checks to see if A transpose times A is invertible
    except:
        return np.array([[0],[0]])
    eig1, eig2 = np.linalg.eigvals(A_tA) # determines the eigenvalues of A transpose times A
    if eig1 == 0 or eig2 == 0 or eig1/eig2 > threshold or eig1/eig2 < 1/threshold: # ensures the eigenvalues are well-conditioned according to some threshold
        return np.array([[0],[0]])
    d = np.matmul(A_tA_inv, np.matmul(A.T, b)) # determines the value of d using Lucas-Kanade equation
    return d

def main(sigma, window_size, threshold):
    for dirpath, dirnames, files in os.walk('./Q3_optical_flow/'): # determines all the folders with images
        if dirpath == './Q3_optical_flow/':
            continue
        files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))] # obtains all the files located within a specific folder
        try:
            files.remove('.DS_Store')
        except:
            print("No DS Store file")
        files = sorted(files)[:8] # makes sure the files are in order of the frame time
        im = random.randrange(0,6,1) # randomizes the picture chosen in the folder
        im2 = random.randrange(im+1,7,1)
        files = [files[im], files[im2]]
        for i in range(len(files) - 1):
            print("Examining flow between %s and %s in the directory %s" %(files[i], files[i+1], dirpath))
            image_0 = Image.open(dirpath + '/' + files[i])
            image_1 = Image.open(dirpath + '/' + files[i+1]) # opens the current frame and the next frame
            image_0 = image_0.convert(mode='L')
            image_1 = image_1.convert(mode='L')
            data_0 = np.asarray(image_0)
            data_1 = np.asarray(image_1) # converts these images to numpy arrays

            print("-----computing and applying gaussian filter")
            gaussian = compute_kernel(sigma)
            data_0 = convolve(data_0, gaussian)
            data_1 = convolve(data_1, gaussian) # applies gaussian filter to both images

            gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel x
            gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # sobel y

            print("-----computing gradients with respect to each pixel")
            grad_0 = gradient(convolve(data_0, gx), convolve(data_0, gy))
            grad_1 = gradient(convolve(data_1, gx), convolve(data_1, gy)) # applies the sobel filters to determine gradients

            w = window_size // 2 # determines the pixels of interest around a given pixel according to the window size

            print("-----determining flow for each pixel")
            u, v, x, y = [], [], [], [] # u stores x displacement, v stores y displacement, x stores the x coordinates, y stores the y coordinates
            for j in range(w, grad_0.shape[0]-w+1):
                for k in range(w, grad_0.shape[1]-w+1):
                    d = iteration(data_0, data_1, grad_0, grad_1, k, j, w, threshold) # gets the displacement values
                    u_disp = 0
                    v_disp = 0 # temporary values of displacement for each iteration
                    x_val, y_val = k, j # temporary x and y for new location based on iteration
                    while abs(d[0,0]) > 3 or abs(d[1,0]) > 3: # iterates until the displacements are below 3
                        x_val = int(k + u_disp)
                        y_val = int(j + v_disp) # updates the new values of x and y
                        d = iteration(data_0, data_1, grad_0, grad_1, x_val, y_val, w, threshold) # determines the displacement
                        u_disp = d[0,0]
                        v_disp = d[1,0]
                        check = np.array([[0],[0]])
                        if np.array_equal(d, check):
                            break # if displacement has no magnitude, move out of loop
                    check = np.array([[0],[0]])
                    if not np.array_equal(d, check):
                        print(d)
                        u.append(d[0,0])
                        v.append(d[1,0]) # adds components of the displacement
                        x.append(k)
                        y.append(j) # adds the pixels location
            
            v = [-x for x in v] # flips the y displacement since images are indexed from top to bottom
            fig1, ax1 = plt.subplots()
            ax1.quiver(x,y,u,v,scale=200) # creates the quiver plot
            ax1.imshow(image_0) # overlays the image
            fig1.savefig('%s/%s-%s.png' %(dirpath, files[i][:-4], files[i+1][:-4])) # saves the image in the directory
            plt.close(fig1)
            print('Success! %s-%s.png has been saved successfully in the directory %s' %(files[i][:-4], files[i+1][:-4], dirpath))

if __name__ == "__main__":
    # time_filter(sigma = 0.1) this function only needs to be run once at the beginning to obtain the new files
    main(sigma=1, window_size=5, threshold=5)