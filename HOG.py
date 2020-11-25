import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

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

def main(T, sigma):
    dirpath = './Q4'
    files = [f for f in listdir(dirpath) if isfile(join(dirpath,f))] # iterates through the files in the folder
    try:
        files.remove('.DS_Store')
    except:
        print("No DS Store file") # checks to make sure there is no .DS_Store file
    files = sorted(files)
    for file in files:
        image = Image.open(dirpath + '/' + file)
        image = image.convert(mode='L')
        data = np.asarray(image) # converts the image to a numpy array

        gaussian = compute_kernel(sigma)
        data = convolve(data, gaussian) # convolves a gaussian filter with the image

        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel x
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # sobel y

        grad_x = convolve(data, gx)
        grad_y = convolve(data, gy) # calculates the x and y gradients of the image
        grad_mag = gradient(grad_x, grad_y) # determines the gradient magnitude of each pixel
        grad_mag = np.where(grad_mag > 0.01, grad_mag, 0) # thresholds the lower pixel values
        grad_dir = np.degrees(np.arctan(grad_y/grad_x)) # determines the direction of the gradient at each pixel
        grad_dir = grad_dir + 75 # directions are in a range of -90 to 90, add 75 to get a range of -15 to 175
        
        cut_x = data.shape[1] % T
        cut_y = data.shape[0] % T
        data = data[cut_y//2:data.shape[0]-(cut_y//2), cut_x//2:data.shape[1]-(cut_x//2)] # crops the image to make sure the cell size fits the image

        m = data.shape[0] // T
        n = data.shape[1] // T # determines the values of m and n
        
        hist = np.zeros((m,n,6)) # creates the histogram array
        for i in range(m):
            for j in range(n):
                dir = grad_dir[i*T:(i+1)*T, j*T:(j+1)*T]
                dir = dir.flatten() # gets the directions for the 8x8 cell
                bins = [-15, 15, 45, 75, 105, 135, 165]
                for k in range(dir.shape[0]):
                    for l in range(len(bins)-1):
                        if bins[l] <= dir[k] < bins[l+1]:
                            hist[i,j,l] += 1 # places the respective bin

        fig1, ax1 = plt.subplots() # creates the quiver plot
        for i in range(hist.shape[2]):
            x, y = [], [] # arrays to hold the x and y coordinates of the quivers
            x_dir = math.cos(math.radians(bins[i])) # determines the x direction of the gradient bin
            y_dir = math.sin(math.radians(bins[i])) # determines the y direction of the gradient bin
            u, v = [], [] # arrays to hold the direction of the quivers
            for j in range(hist.shape[0]):
                for k in range(hist.shape[1]):
                    x.append(k*T+T/2) # x value roughly in the middle of the cell
                    y.append(j*T+T/2) # y value roughly in the middle of the cell
                    u.append(x_dir * hist[j,k,i])
                    v.append(y_dir * hist[j,k,i]) # multiplies the direction of the arrow by the number of occurrences
            v = [-x for x in v] # flips the y direction since images are indexed from top to bottom
            ax1.quiver(x,y,u,v,scale=1000) # overlays all the quiver plots
        ax1.imshow(image) # adds the image to the plot
        plt.show()

        hist_new = np.zeros((m-1,n-1,24)) # array for the normalizes histogram
        for i in range(hist.shape[0]-1):
            for j in range(hist.shape[1]-1):
                block = hist[i:i+2,j:j+2,:] # creates the 2x2 block
                h_2 = np.sum(np.square(block)) + 0.001 # value of the denominator without the square root
                h_hat = block / (h_2 ** 0.5) # determines the normalizes value h
                h_hat = h_hat.flatten()
                hist_new[i,j,:] = h_hat # adds the normalized value h to the normalized histogram
        
        f = open('%s.txt' %(file[0]), 'w+') # reads the values in the normalizes histogram to .txt files
        for i in range(hist_new.shape[0]):
            f.write('Row #%s: \n\n' %(i))
            for j in range(hist_new.shape[1]):
                f.write('Column #%s: ' %(j))
                for k in range(hist_new.shape[2]):
                    f.write(str(hist_new[i,j,k]) + ' ')
                f.write('\n')
            f.write('\n\n')
        f.close()

if __name__ == "__main__":
    main(T=8, sigma=1)