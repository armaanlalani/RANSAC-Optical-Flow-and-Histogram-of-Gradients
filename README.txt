Q1: LoG.py: code that determines the value of sigma that maximizes the magnitude of the response for a square
- simply run the code (no parameters or alterations needed)


Q2: RANSAC.py: code that runs both homography and affine RANSAC algorithms on an image
- simply run the code (no parameters or alterations needed)
- the first two images will be for homography and the next two images will be for affine transformations


Q3: OpticalFlow.py: determines the displacement between pixels of successive frames

NOTE: ensure the working directory contains the original image folder Q3_optical_flow which then contains the different folders of images

- first begin by commenting out line 172 which needs to be run once in the beginning on the original images in order to apply the 1D time filter across the frames within a folder (every successive run of the code does not require line 172)
- before running the code over the various folders, ensure the only files in the folders are the original images and the convert_frame pictures
- the main function will then run which will pick 2 random frames (ensuring the second occurs after the first) and determine the displacement between the two
- the file will be saved in the respective directory
- can change the value of sigma, the size of the window, and the threshold between eigenvalues in line 173


Q4: HOG.py: determines the histogram of oriented gradients in an image and saves the normalized histograms to a .txt file

NOTE: ensure the working directory contains the original image folder Q4 which then contains the various image files

- simply run the code
- can also change the values of the cells and sigma in line 128