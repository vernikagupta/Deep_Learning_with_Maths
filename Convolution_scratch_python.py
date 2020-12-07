# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(img, kernel):
    '''Gray scale image and kernel as arguments passed 
    Grab the dimension of image and kernel'''
    ih, iw = img.shape[:2]
    kh, kw = kernel.shape[:2]
       
    # allocate memory for the output image, taking care to "pad"
    # the borders of the input image so the spatial size (i.e.,
    # width and height) are not reduced
    pad = (kw-1)//2
    print("pad value is {}".format(pad))
    image = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cv2.imshow("padded image",image)
    output = np.zeros((ih, iw), dtype="float")
    # sliding kernel over the input image
    # from left to right and top to bottom
    for y in np.arange(pad, ih+pad):
        for x in np.arange(pad, iw+pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()
            # store the convolved value in the output (x, y)-
            # coordinate of the output image at the same (x, y)-coordinates
            # (relative to the input image)
            output[y-pad, x-pad] = k 
    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='image to be convolved')
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image

laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

# construct the kernel bank, a list of kernels we’re going to appl
# using both our custom ‘convole‘ function and OpenCV’s ‘filter2D‘
# function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY))

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, kernel) in kernelBank:
    # apply the kernel to the grayscale image using both our custom
    # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)
    # show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convole".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

''' in most cases, we want our output image to have the same dimensions as our input
image. To ensure the dimensions are the same, we apply padding. Here we are
simply replicating the pixels along the border of the image, such that the
output image will match the dimensions of the input image'''
