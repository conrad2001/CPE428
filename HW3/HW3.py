"""
CPE 428
HW 3
Conrad Chan
"""

import cv2
import os
import numpy as np



def HW3():
    """

    :return:
    """
    # Step 1: Load the cat and dog images, convert to floating point on [0 1] range
    dog = cv2.imread(r'dog.bmp').astype('float') / 255
    cat = cv2.imread(r'cat.bmp').astype('float') / 255

    # Step 2: Make a low-pass kernel of size 31x31 with sigma=5.  See cv.getGaussianKernel().
    kernel_size = 31
    g = cv2.getGaussianKernel(ksize=kernel_size, sigma=5)
    # This will return a vector g.  To create the kernel matrix, compute g = g*g.transpose().
    g = g*g.transpose()
    # Note: if you want to show the kernel using imshow(), you need to scale it up by multiplying by 255, for example.
    # cv2.imshow('gaussian kernel', g*255)
    # cv2.waitKey(1000)

    # Step 3: Make a high-pass kernel using the low-pass kernel you made in the last step
    k = int((kernel_size-1)/2)         # mid point
    a = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    a[k][k] = 1     # generate all pass filer (one in the middle)
    h = a-g

    # Step 4: Filter the dog image with the low-pass kernel.  Filter the cat image with the high-pass kernel
    dog = cv2.filter2D(src=dog, ddepth=-1, kernel=g)
    cat = cv2.filter2D(src=cat, ddepth=-1, kernel=h)
    cv2.imshow('dog', dog)
    cv2.imshow('cat', cat)

    # Step 5: Add the low-passed dog and high-passed cat together to produce the hybrid image.
    hybrid = dog+cat
    cv2.imshow('hybrid_out.png', hybrid)
    cv2.waitKey(5000)

    # bonus
    # Make your own custom hybrid image using two images that you find / capture


if __name__ == '__main__':
    HW3()