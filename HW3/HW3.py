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


    # bonus
    # Make your own custom hybrid image using two images that you find / capture
    img1 = cv2.imread(r'1.JPG').astype('float') / 255
    img1 = cv2.resize(img1, (360,360))
    img2 = cv2.imread(r'2.JPG').astype('float') / 255
    img2 = cv2.resize(img2, (360, 360))
    img1 = cv2.filter2D(src=img1, ddepth=-1, kernel=g)
    img2 = cv2.filter2D(src=img2, ddepth=-1, kernel=h)
    img3 = img1+img2
    cv2.imshow('img3.png', img3)
    cv2.waitKey(5000)


def LPF(img, kernel_size):
    img = img.astype('float') / 255
    # Step 2: Make a low-pass kernel of size 31x31 with sigma=5.  See cv.getGaussianKernel().
    g = cv2.getGaussianKernel(ksize=kernel_size, sigma=5)
    # This will return a vector g.  To create the kernel matrix, compute g = g*g.transpose().
    g = g*g.transpose()
    # Step 4: Filter the dog image with the low-pass kernel.  Filter the cat image with the high-pass kernel
    img = cv2.filter2D(src=img, ddepth=-1, kernel=g)
    return (img*255).astype('uint8')

def HPF(img, kernel_size):
    img = img.astype('float') / 255
    # Step 2: Make a low-pass kernel of size 31x31 with sigma=5.  See cv.getGaussianKernel().
    g = cv2.getGaussianKernel(ksize=kernel_size, sigma=5)
    # This will return a vector g.  To create the kernel matrix, compute g = g*g.transpose().
    g = g*g.transpose()
    # Step 3: Make a high-pass kernel using the low-pass kernel you made in the last step
    k = int((kernel_size-1)/2)         # mid point
    a = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    a[k][k] = 1     # generate all pass filer (one in the middle)
    h = a-g
    # Step 4: Filter the dog image with the low-pass kernel.  Filter the cat image with the high-pass kernel
    img = cv2.filter2D(src=img, ddepth=-1, kernel=h)
    return (img*255).astype('uint8')




if __name__ == '__main__':
    HW3()