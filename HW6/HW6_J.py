import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import copy


def HW6():
    """
    Part 1: Read the left and right images.  Detect SIFT features on each image.
        Match the features using nearest neighbor matching and the ratio test.
    Part 2: Estimate the essential matrix between the two images using cv.findEssentialMatrix()
        and the cv.RANSAC option.  Decompose the essential matrix into a translation vector and
        two possible rotation vectors using cv.decomposeEssentialMatrix().  Choose the rotation
        matrix with no negative numbers on the diagonal as the correct one.
    Part 3: Rectify the two images using cv.stereoRectify(), cv.initUndistortRectifyMap()
        and cv.remap().  Estimate the stereo disparity using StereoBM.  I recommend a max
        disparity of 32.  Show the disparity map.
    """
    focal = 485.82423388827533
    Cx = 134.875
    Cy = 239.875


    #obtain the left img feature
    right = cv.imread("plushies_right.png")
    row, column, _ = right.shape
    column = int(column*0.3)
    row = int(row*0.3)
    right = cv.resize(right, (column, row))
    gray = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)  # get key point and descriptor

    #obtain the right img feature
    left = cv.imread("plushies_left.png")
    row, column, _ = left.shape
    column = int(column*0.3)
    row = int(row*0.3)
    left = cv.resize(left, (column, row))
    gray = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray, None)  # get key point and descriptor

    #match the 2 imgs
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    match_imgs = cv.drawMatchesKnn(right,kp1,left,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    k = np.array([[focal, 0, Cx], [0, focal, Cy], [0, 0, 1]])
    E, mask = cv.findEssentialMat(pts1, pts2, k, cv.RANSAC)

    R1,R2,T = cv.decomposeEssentialMat(E)
    if (R1[0][0]>0 and R1[1][1]>0 and R1[2][2]>0):  R = R1
    else:   R = R2

    #help(cv.stereoRectify)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(k, None, k, None, (column, row), R, T)

    #help(cv.initUndistortRectifyMap)
    map1, map2 = cv.initUndistortRectifyMap(k, None, R1, k, (column, row), cv.CV_32FC1)
    map1_, map2_ = cv.initUndistortRectifyMap(k, None, R2, k, (column, row), cv.CV_32FC1)

    #help(cv.remap)
    right = cv.remap(right, map1, map2, cv.INTER_LINEAR)
    left = cv.remap(left, map1_, map2_, cv.INTER_LINEAR)

    right_ = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    left_ = cv.cvtColor(left, cv.COLOR_BGR2GRAY)

    #help(cv.StereoBM_create)
    retval = cv.StereoBM_create(numDisparities = 32, blockSize = 11)

    #help(retval.compute)
    disparity = retval.compute(right_, left_)

    plt.imshow(disparity, 'gray')
    plt.show()

    cv.imshow("left", left)
    cv.imshow("right", right)
    cv.imshow("match_img", match_imgs)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    HW6()

if __name__ == "__main__":
    main()