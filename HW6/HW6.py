import cv2
import numpy as np
from matplotlib import pyplot as plt



def HW6():
    """Part 1: Read the left and right images.  Detect SIFT features on each image.
    Match the features using nearest neighbor matching and the ratio test.
    """
    img1 = cv2.imread('plushies_right.png')
    img2 = cv2.imread('plushies_left.png')
    RESIZE = 0.25
    size = int(img1.shape[1]*RESIZE), int(img1.shape[0]*RESIZE)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    kp1, des1 = SIFT(img1)
    kp2, des2 = SIFT(img2)
    # Create a "brute force" matcher (BFMatcher) object.
    bf = cv2.BFMatcher()
    # 2.At each video frame, compute the first two nearest-neighbor matches for each keypoint in the video frame
    matches = bf.knnMatch(des1, des2, k=2)
    # 4. Draw the matches between the video frame and the target image using cv.drawMatches()
    # 3.Apply the "ratio test" to filter out unreliable matches, using a threshold of 0.7
    # store all the good matches as per Lowe's ratio test.
    good = []
    src_pts = []
    dst_pts = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    MIN_MATCH_COUNT = 10


    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2)
        # 1.Fit a homography to the matches using findHomography() with the RANSAC option.
        # Note that you might need to invert the resulting homography matrix using np.linalg.inv().
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        """Part 2: Estimate the essential matrix between the two images using cv.findEssentialMatrix() and the cv.RANSAC option.
          Decompose the essential matrix into a translation vector and two possible rotation vectors using cv.decomposeEssentialMatrix().
            Choose the rotation matrix with no negative numbers on the diagonal as the correct one."""
        fx = fy = 485.82423388827533  # focal length
        cx = 134.875  # principle point
        cy = 239.875  # principle point
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        R = R1 if diag_pos(R1) else R2
        """Part 3: Rectify the two images using cv.stereoRectify(), cv.initUndistortRectifyMap() and cv.remap().
          Estimate the stereo disparity using StereoBM.  I recommend a max disparity of 32.  Show the disparity map."""
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, None, K, None, size, R, t)
        map_x1, map_y1 = cv2.initUndistortRectifyMap(K, None, R1, K, size, cv2.CV_32FC1)
        map_x2, map_y2 = cv2.initUndistortRectifyMap(K, None, R2, K, size, cv2.CV_32FC1)
        dst1 = cv2.remap(img1, map_x1, map_y1, cv2.INTER_LANCZOS4)
        dst2 = cv2.remap(img2, map_x2, map_y2, cv2.INTER_LANCZOS4)
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
        disparity = stereo.compute(cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY))

        plt.imshow(disparity, 'gray')
        cv2.imshow('dst1', dst1)
        cv2.imshow('dst2', dst2)
        cv2.imshow('matches', inlier_matches)
        plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    #cv2.imshow('inliner matches', inlier_matches)
    while cv2.waitKey(1) != ord('q'):
        1








def SIFT(img1):
    # part 1
    # 1.Load the "stones" image and convert it to grayscale.

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 2.Detect SIFT features and compute descriptors for them using sift.detectAndCompute().
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def diag_pos(M):
    pos = True
    for i in range(M.shape[0]):
        if M[i][i] < 0:
            print(M[i][i])
            pos = False
    return pos


if __name__ == '__main__':
    HW6()
