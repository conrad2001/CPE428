import cv2
import numpy as np
import matplotlib.pyplot as plt

def HW4(bonus=False):
    """

    :return:
    """
    MIN_MATCH_COUNT = 10
    # part 1
    # 1.Load the "stones" image and convert it to grayscale.
    img1 = cv2.imread('stones.png')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 2.Detect SIFT features and compute descriptors for them using sift.detectAndCompute().
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des_stone = sift.detectAndCompute(gray, None)
    img1 = cv2.drawKeypoints(img1,kp1,img1)
    # 3.Show the SIFT features on the image.
    # cv2.imshow('sift', draw_keypts)
    # 4.Iterate through the frames of the video and detect SIFT features & descriptors on each frame
    cap = cv2.VideoCapture('input.mov')
    # Part 2, 1.Create a "brute force" matcher (BFMatcher) object.
    bf = cv2.BFMatcher()
    i = 0
    inlier_ = outlier_ = 0
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    filename = 'overlay_Chan.avi' if bonus else 'boarder_Chan.avi'
    video = cv2.VideoWriter(filename=filename, fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=30,
                            frameSize=frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp2, des_frame = sift.detectAndCompute(gray, None)
        # frames.append(cv2.drawKeypoints(frame, kp2, frame))
        # part 2
        # 2.At each video frame, compute the first two nearest-neighbor matches for each keypoint in the video frame
        # (matching to the target "stones" image).
        matches = bf.knnMatch(des_stone, des_frame, k=2)
        # 3.Apply the "ratio test" to filter out unreliable matches, using a threshold of 0.7
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # 4. Draw the matches between the video frame and the target image using cv.drawMatches()
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # part 3
            # 1.Fit a homography to the matches using findHomography() with the RANSAC option.
            # Note that you might need to invert the resulting homography matrix using np.linalg.inv().
            H, inlier = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = inlier.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            # Draw polygon around the tracking target in the video frame using cv.polylines()
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
        # 4.Draw the inlier matches between the video frame and the target image using cv.drawMatches()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        inlier_matches = cv2.drawMatches(img1, kp1, frame, kp2, good, None, **draw_params)
        # cv2.imwrite('inlier_matches'+str(i)+'.png', inlier_matches)
        if not i:
            inlier_h, inlier_w, inlier_d = inlier_matches.shape
            video2 = cv2.VideoWriter(filename='matches.avi', fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                                     fps=15,
                                     frameSize=(inlier_w, inlier_h))
        video2.write(inlier_matches)
        if bonus:
            # bonus
            # Use the calculate homography to warp the overlay image (see cv.perspectiveWarp() ).
            overlay = cv2.imread('overlay.png')
            # To put the overlay on the image, you need to create a mask which has ones over the warped overlay and zeros
            # outside of it.  To do this, create an image of all ones and
            # warp it using perspectiveWarp().
            mask = np.ones(overlay.shape)
            mask = cv2.warpPerspective(mask, H, frame_size)
            overlay = cv2.warpPerspective(overlay, H, frame_size)
            # Calculate the composite image as (1-mask) * frame + mask * overlay.
            comp_img = np.multiply(1-mask, frame) + np.multiply(mask, overlay)
            frame = np.uint8(comp_img)
        video.write(frame)
        i += 1
    # part 3  5. Calculate the average percentage of inliers.
    for i in range(len(good)):
        if matchesMask[i]:
            inlier_ += 1
        else:
            outlier_ += 1
    percent_inlier = inlier_ / (inlier_ + outlier_) * 100
    percent_outlier = 100-percent_inlier
    print("percent inlier = %.6f" % percent_inlier)
    print("percent outlier = %.6f" % percent_outlier)
    video.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # True to generate output with overlay
    # False to generate output with border
    HW4(True)
