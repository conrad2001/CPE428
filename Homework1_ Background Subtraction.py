"""
HW 1
Conrad Chan
CPE 428
"""

import numpy as np
import cv2
import os


def part1():
    #part 1
    """
    """
    # 1. In Python, load 'frames/000000.jpg' using cv2.imread() and show it using cv2.imshow().
    path = r'frames\000000.jpg'
    image = cv2.imread(path)
    image = np.array(image, dtype=np.uint8)
    cv2.imshow('000000.jpg', image)
    # Print the shape of the Numpy array containing the image -- what do the sizes of the dimensions mean?
    #print(np.shape(image))
    # Print the image itself -- what do these numbers mean?
    print(image)
    # Convert the image to grayscale using cv2.cvtColor() and show it.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', gray)
    # Save the grayscale image to a PNG using cv2.imwrite().
    directory = r'C:\Users\User01\PycharmProjects\CPE428\gray'
    os.chdir(directory)     # change to specified directory
    cv2.imwrite('0.png', gray)
    cv2.waitKey(3000)

def part2():
    """
    part2
    :return:
    """
    # Set up a video capture using cv2.VideoCapture('frames/%06d.jpg').
    cap = cv2.VideoCapture('frames/%06d.jpg')

    # Show each frame of the video
    i = 0
    grays = []
    directory = r'C:\Users\User01\PycharmProjects\CPE428\part2'
    os.chdir(directory)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        window_name = 'image' + str(i) + '.jpg'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow(window_name, gray)     # Show each frame of the video.
        cv2.imwrite("%02d" % i + '.jpg', gray)
        grays.append(np.array(gray))
        i += 1
    # calculate the mean frame
    mean = np.mean([gray for gray in grays], axis=0).astype('uint8')
    print(mean)
    # Show the average image, what we will call the "background" image.  The cars have disappeared!  Why?
    cv2.imshow('average frame', mean)
    # Save the background image to a PNG.
    directory = r'C:\Users\User01\PycharmProjects\CPE428\gray'
    os.chdir(directory)
    cv2.imwrite('1.png', mean)
    cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()

def part3():
    """
    part3
    :return:
    """

    # Load the images from parts I and II.
    image0 = cv2.imread(r'gray\0.png', cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(r'gray\1.png', cv2.IMREAD_GRAYSCALE)
    # calculate the absolute difference btw image 1 and 2
    abs_diff = cv2.absdiff(image0, image1)
    threshold = 30
    # Threshold the absolute difference image to obtain a binary mask corresponding to the foreground pixels
    Ret, threshold_frame = cv2.threshold(abs_diff, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold" + str(threshold), threshold_frame)
    # Threshold the absolute difference image using Otsu's method
    Ret_Otsu, Otsu = cv2.threshold(abs_diff, threshold, 255, cv2.THRESH_OTSU)
    cv2.imshow("Otsu's threshold" + str(threshold), Otsu)
    cv2.waitKey(10000)
    # How well does each technique work?  What could be improved about the output?

# Bonus:
#
def movie():
    """Run the thresholding technique on each frame of the video and show the result as a movie.
    """
    cap = cv2.VideoCapture('frames/%06d.jpg')
    i = 0
    threshold = 30
    size = 0
    video_arr = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # change to grayscale
        Ret_Otsu, Otsu = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
        height, width = np.shape(Otsu)
        Otsu = cv2.cvtColor(Otsu, cv2.COLOR_GRAY2BGR)
        size = (width, height)
        video_arr.append(Otsu)
        i += 1
    directory = r'C:\Users\User01\PycharmProjects\CPE428\movie'
    os.chdir(directory)
    video = cv2.VideoWriter(filename="Otsu.avi", fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=5, frameSize=size)
    for Otsu in video_arr:
        video.write(Otsu)
    video.release()

# Detect a bounding box around each car and show the result as a movie.
def main():
    movie()

if __name__ == '__main__':
    main()

