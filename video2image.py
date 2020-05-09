import cv2
import os
import numpy as np
from HW3 import HW3


path = 'videos/Record_17.avi'
save_path = r'videos/images'


def video2image():
    """
    part2
    :return:
    """
    # Set up a video capture using cv2.VideoCapture('frames/%06d.jpg').
    cap = cv2.VideoCapture(path)
    # Show each frame of the video
    i = 0
    grays = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        window_name = 'image' + str(i) + '.jpg'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(window_name, gray)     # Show each frame of the video.
        grays.append(np.array(gray))
        i += 1
    # calculate the mean frame
    mean = np.mean([gray for gray in grays], axis=0).astype('uint8')
    # Show the average image, what we will call the "background" image.  The cars have disappeared!  Why?
    cv2.imshow('average frame', mean)
    # Save the background image to a PNG.
    cv2.imwrite('part2.png', mean)
    cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()

def bonus1():
    """Run the thresholding technique on each frame of the video and show the result as a movie.
    """
    cap = cv2.VideoCapture(path)
    threshold = 30
    size = 0
    Otsu_arr = []
    i = 0
    os.chdir(r'videos/image/')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # change to grayscale
        Ret_Otsu, Otsu = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
        height, width = np.shape(Otsu)
        Otsu = cv2.cvtColor(Otsu, cv2.COLOR_GRAY2BGR)
        size = (width, height)
        Otsu_arr.append(Otsu)
        cv2.imwrite('image'+str(i)+'.jpg', Otsu)
        i += 1
    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\videos')
    video = cv2.VideoWriter(filename="threshold1.avi", fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=15, frameSize=size)
    for Otsu in Otsu_arr:
        video.write(Otsu)
    video.release()

def bonus2():
    """Detect a bounding box around each car and show the result as a movie.
    """
    cap = cv2.VideoCapture(path)
    threshold = 30
    size = 0
    previous = None
    Otsu_arr = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # frame = (HW3.HPF(frame)*255).astype('uint8')
            current = frame
            if previous is not None:
                abs_diff = cv2.absdiff(previous, current)
                gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)      # change to grayscale
                Ret_Otsu, Otsu = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
                height, width = np.shape(Otsu)
                bound_img = np.copy(Otsu)
                bound_img = cv2.cvtColor(bound_img, cv2.COLOR_GRAY2BGR)
                blurr = cv2.blur(bound_img, ksize=(3, 2))
                # bound_img = bound_img - blurr
                contours, hierarchy = cv2.findContours(Otsu, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for cidx, cnt in enumerate(contours):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    # Draw a rectangle with blue line borders of thickness of 1 px
                    colors = (255,0,0)
                    #cv2.rectangle(bound_img, pt1=(x, y), pt2=(x + w, y + h), color=colors, thickness=1)
                Otsu_arr.append(bound_img)
                size = (width, height)
            previous = current
    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\videos')
    video = cv2.VideoWriter(filename="threshold2.avi", fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=20, frameSize=size)
    for Otsu in Otsu_arr:
        video.write(Otsu)
    video.release()


def shortenVideo():
    """Run the thresholding technique on each frame of the video and show the result as a movie.
    """
    cap = cv2.VideoCapture(path)
    height = width = depth = 0
    Otsu_arr = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        Otsu_arr.append(frame)
        if not i:
            height, width, depth = np.shape(frame)
    size = width, height
    print(size)
    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\videos')
    video = cv2.VideoWriter(filename="video1.avi", fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=10,
                            frameSize=size)
    for i in range(len(Otsu_arr)//4):
        video.write(Otsu_arr[i])
    video.release()


if __name__ == '__main__':
    shortenVideo()