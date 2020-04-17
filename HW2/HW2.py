"""
CPE 428
HW2
Conrad Chan
"""
import numpy as np
import os
import cv2


def HW2(video_name):
    """
    :return:
    """
    # part 1
    # Use a command line argument to specify the path of the video to open.
    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\HW2')
    # Convert each frame of the video to grayscale and apply a Gaussian blur of size 9x9 with sigma=2.
    cap = cv2.VideoCapture(video_name)
    i = 0
    frames = []
    width = height = 0
    f = 485.82423388827533  # focal length
    cx = 134.875     # principle point
    cy = 239.875     # principle point
    R = 3   # object radius
    count = 1
    X_prev = Y_prev = Z_prev = None
    stick_lengths = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        window_name = 'image' + str(i) + '.jpg'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(src=gray, ksize=(9, 9), sigmaX=2)
        # Run HoughCircles on each frame.
        rows = gray.shape[0]
        circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=rows/8,
                                   param1=95, param2=30, minRadius=1, maxRadius=60)
        # Draw each detected circle on the original color image and show the result.
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                count += 1
                center = x, y = i[0], i[1]
                # circle center
                cv2.circle(frame, center, radius=1, color=(0, 100, 100), thickness=1)
                # circle outline
                radius = i[2]
                cv2.circle(frame, center, radius, color=(255, 0, 255), thickness=1)

                # part 2
                # For each detected circle, compute the projected position of the ball in
                # camera coordinates ("remove" the focal length and principal point).
                x_camera = x-cx/f
                y_camera = y-cy/f
                # Compute the Z value of the ball's (X,Y,Z) coordinates based on the
                # known radius of the physical ball and the radius of the detected circle.
                Z = R/radius*f
                # Compute the X and Y values of the ball's (X,Y,Z) coordinates
                X = (x-cx)*Z/f
                Y = (y-cy)*Z/f
                # Write the Z of the ball (as an integer) on the image in the center of the
                # circle before you show the image.
                cv2.putText(img=frame, text=str(int(Z))+' cm', org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

                # part 3
                # Implement a function that calculates
                # the 2D projection of a 3D point (including the application of the intrinsics).
                # front_upper_right, front_upper_left, front_lower_right, front_lower_left
                front_points = cal_3D_pts(X+R, Y-R, Z-R), cal_3D_pts(X-R, Y-R, Z-R), cal_3D_pts(X+R, Y+R, Z-R), \
                               cal_3D_pts(X - R, Y + R, Z - R)
                # back_upper_right, back_upper_left, back_lower_right, back_lower_left
                back_points = cal_3D_pts(X+R, Y-R, Z+R), cal_3D_pts(X-R, Y-R, Z+R), cal_3D_pts(X+R, Y+R, Z+R), \
                              cal_3D_pts(X - R, Y + R, Z+R)
                # Implement a function that draws on an image the projection of a line between two 3D points.
                draw_lines(frame, front_points)
                draw_lines(frame, back_points)
                # Use the above functions to draw a 3D box on each detected ball.
                # The radius of the box should equal the radius of the ball.  See the example videos above.
                draw_box(frame, front_points, back_points)
                # Bonus
                if count%2:
                    stick_length = np.sqrt((X - X_prev)**2+(Y - Y_prev)**2+(Z - Z_prev)**2)
                    if int(stick_length):
                        stick_lengths.append(stick_length)
                X_prev = X
                Y_prev = Y
                Z_prev = Z
        height, width = np.shape(gray)
        frames.append(frame)
    # show the result image
    cv2.imshow(video_name+'_show.png', frames[50])
    video_name = [token for token in video_name if token not in ".mov"]
    video_name = ''.join(video_name)
    if video_name == 'wand':
        mean = np.mean([stick_length for stick_length in stick_lengths])
        stdiv = np.std(stick_lengths)
        error = abs(mean - 36) / 36 * 100
        print('mean of ' + video_name + ' is ' + str(format(mean, '.2f')) + ' cm')
        print('standard deviation of ' + video_name + ' is ' + str(format(stdiv, '.2f')))
        print('compare to the true measurement of 36cm, the % error = ' + str(format(error, '.2f')) + '%')

    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\HW2\output')
    video = cv2.VideoWriter(filename=(video_name+'.avi'), fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=30,
                            frameSize=(width, height))
    for frame in frames:
        video.write(frame)
    video.release()
    cv2.waitKey(5000)


def cal_3D_pts(X, Y, Z):
    f = 485.82423388827533
    cx = 134.875     # principle point
    cy = 239.875     # principle point
    x = f*X/Z+cx
    y = f*Y/Z+cy
    return int(x), int(y)



def draw_box(img, front_pts, back_pts):
    front_upper_right, front_upper_left, front_lower_right, front_lower_left = \
        front_pts[0], front_pts[1], front_pts[2], front_pts[3]
    back_upper_right, back_upper_left, back_lower_right, back_lower_left = \
        back_pts[0], back_pts[1], back_pts[2], back_pts[3]

    cv2.line(img, front_upper_right, back_upper_right, color=(255, 0, 255), thickness=1)
    cv2.line(img, front_upper_left, back_upper_left, color=(255, 0, 255), thickness=1)
    cv2.line(img, front_lower_right, back_lower_right, color=(255, 0, 255), thickness=1)
    cv2.line(img, front_lower_left, back_lower_left, color=(255, 0, 255), thickness=1)


def draw_lines(img, points):
    upper_right, upper_left, lower_right, lower_left = \
        points[0], points[1], points[2], points[3]
    cv2.line(img, upper_right, upper_left, color=(255, 0, 255), thickness=1)
    cv2.line(img, upper_right, lower_right, color=(255, 0, 255), thickness=1)
    cv2.line(img, lower_right, lower_left, color=(255, 0, 255), thickness=1)
    cv2.line(img, lower_left, upper_left, color=(255, 0, 255), thickness=1)

if __name__ == '__main__':
    HW2('wand.mov')
    HW2('red.mov')
    HW2('blue.mov')
    HW2('redblue.mov')