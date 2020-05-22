import cv2 as cv
import numpy as np
import copy
from random import seed
from random import random
from imutils.video import VideoStream
from imutils.video import FPS

def back_ground(screen_size):
    #get background img
    background_img = cv.imread("Springtime-Golf-Course_AmeriTurf_2019.jpg")
    background_img = cv.resize(background_img,screen_size)
    cv.line(background_img,maping(np.matrix([[-30], [1], [0]])),maping(np.matrix([[-30], [500], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [1], [0]])),maping(np.matrix([[30], [500], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [1], [40]])),maping(np.matrix([[-30], [500], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [1], [40]])),maping(np.matrix([[30], [500], [40]])),(0,255,0),1)

    cv.line(background_img,maping(np.matrix([[-30], [500], [0]])),maping(np.matrix([[30], [500], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [400], [0]])),maping(np.matrix([[30], [400], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [300], [0]])),maping(np.matrix([[30], [300], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [200], [0]])),maping(np.matrix([[30], [200], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [100], [0]])),maping(np.matrix([[30], [100], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [50], [0]])),maping(np.matrix([[30], [50], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [150], [0]])),maping(np.matrix([[30], [150], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [250], [0]])),maping(np.matrix([[30], [250], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [350], [0]])),maping(np.matrix([[30], [350], [0]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [450], [0]])),maping(np.matrix([[30], [450], [0]])),(0,255,0),1)

    cv.line(background_img,maping(np.matrix([[-30], [500], [0]])),maping(np.matrix([[-30], [500], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [400], [0]])),maping(np.matrix([[-30], [400], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [300], [0]])),maping(np.matrix([[-30], [300], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [200], [0]])),maping(np.matrix([[-30], [200], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [100], [0]])),maping(np.matrix([[-30], [100], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [50], [0]])),maping(np.matrix([[-30], [50], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [150], [0]])),maping(np.matrix([[-30], [150], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [250], [0]])),maping(np.matrix([[-30], [250], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [350], [0]])),maping(np.matrix([[-30], [350], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [450], [0]])),maping(np.matrix([[-30], [450], [40]])),(0,255,0),1)

    cv.line(background_img,maping(np.matrix([[30], [500], [0]])),maping(np.matrix([[30], [500], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [400], [0]])),maping(np.matrix([[30], [400], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [300], [0]])),maping(np.matrix([[30], [300], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [200], [0]])),maping(np.matrix([[30], [200], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [100], [0]])),maping(np.matrix([[30], [100], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [50], [0]])),maping(np.matrix([[30], [50], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [150], [0]])),maping(np.matrix([[30], [150], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [250], [0]])),maping(np.matrix([[30], [250], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [350], [0]])),maping(np.matrix([[30], [350], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[30], [450], [0]])),maping(np.matrix([[30], [450], [40]])),(0,255,0),1)

    cv.line(background_img,maping(np.matrix([[-30], [500], [40]])),maping(np.matrix([[30], [500], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [400], [40]])),maping(np.matrix([[30], [400], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [300], [40]])),maping(np.matrix([[30], [300], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [200], [40]])),maping(np.matrix([[30], [200], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [100], [40]])),maping(np.matrix([[30], [100], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [50], [40]])),maping(np.matrix([[30], [50], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [150], [40]])),maping(np.matrix([[30], [150], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [250], [40]])),maping(np.matrix([[30], [250], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [350], [40]])),maping(np.matrix([[30], [350], [40]])),(0,255,0),1)
    cv.line(background_img,maping(np.matrix([[-30], [450], [40]])),maping(np.matrix([[30], [450], [40]])),(0,255,0),1)

    return background_img

def maping(point):
    actual_x = point[0][0]
    actual_y = point[1][0]
    actual_z = point[2][0]-10
    x = actual_x/actual_y*focal_l+Cx
    y = -actual_z/actual_y*focal_l+Cy

    return (int(x), int(y))

def draw_man(point, img):
    actial_x, actual_y, actual_z = point

    man_loca = np.matrix([[actial_x], [actual_y], [actual_z]])
    p1 = man_loca + np.matrix([[1], [1], [0]])
    p2 = p1 + np.matrix([[0], [0], [2]])
    p3 = man_loca + np.matrix([[-1], [1], [0]])
    p4 = p3 + np.matrix([[0], [0], [2]])
    p5 = man_loca + np.matrix([[1], [-1], [0]])
    p6 = p5 + np.matrix([[0], [0], [2]])
    p7 = man_loca + np.matrix([[-1], [-1], [0]])
    p8 = p7 + np.matrix([[0], [0], [2]])
    cv.line(img,maping(p1),maping(p2),(255,0,0),2)
    cv.line(img,maping(p1),maping(p3),(255,0,0),2)
    cv.line(img,maping(p1),maping(p5),(255,0,0),2)
    cv.line(img,maping(p2),maping(p4),(255,0,0),2)
    cv.line(img,maping(p2),maping(p6),(255,0,0),2)
    cv.line(img,maping(p3),maping(p7),(255,0,0),2)
    cv.line(img,maping(p3),maping(p4),(255,0,0),2)
    cv.line(img,maping(p4),maping(p8),(255,0,0),2)
    cv.line(img,maping(p5),maping(p6),(255,0,0),2)
    cv.line(img,maping(p5),maping(p7),(255,0,0),2)
    cv.line(img,maping(p6),maping(p8),(255,0,0),2)
    cv.line(img,maping(p7),maping(p8),(255,0,0),2)

def get_player_img(focal_l, Cx, Cy, cap):
    seed(1)
    head_size = 50  #define half of the head size in pixels
    return_img_size = 25
    return_imgs = []

    while 1:
        ret, img = cap.read()
        #print(img.shape) (480,640)
        center_y, center_x, _ = img.shape
        center_x = int(center_x/2)
        center_y = int(center_y/2)


        color = (int(random()*255),int(random()*255),int(random()*255))
        cv.putText(img,'Stand 1m away. Place your head in the box!',(30,40), cv.FONT_HERSHEY_SIMPLEX, 0.8,color,2,cv.LINE_AA)
        cv.rectangle(img,(center_x-head_size,center_y-head_size),(center_x+head_size,center_y+head_size),color,2)

        #face detection and drawing
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            #cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            area = w*h
            #print(area)
            if area < 8500 and area > 6000 and w/2+x > center_x-10 and w/2+x < center_x+10 and h/2+y > center_y-15 and h/2+y < center_y+15:
                return_img = img[center_y-return_img_size+20:center_y+return_img_size+20, center_x-return_img_size:center_x+return_img_size]
                return_imgs.append(return_img)
            elif area > 8500:
                cv.putText(img,'Stand father away!',(10,100), cv.FONT_HERSHEY_SIMPLEX, 1,color,2,cv.LINE_AA)
            elif area < 6000:
                cv.putText(img,'Stand closer!',(10,100), cv.FONT_HERSHEY_SIMPLEX, 1,color,2,cv.LINE_AA)

        cv.imshow('game',img)
        k = cv.waitKey(30) & 0xff
        if len(return_imgs) == 20:
            break
    cv.destroyAllWindows()
    return return_imgs[10]


def get_man_pos(player_img, frame, kp1, des1, sift):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    """
    if len(good) > 10:
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        H, inliers = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = inliers.ravel().tolist()
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        good = []
        for i in range(len(good)):
            if matchesMask[i]:
                good.append(good[i])
    """
    frame_2 = cv.drawMatchesKnn(player_img,kp1,frame,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    """
    x_lst = []
    y_lst = []
    for m in good:
        x_,y_ = kp2[m[0].trainIdx].pt
        x_lst.append(x_)
        y_lst.append(y_)
    """

    error = 10
    point_matrix = np.matrix([[],[]])
    for m in good:
        x_,y_ = kp2[m[0].trainIdx].pt
        point_matrix = np.c_[point_matrix, np.matrix([[x_],[y_]])]

    has_error = True
    while has_error:
        has_error = False
        _, num_pts = point_matrix.shape

        sum = point_matrix.sum(axis=1)
        avg = sum/num_pts   #[[x_avg],[y_avg]]

        if num_pts>7:
            point_matrix = point_matrix [ :, point_matrix[0].argsort()] #sorted by x
            point_matrix = point_matrix.reshape(2,-1)

            if point_matrix[(0,num_pts-1)] > point_matrix[(0,num_pts-2)] + error:
                point_matrix = point_matrix[:,:-1]
                num_pts = num_pts-1
                has_error = True

            if point_matrix[(0,0)] < point_matrix[(0,1)] - error:
                point_matrix = point_matrix[:,1:]
                num_pts = num_pts-1
                has_error = True

            point_matrix = point_matrix [ :, point_matrix[1].argsort()] #sorted by y
            point_matrix = point_matrix.reshape(2,-1)

            if point_matrix[(1,num_pts-1)] > point_matrix[(1,num_pts-2)] + error:
                point_matrix = point_matrix[:,:-1]
                num_pts = num_pts-1
                has_error = True

            if point_matrix[(1,0)] < point_matrix[(1,1)] - error:
                point_matrix = point_matrix[:,1:]
                num_pts = num_pts-1
                has_error = True
        else:
            break

    _, num_pts = point_matrix.shape

    """
    for i in range(num_pts):
        cv.circle(frame, (int(point_matrix[(0,i)]), int(point_matrix[(1,i)])), 5, (0,255,0), 2)   #(img, center, radius, color, thickness)

    cv.imshow('game',frame)
    """

    if num_pts > 0:
        sum = point_matrix.sum(axis=1)
        avg = sum/num_pts   #[[x_avg],[y_avg]]
        return (int(avg[0,:]), int(avg[1,:]))
    else:
        return None

def map_man_2_img(man_pos, original_map, final_map):
    man_x, man_y = man_pos
    ori_x, ori_y = original_map
    map_x, map_z = final_map

    if man_x< 60:
        man_x = 60
    if man_y < 80:
        man_y = 80
    if man_x > ori_x-60:
        man_x = ori_x-60
    if man_y > ori_y-80:
        man_y = ori_y-80

    man_x -= 60
    man_y -= 80
    ori_x -= 120
    ori_y -= 160

    x = map_x/2 - man_x/ori_x*map_x
    y = map_z - man_y/ori_y*map_z

    return (int(x),70,int(y))
    #return (int(),80,int())



def final():
    global final_img, focal_l, Cx, Cy
    man_pos = None
    cap = cv.VideoCapture(0)

    screen_size = (1300, 650)
    screen_width, screen_hight = screen_size
    Cx = screen_width / 2.0
    Cy = screen_hight / 2.0 + 40
    focal_l = 2500
    background_img = back_ground(screen_size)
    head_size = 100  # define half of the head size in pixels
    # extract the OpenCV version info
    (major, minor) = cv.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv.Tracker_create("csrt".upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    else:
        tracker = cv.TrackerCSRT_create()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    vs = VideoStream(src=0).start()
    fps = None
    face_cascade = None
    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = cv.resize(frame, screen_size)
        (H, W) = frame.shape[:2]
        if initBB is None:
            center_y, center_x, _ = frame.shape
            center_x = int(center_x / 2)
            center_y = int(center_y / 2)

            color = (int(random() * 255), int(random() * 255), int(random() * 255))
            cv.putText(frame, 'Place your head in the box!', (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                       color, 2,
                       cv.LINE_AA)
            cv.rectangle(frame, (center_x - head_size, center_y - head_size),
                         (center_x + head_size, center_y + head_size),
                         color, 2)

            # face detection and drawing
            face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                area = w * h
                # print(area)
                # print(area)
                if area < 50000 and area > 45000:
                    # select the bounding box of the object we want to track
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then starpy opencv_object_tracker.py --video 0.04mm.mp4 --tracker csrtt the FPS throughput estimator as well
                    initBB = (x, y, w, h)
                    tracker.init(frame, initBB)
                    fps = FPS().start()
                    # if the `q` key was pressed, break from the loop
                elif area > 45000:
                    cv.putText(frame, 'Stand father away!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                               cv.LINE_AA)
                elif area < 50000:
                    cv.putText(frame, 'Stand closer!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

        # check to see if we are currently tracking an object
        elif initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                man_pos = x + w // 2, y + h if x < screen_size[0] and y < screen_size[1] else man_pos
                # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.circle(frame, (x + w // 2, y + h // 2), 1, (0, 255, 0), 5)
                # cv.line(frame, (int(init_x), int(init_y)), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                # norm = np.linalg.norm([x + w / 2 - init_x, y + h / 2 - init_y])
                # curr_time = vs.get(cv.CAP_PROP_POS_MSEC)

        origin_frame_y, origin_frame_x, _ = frame.shape
        #frame = cv.resize(frame, screen_size)
        final_img = copy.copy(background_img)

        if initBB is None:
            # show the output frame
            cv.imshow("game", frame)

        elif man_pos is not None and man_pos[0] < screen_size[0] and man_pos[1] < screen_size[1]:
            man_on_img_pos = map_man_2_img(man_pos, (origin_frame_x, origin_frame_y), (60, 38))  # (x,z) of man
            draw_man(man_on_img_pos, final_img)  # accept(x,y), x: left/right   y: front/back   z: hight

            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", "csrt"),
                ("Success", "Yes"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv.putText(final_img, text, (10, H - ((i * 20) + 20)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv.imshow('game', final_img)
        else:
            cv.imshow("game", background_img)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
    cap.release()
    cv.destroyAllWindows()




def main():
    final()

if __name__ == "__main__":
    main()
