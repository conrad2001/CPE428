import cv2 as cv
import numpy as np
import copy
from random import random
from imutils.video import VideoStream
from imutils.video import FPS
from pygame import mixer  # Load the popular external library
import os


def back_ground(screen_size):
    #get background img
    background_img = cv.imread("background2.jpg")
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

    p1 = maping(p1)
    p2 = maping(p2)
    p3 = maping(p3)
    p4 = maping(p4)
    p5 = maping(p5)
    p6 = maping(p6)
    p7 = maping(p7)
    p8 = maping(p8)


    if actial_x <= 0 and (actual_z+1) >= 10: # left up
        cv.fillConvexPoly( img, np.array([p5,p1,p2,p6]),  (155,0,0) )
        cv.fillConvexPoly(img, np.array([p7, p5, p1, p3]), (55, 0, 0))
    if actial_x <= 0 and (actual_z+1) < 10: #left down
        cv.fillConvexPoly(img, np.array([p5, p1, p2, p6]), (155, 0, 0))
        cv.fillConvexPoly( img, np.array([p8,p6,p2,p4]), (55,0,0) )
    if actial_x > 0 and (actual_z+1) >= 10: #right up
        cv.fillConvexPoly( img, np.array([p8,p7,p3,p4]), (155,0,0) )
        cv.fillConvexPoly(img, np.array([p7, p5, p1, p3]), (55, 0, 0))
    if actial_x > 0 and (actual_z+1) < 10: #right down
        cv.fillConvexPoly( img, np.array([p8,p6,p2,p4]), (55,0,0) )
        cv.fillConvexPoly( img, np.array([p8,p7,p3,p4]), (155,0,0) )

    cv.fillConvexPoly(img, np.array([p8, p6, p5, p7]), (255, 0, 0))


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
    z = map_z - man_y/ori_y*map_z

    return (int(x),70,int(z))
    #return (int(),80,int())

def get_tracker(face_cascade, vs, screen_size, tracker):
    initBB = None
    while initBB is None:
        frame = vs.read()
        frame_ = copy.copy(frame)
        frame = cv.resize(frame, screen_size)
        head_size = 55

        center_y_, center_x_, _ = frame.shape    #after scale
        center_x_ = int(center_x_ / 2)
        center_y_ = int(center_y_ / 2)

        center_y, center_x, _ = frame_.shape    #before scale
        center_x = int(center_x / 2)
        center_y = int(center_y / 2)

        color = (int(random() * 255), int(random() * 255), int(random() * 255))
        cv.putText(frame_, 'Place your head in the box!', (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        cv.rectangle(frame_, (center_x - head_size, center_y - head_size), (center_x + head_size, center_y + head_size), color, 2)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            area = w * h
            # print(area)
            # print(area)
            if area < 30000 and area > 20000 and w/2+x > center_x_-20 and w/2+x < center_x_+20 and h/2+y > center_y_-30 and h/2+y < center_y_+30:
                # select the bounding box of the object we want to track
                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then starpy opencv_object_tracker.py --video 0.04mm.mp4 --tracker csrtt the FPS throughput estimator as well
                initBB = (x, y, w, h)
                tracker.init(frame, initBB)
                fps = FPS().start()
                # if the `q` key was pressed, break from the loop
            elif area > 30000:
                cv.putText(frame_, 'Stand father away!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
            elif area < 20000:
                cv.putText(frame_, 'Stand closer!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

        cv.imshow("game", frame_)
        cv.waitKey(10)

    cv.destroyAllWindows()

    return (tracker, fps)

def get_man_pos(tracker, frame, screen_size, man_pos):
    # grab the new bounding box coordinates of the object
    (success, box) = tracker.update(frame)
    # check to see if the tracking was a success
    if success:
        (x, y, w, h) = [int(v) for v in box]
        man_pos = (x + w // 2, y + h//2) if x < screen_size[0] and y < screen_size[1] else man_pos
        return (x, y, w, h)
    return man_pos

def info_update(final_img, fps, speed, enemy_size, num_hits, score):
    (H, W) = final_img.shape[:2]
    # update the FPS counter
    fps.update()
    fps.stop()
    # initialize the set of information we'll be displaying on
    # the frame

    info = [
        ("Tracker", "csrt"),
        ("Success", "Yes"),
        ("FPS", "{:.2f}".format(fps.fps())),
        ("speed", "{:.2f}".format(speed)),
        ("#enemys", enemy_size),
        ("hits", num_hits)
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(final_img, text, (10, H - ((i * 30) + 20)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    blood_test = "HP: "
    for i in range(score*5):
        blood_test += "I"

    cv.putText(final_img, blood_test, (100,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



class enemy:
    def __init__(self):
        self.area = 64
        self.w = random()*14+1
        self.h = self.area/self.w
        self.d = random() * 5 + 4
        self.x = random()*(60-self.w)-(30-self.w/2)
        self.y = random()*600+600
        self.z = random()*(38-self.h)


    def get_pos(self):
        return (self.x,self.y,self.z)
    def get_size(self):
        return (self.w,self.h)
    def update(self, step):
        self.y -= step

        if (self.y<=step+1):
            self.y = 600
            self.w = random() * 14 + 1
            self.h = self.area / self.w
            self.d = random() * 5 + 4
            self.x = random() * (60 - self.w) - (30 - self.w / 2)
            self.z = random() * (38 - self.h)

    def draw_enemy(self, img):
        if self.y<70: return
        man_loca = np.matrix([[self.x], [self.y], [self.z]])
        w = self.w
        h = self.h
        depth = self.d
        p1 = man_loca + np.matrix([[w/2], [depth/2], [0]])
        p2 = p1 + np.matrix([[0], [0], [h]])
        p3 = man_loca + np.matrix([[-w/2], [depth/2], [0]])
        p4 = p3 + np.matrix([[0], [0], [h]])
        p5 = man_loca + np.matrix([[w/2], [-depth/2], [0]])
        p6 = p5 + np.matrix([[0], [0], [h]])
        p7 = man_loca + np.matrix([[-w/2], [-depth/2], [0]])
        p8 = p7 + np.matrix([[0], [0], [h]])

        p1 = maping(p1)
        p2 = maping(p2)
        p3 = maping(p3)
        p4 = maping(p4)
        p5 = maping(p5)
        p6 = maping(p6)
        p7 = maping(p7)
        p8 = maping(p8)


        if self.x <= 0 and (self.z + self.h/2) >= 10:  # left up
            cv.fillConvexPoly(img, np.array([p5, p1, p2, p6]), (0, 0, 155))
            cv.fillConvexPoly(img, np.array([p7, p5, p1, p3]), (0, 0, 55))
        if self.x  <= 0 and (self.z + self.h/2) < 10:  # left down
            cv.fillConvexPoly(img, np.array([p5, p1, p2, p6]), (0, 0, 155))
            cv.fillConvexPoly(img, np.array([p8, p6, p2, p4]), (0, 0, 55))
        if self.x  > 0 and (self.z + self.h/2) >= 10:  # right up
            cv.fillConvexPoly(img, np.array([p8, p7, p3, p4]), (0, 0, 155))
            cv.fillConvexPoly(img, np.array([p7, p5, p1, p3]), (0, 0, 55))
        if self.x  > 0 and (self.z + self.h/2) < 10:  # right down
            cv.fillConvexPoly(img, np.array([p8, p6, p2, p4]), (0, 0, 55))
            cv.fillConvexPoly(img, np.array([p8, p7, p3, p4]), (0, 0, 155))

        cv.fillConvexPoly(img, np.array([p8, p6, p5, p7]), (0, 0, 255))

    def overlap(self, man_pos, speed):
        x,y,z = man_pos
        if abs(self.y-y) < (1+speed/2):    #+3 is for accounting error
            if ((z>self.z) and (z-self.z)<self.h) or ((self.z>z) and (self.z-z)<2):
                if abs(x-self.x)<(self.w/2+1):
                    return True
        return False


def final():
    global final_img, focal_l, Cx, Cy

    screen_size = (1300, 650)
    screen_width, screen_hight = screen_size
    Cx = screen_width / 2.0
    Cy = screen_hight / 2.0 + 100
    focal_l = 2000
    playing = True
    pos = []
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # create tracker
    (major, minor) = cv.__version__.split(".")[:2]
    if int(major) == 3 and int(minor) < 3:
        tracker = cv.Tracker_create("csrt".upper())
    else:
        tracker = cv.TrackerCSRT_create()

    vs = VideoStream(src=0).start()
    tracker, fps = get_tracker(face_cascade, vs, screen_size, tracker)
    frame_count = 0
    man_pos = None
    os.chdir(r'C:\Users\User01\PycharmProjects\CPE428\evaluation')
    video = cv.VideoWriter(filename=('opencv_tracker.avi'), fourcc=cv.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                           fps=12,
                           frameSize=screen_size)


    while playing:
        frame = vs.read() # check to see if we have reached the end of the stream
        if frame is None:
            break

        frame = cv.resize(frame, screen_size)

        man_pos = get_man_pos(tracker, frame, screen_size, man_pos) #update tracker and get man position
        x, y, w, h = man_pos
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv.imshow('evaluation', frame)
        video.write(frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or frame_count == 100:
            break
        frame_count += 1
        pos.append([x, y, w, h, x+w/2, y+h/2])
    np.savetxt("track_pos.csv", pos, delimiter=",")
    video.release()
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
    cv.destroyAllWindows()

def main():
    again = True
    while again:
        again = final()

if __name__ == "__main__":
    main()
