# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default=r"videos\0.04.avi",
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
wait = 1
i = 0
pos = []
total_frame = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    total_frame += 1
    frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]
    """if total_frame == 1:
        video = cv2.VideoWriter(filename=(args["video"] + '_tracked' + '.avi'),
                                fourcc=cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps=30,
                                frameSize=(W, H))"""
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success

        if success:
            (x, y, w, h) = [int(v) for v in box]
            if not i:
                init_x, init_y = x + w/2, y + h/2
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.circle(frame, (x + w//2, y + h//2), 1,  (0, 255, 0), 2)
            cv2.line(frame, (int(init_x), int(init_y)), (x + w//2, y + h//2), (0, 255, 0), 2)
            norm = np.linalg.norm([x + w/2 - init_x, y + h/2 - init_y])
            curr_time = vs.get(cv2.CAP_PROP_POS_MSEC)
            pos.append([curr_time, norm])
            # video.write(frame)
        i += 1
        # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
            ("Time", curr_time),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then starpy opencv_object_tracker.py --video 0.04mm.mp4 --tracker csrtt the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

meas_length = vs.get(cv2.CAP_PROP_POS_MSEC)
vs.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
actual_length = vs.get(cv2.CAP_PROP_POS_MSEC)
start_time = pos[0][0]
max_pos = pos[-1][1]
print(max_pos)
for i in range(len(pos)):
    pos[i][0] -= start_time
    pos[i][0] *= (actual_length/ meas_length / 1000)    # actual time
    pos[i][1] /= max_pos    # normalized position


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()

np.savetxt("pos.csv", pos, delimiter=",")
# close all windows
cv2.destroyAllWindows()