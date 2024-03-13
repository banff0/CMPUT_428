import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import lk
import LKTracker
import time

def get_bbox(pt):
    size = 15
    bbox = [[pt[0]-size, pt[1]-size], 
            [pt[0]+size, pt[1]-size],
            [pt[0]+size, pt[1]+size],
            [pt[0]-size, pt[1]+size]]
    print(bbox, pt)
    return np.array(bbox).T

def get_center(pts):
    pts = np.concatenate((pts.T, np.ones([4, 1])), axis=1) 

    l1 = np.cross(pts[0], pts[2])
    l2 = np.cross(pts[1], pts[3])

    pm = np.cross(l1, l2)

    norm_pm = pm / pm[2]
    norm_pm = norm_pm.astype(np.int32)

    return norm_pm[:2]

TRACKER_TYPE = 1
NUM_PTS = 6

# cam = cv2.VideoCapture("box/frame%5d.jpg")
cam = cv2.VideoCapture("CMPUT 412 Final Challange.mov")
# cv2.namedWindow("test")
ret, frame = cam.read()
print(frame.shape)

fig, ax = plt.subplots() 
ax.imshow(frame) 
ax.axis('off') 
     
plt.title("Image") 
   
pts = plt.ginput(NUM_PTS)

trackers = []

# 4 trackers on edge
if TRACKER_TYPE == 1:
    for i in range(NUM_PTS):
        trackers.append(LKTracker.Tracker(frame, get_bbox(pts[i])))
    # tracker1 = LKTracker.Tracker(frame, get_bbox(pts[0]))
    # tracker2 = LKTracker.Tracker(frame, get_bbox(pts[1]))
    # tracker3 = LKTracker.Tracker(frame, get_bbox(pts[2]))
    # tracker4 = LKTracker.Tracker(frame, get_bbox(pts[3]))
# one tracker for whole obj
else:
    pts = [pts[0],
        pts[1],
        pts[2],
        pts[3],]
    pts = np.array(pts).T
    print(pts)


    lk.initTracker(frame, pts)

h, w, _ = frame.shape


while True:
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame,5)
    # 4 trackers at corners
    if TRACKER_TYPE == 1:
        corners = np.zeros([2, NUM_PTS])
        for i in range(NUM_PTS):
            tracker = trackers[i]
            c = tracker.updateTracker(frame)
            corners[:, i] = get_center(c)

        # c1 = tracker1.updateTracker(frame)
        # corners[:, 0] = get_center(c1)
        # c2 = tracker2.updateTracker(frame)
        # corners[:, 1] = get_center(c2)
        # c3 = tracker3.updateTracker(frame)
        # corners[:, 2] = get_center(c3)
        # c4 = tracker4.updateTracker(frame)
        # corners[:, 3] = get_center(c4)

        # tracker1.drawRegion(cframe, c1, [0, 255, 0], 2)
    # 1 trackers for whole obj
    else:
        corners = lk.updateTracker(frame)
        lk.drawRegion(cframe, corners, [0, 0, 255], 2)

    pts = np.concatenate((corners.T, np.ones([NUM_PTS, 1])), axis=1) 

    for i in range(NUM_PTS):
        cframe = cv2.circle(cframe, (int(pts[i][0]), int(pts[i][1])), 10, [255, 255, 0], -1)

    # cframe = cv2.circle(cframe, (int(pts[0][0]), int(pts[0][1])), 10, [255, 255, 0], -1)
    # cframe = cv2.circle(cframe, (int(pts[1][0]), int(pts[1][1])), 10, [255, 255, 0], -1)
    # cframe = cv2.circle(cframe, (int(pts[2][0]), int(pts[2][1])), 10, [255, 255, 0], -1)
    # cframe = cv2.circle(cframe, (int(pts[3][0]), int(pts[3][1])), 10, [255, 255, 0], -1)
    cframe = cv2.resize(cframe, (1080, 1080))

    cv2.imshow("test", cframe.astype(np.uint8))

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break

    # time.sleep(0.1)