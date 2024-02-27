import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import lk
import LKTracker

TRACKER_TYPE = 1

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

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
ret, frame = cam.read()
print(frame.shape)

fig, ax = plt.subplots() 
ax.imshow(frame) 
ax.axis('off') 
     
plt.title("Image") 
   
pts = plt.ginput(4)

# 4 trackers on edge
if TRACKER_TYPE == 1:
    tracker1 = LKTracker.Tracker(frame, get_bbox(pts[0]))
    tracker2 = LKTracker.Tracker(frame, get_bbox(pts[1]))
    tracker3 = LKTracker.Tracker(frame, get_bbox(pts[2]))
    tracker4 = LKTracker.Tracker(frame, get_bbox(pts[3]))
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
        corners = np.zeros([2, 4])

        c1 = tracker1.updateTracker(frame)
        corners[:, 0] = get_center(c1)
        c2 = tracker2.updateTracker(frame)
        corners[:, 1] = get_center(c2)
        c3 = tracker3.updateTracker(frame)
        corners[:, 2] = get_center(c3)
        c4 = tracker4.updateTracker(frame)
        corners[:, 3] = get_center(c4)

        # tracker1.drawRegion(cframe, c1, [0, 255, 0], 2)
    # 1 trackers for whole obj
    else:
        corners = lk.updateTracker(frame)
        # lk.drawRegion(cframe, corners, [0, 0, 255], 2)
    
    pts = np.concatenate((corners.T, np.ones([4, 1])), axis=1) 


    l1 = np.cross(pts[0], pts[1])
    l2 = np.cross(pts[2], pts[3])

    l3 = np.cross(pts[0], pts[2])
    l4 = np.cross(pts[1], pts[3])

    pm = np.cross(l1, l2)

    norm_pm = (pm / pm[2]).astype(np.int32)

    norm_pm = norm_pm.astype(np.int32)

    vanish = np.cross(l3, l4)

    norm_vanish = (vanish / vanish[2]).astype(np.int32)

    lm = np.cross(norm_pm, norm_vanish)
    print(lm, (lm / lm[2]))

    A, B, C = (lm / lm[2])
    # print(A, B, C)
    # print([0, -C//B], [w, (-(A*w)-C) // B])
    # cframe = cv2.line(cframe, (int(C//A), 0), (int((-(B*h)-C) // B), h), [0, 0, 0], 3)
    cframe = cv2.line(cframe, (norm_pm[0], norm_pm[1]), (norm_vanish[0], norm_vanish[1]), [0, 0, 0], 2) 
    # cframe = cv2.line(cframe, (norm_pm[0], norm_pm[1]), (norm_vanish[0], norm_vanish[1]), [0, 0, 0], 2) 

    cframe = cv2.circle(cframe, (int(norm_pm[0]), int(norm_pm[1])), 10, [0, 0, 255], -1)
    cframe = cv2.circle(cframe, (int(norm_vanish[0]), int(norm_vanish[1])), 10, [0, 0, 255], -1)
    cframe = cv2.circle(cframe, (int(pts[0][0]), int(pts[0][1])), 10, [255, 255, 0], -1)
    cframe = cv2.circle(cframe, (int(pts[1][0]), int(pts[1][1])), 10, [255, 255, 0], -1)
    cframe = cv2.circle(cframe, (int(pts[2][0]), int(pts[2][1])), 10, [255, 255, 0], -1)
    cframe = cv2.circle(cframe, (int(pts[3][0]), int(pts[3][1])), 10, [255, 255, 0], -1)
    # cframe = cv2.rectangle(cframe, p1, p2, (255,0,0), 2, 1)



    cv2.imshow("test", cframe.astype(np.uint8))
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
