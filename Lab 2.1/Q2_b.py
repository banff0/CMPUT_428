import matplotlib.pyplot as plt 
import numpy as np 
import cv2


tracker = cv2.TrackerKCF_create() 

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
ret, frame = cam.read()

bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)

while True:
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame,5)

    ret, bbox = tracker.update(frame)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cframe = cv2.rectangle(cframe, p1, p2, (255,0,0), 2, 1)

    cv2.imshow("test", cframe.astype(np.uint8))
    # sleep(0.1)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
