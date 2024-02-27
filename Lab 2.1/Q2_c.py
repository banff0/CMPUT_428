import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import lk


cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
ret, frame = cam.read()
print(frame.shape)

fig, ax = plt.subplots() 
ax.imshow(frame) 
ax.axis('off') 
     
plt.title("Image") 
   
pts = np.array(plt.ginput(8))

const_pts = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)

print(const_pts)

hconst_pts = np.concatenate((const_pts, np.ones([4, 1])), axis=1)

print(hconst_pts)

cl1 = np.cross(hconst_pts[0], hconst_pts[1])
cl2 = np.cross(hconst_pts[2], hconst_pts[3])

pts = np.array([pts[0], pts[1], pts[2], pts[3]])


lk.initTracker(frame, pts.T)

h, w, _ = frame.shape

# bbox = cv2.selectROI(frame, False)
# ret = tracker.init(frame, bbox)

while True:
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.medianBlur(frame,5)

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

    A, B, C = (lm / lm[2])
    # print(A, B, C)
    # cframe = cv2.line(cframe, (int(C//A), 0), (int((-(B*h)-C) // B), h), [0, 0, 0], 3)
    cframe = cv2.line(cframe, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), [255, 255, 255], 5) 
    cframe = cv2.line(cframe, (int(pts[2][0]), int(pts[2][1])), (int(pts[3][0]), int(pts[3][1])), [255, 255, 255], 5) 
    cframe = cv2.line(cframe, (const_pts[0]), (const_pts[1]), [0, 0, 255], 5) 
    cframe = cv2.line(cframe, (const_pts[2]), (const_pts[3]), [0, 0, 255], 5) 

    p_err = np.cross(np.cross(l3, l4), np.cross(cl1, cl2))
    p_err = p_err / p_err[2]
    
    # print(pts[0].shape, np.cross(pts[0], pts[1]).shape, hconst_pts[0].shape, np.cross(hconst_pts[0], hconst_pts[1]).shape)
    norm_cl1 = cl1 / cl1[2]
    l_err = np.matmul(pts[0], norm_cl1) + np.matmul(pts[1], norm_cl1)

    # l_err =l_err / l_err[2]


    print(p_err)
    print(l_err)

    cv2.putText(cframe, f"Lerr: {l_err}", (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
    cv2.putText(cframe, f"Perr: {p_err[0]}, {p_err[1]}", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))


    cv2.imshow("test", cframe.astype(np.uint8))
    # sleep(0.1)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
