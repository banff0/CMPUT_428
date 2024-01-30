import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt 
from math import ceil

data_loc = "./data/0_Easy/"
data_suffix = "box"

# cam = cv2.VideoCapture(dat
# a_loc + f"{data_suffix}/frame%05d.jpg")
cam = cv2.VideoCapture("armD32im1/%d.png")
# cam = cv2.VideoCapture("Flowers/%d.png")
# cam = cv2.VideoCapture("./data/0_Easy/book3/frame%05d.jpg")

# cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
prev_img = None
VERBOSE = False
VIDEO = False
TYPE = 1

blockSize = 8 # 4x4, 8x8, 16x16
thresh = 2.50
scale = 2
vid_length = 7 # seconds


def draw_arrow(img, block_x, block_y, vec):
    center_x = int(block_x[1] - blockSize / 2)
    center_y = int(block_y[1] - blockSize / 2)
    start = (center_x, center_y)

    end = (start[0] + vec[0], start[1] + vec[1])

    cv2.arrowedLine(img, start, end, [0, 0, 255], 1)

def get_bbox(label):
    label = label.split()
    
    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    j = 0
    for i in range(1, len(label), 2):
        pts[j, :] = [int(float(label[i])), int(float(label[i+1]))]
        j +=1 


    # start = [int(float(label[1])), int(float(label[2]))]

    # end = [int(float(label[-4])), int(float(label[-3]))]
    return pts

def get_region(img, bbox):
    bbox = np.clip(bbox, 0, 1080)
    return img[ceil(bbox[1]):ceil(bbox[1]+bbox[3]),  ceil(bbox[0]):ceil(bbox[0]+bbox[2])]

ret, frame = cam.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_img = frame

# get the gradients of the image in both x, y=mp4
if VERBOSE: print(np.gradient(np.float32(frame)))
dy, dx = np.gradient(np.float32(frame))
h, w = dx.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out_vid = cv2.VideoWriter(f'output{TYPE}.avi', fourcc, 20.0, (640,  480))
out_vid = cv2.VideoWriter(f'Optic Flow.avi', fourcc, 20.0, (640,  480))
if VERBOSE: print(dx.shape, dy.shape)

# labels = open(data_loc + f"/{data_suffix}.txt").readlines()
# labels.pop(0)
# print(labels[0], get_bbox(labels[0]))

epsilon = 0.25
max_iter = 200
bbox = np.array(cv2.selectROI("select the region for tracking", frame, fromCenter=False))
cv2. destroyAllWindows()
template = get_region(frame, bbox)
# print(template.shape)
# print(bbox)
p = np.zeros_like(bbox, dtype=np.float32)

cv2.imshow("test", frame.astype(np.uint8))

while True:
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame,5)

    u = np.array([99999, 99999, 999999, 999999])
    i = 0
    while abs(np.linalg.norm(u)) > epsilon and i < max_iter:
        i += 1
        pred = get_region(frame, bbox + p)
        if template.shape != pred.shape:
            p = np.zeros_like(bbox)
            break
        # print(pred.shape)
        # diff = np.float32(template) - np.float32(pred)
        diff = np.float32((template - np.mean(template)) / np.std(template)) - np.float32((pred - np.mean(pred)) / np.std(pred))
        dy, dx = np.gradient(pred)
        dy, dx = np.gradient((pred - np.mean(pred)) / np.std(pred))
        dI = np.array([dx.flatten(), dy.flatten()])

        dI = dI.T
        diff = diff.flatten().T
        # print(dI.shape, diff.shape)
        u = np.linalg.lstsq(dI, diff)[0] *0.75
        # print(u)
        # print(p.shape, u.shape)

        # print("BEFORE", p)
        # print(p[0] + u[1])
        p[0] = p[0] + u[1]
        p[1] = p[1] + u[0]
        # print("AFTER", p)
        # p[2 :] += u.astype(np.int32)
        # print(np.linalg.norm(u))
    
    bbox = bbox + p
    # template = get_region(frame, bbox)
    p = np.zeros_like(bbox)

    cframe = cv2.rectangle(cframe, [int(bbox[0]), int(bbox[1])], [int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])], [0, 0, 255], 2)

    cv2.imshow("test", cframe.astype(np.uint8))
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break

'''
1. Find a template from frame 0
2. init p = 0
3. init bbox pos to be the pos where the template came from


for each frame of video:
    1. I = next frame 
    2. init u = [99, 99]
    while norm(u) > epsilon:
        1. get the temporal diff. I[x+p] - T
            Make I[x+p] a meshgrid - Later
        2. solve I \ Mu, this gives us u
        3. update p += u
    3. move the bbox from prev pos to bbox_pos + p
    4. Update T = I[x+p]
    5. reinit p = 0

'''