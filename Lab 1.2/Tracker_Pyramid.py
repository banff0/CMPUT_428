from matplotlib.image import BboxImage
import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt 
from math import ceil, floor

data_loc = "./data/0_Easy/"
data_suffix = "box"

# cam = cv2.VideoCapture(data_loc + f"{data_suffix}/frame%05d.jpg")
cam = cv2.VideoCapture("armD32im1/%d.png")
# cam = cv2.VideoCapture("Flowers/%d.png")
# cam = cv2.VideoCapture("./data/0_Easy/book3/frame%05d.jpg")

# cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
prev_img = None
VERBOSE = False

thresh = 2.50

# def get_bbox(label):
#     label = label.split()
    
#     pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
#     j = 0
#     for i in range(1, len(label), 2):
#         pts[j, :] = [int(float(label[i])), int(float(label[i+1]))]
#         j +=1 


#     # start = [int(float(label[1])), int(float(label[2]))]

#     # end = [int(float(label[-4])), int(float(label[-3]))]
#     return pts

# https://stackoverflow.com/questions/59093533/how-to-pad-an-array-non-symmetrically-e-g-only-from-one-side
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:
        vector[-pad_width[1]:] = pad_value


def get_region(img, bbox):
    # bbox = np.clip(bbox, 0, 1080)
    ret = img[ceil(bbox[1]):ceil(bbox[1]+bbox[3]),  ceil(bbox[0]):ceil(bbox[0]+bbox[2])]
    i = 0
    while ret.shape[1] < bbox[-2]:
        ret = np.pad(ret, ((0, 0), (0, 1)), pad_with, padder=0)
    return ret

def maintin_shape(img, template, bbox):
    w, h = template.shape
    pw, ph = get_region(img, bbox).shape
    i = 0
    while ph != h or pw != w: 
        # print(pw, w)
        # print(ph, h)
        if pw < w:
            bbox[3] = bbox[3] + 1
            # print("ADD")
        elif pw > w:
            bbox[3] = bbox[3] - 1
            # print("SUBTRACT")
        
        if ph < h:
            bbox[2] = bbox[2] + 1
            # print("PLUS")
        elif ph > h:
            bbox[2] = bbox[2] - 1
            # print("MINUS")
        # print(bbox, w, h, pw, ph)
        pw, ph = get_region(img, bbox).shape
        i += 1
        if i > 10:
            bbox[3] = w
            bbox[2] = h
            # print([int(bbox[1]), int(bbox[1]+bbox[3]),  int(bbox[0]), int(bbox[0]+bbox[2])], bbox, template.shape)
            # print([ceil(bbox[1]), ceil(bbox[1]+bbox[3]),  ceil(bbox[0]), ceil(bbox[0]+bbox[2])], bbox, template.shape)
            break
    return bbox

def pyr_track(img, template, bbox, depth, resize_factor = 2):
    if depth == 0:
        # print(f"DEPTH: {depth}")
        bbox = LK(img, template, bbox)
        return bbox

    # downsample the img, bbox, and template
    bbox = bbox // resize_factor
    rows, cols = map(int, template.shape)
    down_template = cv2.pyrDown(template, dstsize=(cols // resize_factor, rows // resize_factor))
    rows, cols = map(int, img.shape)
    down_img = cv2.pyrDown(img, dstsize=(cols // resize_factor, rows // resize_factor))

    # get the bbox from the downsampled img as a starting point
    bbox = pyr_track(down_img, down_template, bbox, depth - 1) * 2
    # compute the new bbox at this depth
    # print(f"DEPTH: {depth}")
    bbox = LK(img, template, bbox)
    return bbox

def LK(img, template, bbox):
    max_iter = 300
    # init p and u
    p = np.zeros_like(bbox, dtype=np.float32)
    u = np.array([99999999, 99999999, 999999999, 999999999])
    # iter until small update to p
    i = 0
    while abs(np.linalg.norm(u)) > epsilon and i < max_iter:
        i += 1
        pred = get_region(img, bbox + p)
        if template.shape != pred.shape:
            new_box = maintin_shape(img, template, bbox + p)
            pred = get_region(img, new_box)
            # print(template.shape, pred.shape)
            # p = np.zeros_like(BboxImage, dtype=np.float32)
            # print("HERE")
            # break
            

        diff = np.float32((template - np.mean(template)) / np.std(template)) - np.float32((pred - np.mean(pred)) / np.std(pred))
        dy, dx = np.gradient((pred - np.mean(pred)) / np.std(pred))

        # diff = np.float32(template) - np.float32(pred)
        # dy, dx = np.gradient(pred)

        dI = np.array([dx.flatten(), dy.flatten()])

        dI = dI.T
        diff = diff.flatten().T
        u = np.linalg.lstsq(dI, diff)[0] * 0.1
        # print(np.linalg.norm(u))

        p[0] = p[0] + u[1]
        p[1] = p[1] + u[0]

    bbox = maintin_shape(img, template, bbox + p)
    # bbox = bbox + p
    return bbox

ret, frame = cam.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_img = frame

# get the gradients of the image in both x, y=mp4
if VERBOSE: print(np.gradient(np.float32(frame)))
dy, dx = np.gradient(np.float32(frame))
img_h, img_w = dx.shape

epsilon = 0.001
# epsilon = 1e-5
bbox = np.array(cv2.selectROI("select the region for tracking", frame, fromCenter=False))
cv2. destroyAllWindows()
template = get_region(frame, bbox)
print(template.shape)
# print(bbox)
p = np.zeros_like(bbox, dtype=np.float32)

cv2.imshow("test", frame.astype(np.uint8))

frame_num = 0
while True:
    frame_num += 1
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame,5)

    try:
        bbox = pyr_track(frame, template, bbox, depth=3, resize_factor=2)
        # if frame_num > 5:
        #     template = get_region(frame, bbox)
        # break

        
    except ValueError as e:
        # raise e
        # print("ERROR:", e)
        pass

    cframe = cv2.rectangle(cframe, [int(bbox[0]), int(bbox[1])], [int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])], [0, 0, 255], 2)

    cv2.imshow("test", cframe.astype(np.uint8))
    sleep(0.1)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
