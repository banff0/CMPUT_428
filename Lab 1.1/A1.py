import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt 

# cam = cv2.VideoCapture("armD32im1/%d.png")
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
prev_img = None
VERBOSE = False
VIDEO = True
TYPE = 1

blockSize = 16 # 4x4, 8x8, 16x16
thresh = 2.50
scale = 2
vid_length = 7 # seconds


def draw_arrow(img, block_x, block_y, vec):
    center_x = int(block_x[1] - blockSize / 2)
    center_y = int(block_y[1] - blockSize / 2)
    start = (center_x, center_y)

    end = (start[0] + vec[0], start[1] + vec[1])

    cv2.arrowedLine(img, start, end, [0, 0, 255], 2)


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

while True:
    # sleep(0.25)
    # get the image from the camera
    ret, frame = cam.read()
    cframe = frame.copy()
    if not ret:
        print("failed to grab frame")
        break
    # convert framew to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.medianBlur(frame,5)
    

    diff = np.float32(prev_img) - np.float32(frame)

    # get the tiles
    X = np.arange(0, w, blockSize, dtype=int) + int(blockSize / 2)
    Y = np.arange(0, h, blockSize, dtype=int) + int(blockSize / 2)
    
    # iter through each tile to calculate the movement
    if VERBOSE: print(diff.shape, X.shape, Y.shape)
    if TYPE == 1:
        for i in range(1, len(X)):
            for j in range(1, len(Y)):
                block_diff = diff[Y[j-1] : Y[j], X[i-1] : X[i]]
                if np.mean(block_diff) > thresh or np.mean(block_diff) < -thresh:
                    block_diff = block_diff.flatten()
                    dI = np.array([dx[Y[j-1] : Y[j], X[i-1] : X[i]].flatten(), dy[Y[j-1] : Y[j], X[i-1] : X[i]].flatten()])

                    dI = dI.T
                    block_diff = block_diff.T
                    
                    dir = np.linalg.lstsq(dI, block_diff)[0]
                    dir = (dir * scale).astype(np.int32)
                    if VERBOSE: print(dir)
                    draw_arrow(cframe, [X[i-1], X[i]], [Y[j-1], Y[j]], dir)
    # do the thresholding
    else:
        cframe = ((diff > thresh * 2) | (diff < -thresh * 2)) * 255
        cframe = cv2.cvtColor(cframe.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # print(cframe.shape)

    prev_img = frame
    # get the gradients of the image in both x, y
    dy, dx = np.gradient(frame)

    img_counter += 1
    out_vid.write(cframe.astype(np.uint8))
    cv2.imshow("test", cframe.astype(np.uint8))

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break


cam.release()
out_vid.release()
cv2.destroyAllWindows()
