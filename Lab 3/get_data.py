import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import LKTracker

def _get_bbox(pt):
    size = 15
    bbox = [[pt[0]-size, pt[1]-size], 
            [pt[0]+size, pt[1]-size],
            [pt[0]+size, pt[1]+size],
            [pt[0]-size, pt[1]+size]]
    print(bbox, pt)
    return np.array(bbox).T

def _get_center(pts):
    pts = np.concatenate((pts.T, np.ones([4, 1])), axis=1) 

    l1 = np.cross(pts[0], pts[2])
    l2 = np.cross(pts[1], pts[3])

    pm = np.cross(l1, l2)

    norm_pm = pm / pm[2]
    norm_pm = norm_pm.astype(np.int32)

    return norm_pm[:2]

def get_mat_data(file):
    data = loadmat(f"data/{file}.mat")
    # n_frames = data["NrFrames"].item()
    W = data["W"]
    n_frames = len(W)//2

    x = W[n_frames:, :]
    y = W[0:n_frames, :]

    imgs = np.zeros([x.shape[0], x.shape[1], 2])
    for i in range(len(x)):
        for j in range(x.shape[1]):
            imgs[i, j] = [x[i, j], y[i, j]]

    return imgs

def get_vid_data(file, num_pts):
    cam = cv2.VideoCapture(file)

    ret, frame = cam.read()
    print(frame.shape)

    fig, ax = plt.subplots() 
    ax.imshow(frame) 
    ax.axis('off') 
        
    plt.title("Image") 
    
    pts = plt.ginput(num_pts)

    trackers = []
    for i in range(num_pts):
        trackers.append(LKTracker.Tracker(frame, _get_bbox(pts[i])))
    
    h, w, _ = frame.shape

    while True:
        ret, frame = cam.read()
        try:
            cframe = frame.copy()
        except AttributeError:
            break
        if not ret:
            print("failed to grab frame")
            break
        frame = cv2.medianBlur(frame,5)
        corners = np.zeros([2, num_pts])
        for i in range(num_pts):
            tracker = trackers[i]
            c = tracker.updateTracker(frame)
            corners[:, i] = _get_center(c)
        pts = np.concatenate((corners.T, np.ones([num_pts, 1])), axis=1) 

        for i in range(num_pts):
            cframe = cv2.circle(cframe, (int(pts[i][0]), int(pts[i][1])), 10, [255, 255, 0], -1)
        cframe = cv2.resize(cframe, (1080, 1080))

        cv2.imshow("test", cframe.astype(np.uint8))

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ASCII:ESC pressed
            print("Escape hit, closing...")
            break

        
        
if __name__ == "__main__":
    # Load pts and imgs
    data = loadmat("data/affrec1.mat")
    # n_frames = data["NrFrames"].item()
    W = data["W"]
    n_frames = len(W)//2

    # imgs = data["mexVims"]
    print(n_frames)
    print("W:", W.shape)

    # Plot
    for frame_i in range(n_frames):
        img = cv2.cvtColor(imgs[:, :, frame_i], cv2.COLOR_BAYER_BG2RGB)
        img = cv2.resize(img, (240, 320))
        
        y = W[frame_i, :]
        x = W[n_frames + frame_i, :]

        plt.imshow(img)
        plt.scatter(x, y)
        plt.pause(0.1)
        plt.clf()