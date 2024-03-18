import numpy as np
from get_data import get_mat_data, get_vid_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def center_data(pts):
    pts -= np.mean(pts, axis=0)
    return pts

def construct_W(imgs):
    # imgs = [m, n, 2]
    # print("imgs shape", imgs.shape)
    # number of frames
    m = imgs.shape[0]
    # number of points
    n = imgs.shape[1]

    W = np.zeros((2*m, n))
    for i in range(len(imgs)):
        pts = imgs[i]
        pts = center_data(pts)
        W[i*2] = pts[:, 0]
        W[i*2+1] = pts[:, 1]

    return W

def SfM(imgs):
    W = construct_W(imgs)
    # print("W", W.shape)
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    Vt = Vt[:3, :]
    X = Vt.T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

# SfM(get_mat_data("affrec1"))
# SfM(get_mat_data("affrec3"))
# SfM(get_mat_data("HouseTallBasler64"))
SfM(get_vid_data("car_test.mp4", 6, show=True))
# SfM(get_vid_data("test_box.mp4", 8, show=True))


