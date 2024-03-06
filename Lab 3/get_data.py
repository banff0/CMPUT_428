import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat


def get_imgs(file):
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