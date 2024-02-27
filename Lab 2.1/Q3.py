import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import sys
import argparse

NUM_PTS = 4
NORM = True
L1 = True

def DLT():
    
    img1 = plt.imread("./key1.jpg")
    img2 = plt.imread("./key2.jpg")

    h, w, _ = img1.shape
    T = np.linalg.inv(np.array([[w + h, 0, w/2],
                [0, w + h, h/2],
                [0, 0, 1]]))



    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1)
    ax[0].axis('off') 
    ax[1].imshow(img2)
    ax[1].axis('off') 

    fig.set_figwidth(10)
    fig.set_figheight(7)



    pts = np.concatenate((np.array(plt.ginput(NUM_PTS * 2)), np.ones([NUM_PTS * 2, 1])), axis=1)

    if NORM:
        for i in range(len(pts)):
            pts[i] = np.matmul(T, pts[i])

    x = pts[:NUM_PTS]

    A = np.zeros([NUM_PTS*2, 9])

    j = 0
    for i in range(0, NUM_PTS):
        pt = pts[i]
        pt_prime = pts[i+NUM_PTS]
        A[j, 3:] = [-pt[0], -pt[1], -pt[2], pt_prime[1] * pt[0], pt_prime[1] * pt[1], pt_prime[1] * pt[2]]
        A[j+1, :] = [pt[0], pt[1], pt[2], 0, 0, 0, -pt_prime[0] * pt[0], -pt_prime[0] * pt[1], -pt_prime[0] * pt[2]]
        j += 2

    U, D, V = np.linalg.svd(A)

    h = V[-1, :]
    # print(h)

    if L1:
        nabla = np.linalg.norm(A * h, ord=1, axis=0)
        prev_norm = 1000000000
        norm = np.linalg.norm(nabla)

        H = h.reshape([3, 3])

        while abs(prev_norm - norm)  > 0.001:

            diag = np.zeros([NUM_PTS * 2, NUM_PTS * 2])
            np.fill_diagonal(diag, 1/np.sqrt(nabla))

            U, D, V = np.linalg.svd(np.matmul(diag, A))

            h = V[-1, :]
            # print(h)
            nabla = np.linalg.norm(A * h, ord=1, axis=0)
            # print(np.linalg.norm(nabla), np.mean(nabla))
            prev_norm = norm
            norm = np.linalg.norm(nabla)

    H = h.reshape([3, 3])
    if NORM:
        H = np.matmul(np.matmul(np.linalg.inv(T), H), T)

    new_img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR)

    cv2.imshow("Iter", new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-n", "--norm", help="Add the normalization procedure", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-l", "--l1", help="Use the L1 norm", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-p", "--points", help="Set the number of point correspondences", default=4, type=int)

    args=parser.parse_args()
    NUM_PTS = args.points
    NORM = args.norm
    L1 = args.l1

    DLT() 

    

    # default args
    # args = [False, False, 4] # Norm, L1, # of pts
    # for i in range(1, len(sys.argv)):
    #     args[i-1] = 
