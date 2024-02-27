import matplotlib.pyplot as plt 
import numpy as np 
import cv2


NUM_PTS = 4

img1 = plt.imread("./key1.jpg")
img2 = plt.imread("./key2.jpg")


fig, ax = plt.subplots(1,2)
ax[0].imshow(img1)
ax[0].axis('off') 
ax[1].imshow(img2)
ax[1].axis('off') 

fig.set_figwidth(10)
fig.set_figheight(7)



pts = np.array(plt.ginput(NUM_PTS * 2))

# print(pts)
x = np.concatenate((pts[:NUM_PTS], np.ones([NUM_PTS, 1])), axis=1)
# print(x)


A = np.zeros([NUM_PTS*2, 9])

j = 0
for i in range(0, NUM_PTS):
    pt = pts[i]
    pt_prime = pts[i+NUM_PTS]
    A[j, 3:] = [-pt[0], -pt[1], -1, pt_prime[1] * pt[0], pt_prime[1] * pt[1], pt_prime[1]]
    A[j+1, :] = [pt[0], pt[1], 1, 0, 0, 0, -pt_prime[0] * pt[0], -pt_prime[0] * pt[1], -pt_prime[0]]
    j += 2

U, D, V = np.linalg.svd(A)

# h = V[:, -1]
h = V[-1, :]
H = h.reshape([3, 3])

# print(H)
# print(H * x)

new_img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR)

cv2.imshow("TITLE", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# print(V[:, -1])
# print(V[-1, :])

