import matplotlib.pyplot as plt 
import numpy as np 
import cv2


num_pts = 4

img = plt.imread("./feild.jpg")

print(img.shape)


fig, ax = plt.subplots() 
ax.imshow(img) 
ax.axis('off') 
     
plt.title("Image") 
   
pts = np.array(plt.ginput(num_pts))


pts = np.concatenate((pts, np.ones([num_pts, 1])), axis=1) 

print(pts)

l1 = np.cross(pts[0], pts[1])
l2 = np.cross(pts[2], pts[3])

l3 = np.cross(pts[0], pts[2])
l4 = np.cross(pts[1], pts[3])

print(l1)
print(l2)

pm = np.cross(l1, l2)

norm_pm = pm / pm[2]

norm_pm = norm_pm.astype(np.int32)

print(pm)

   
img = cv2.circle(img, (int(norm_pm[0]), int(norm_pm[1])), 10, [0, 0, 255], -1)
img = cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), 10, [255, 255, 0], -1)
img = cv2.circle(img, (int(pts[1][0]), int(pts[1][1])), 10, [255, 255, 0], -1)
img = cv2.circle(img, (int(pts[2][0]), int(pts[2][1])), 10, [255, 255, 0], -1)
img = cv2.circle(img, (int(pts[3][0]), int(pts[3][1])), 10, [255, 255, 0], -1)


vanish = np.cross(l3, l4)

norm_vanish = (vanish / vanish[2]).astype(np.int32)

lm = np.cross(vanish, pm)

norm_lm = (lm / lm[2]).astype(np.int32)

print(norm_pm, norm_lm)


print(norm_lm[0] * norm_pm[0] + norm_lm)

img = cv2.line(img, (norm_pm[0], norm_pm[1]), (norm_vanish[0], norm_vanish[1]), [0, 0, 0], 2) 

cv2.imshow("window_name", img)

  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows()

