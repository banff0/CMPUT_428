from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from math import radians as d


# img_name = "62f99534-6727-480b-8dfa-62d6fd8ea240.png"
img_name = "focal.jpg"
# img = cv2.imread(img_name) 

x = 15 #cm
z = 30 #cm

img = plt.imread(img_name)

# print(img.shape)


fig, ax = plt.subplots() 
ax.imshow(img) 
ax.axis('off') 
     
plt.title("Image") 
   
pts = np.array(plt.ginput(1))
xl = pts[0][0]
 
f = (xl / x) * z
print(f"focal length: {f}")

