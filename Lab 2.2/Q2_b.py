from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from math import radians as d

def get_pts(img_name, num_pts):
    try:
        img = plt.imread(img_name)
        fig, ax = plt.subplots() 
        ax.imshow(img) 
        ax.axis('off') 
            
        plt.title("Image") 
        pts = np.array(plt.ginput(num_pts))
        x = pts[:, 0]
        y = pts[:, 1]
        
        return x, y
    finally:
        plt.close()


# z = f*b / (xl - xr) = f*b/d



f = 2715 # from Q2_a
b = 15 #cm

xl, yl = get_pts("left.jpg", 6)
xr, yr = get_pts("right.jpg", 6)

# print(xl, xr)
# print(yl)
x = []
y = []
z = []

for i in range(len(xl)):
    print(xl[i] ,xr[i], (xl[i] - xr[i]), "------------------")
    z.append((f*b) / (xl[i] - xr[i]))
    x.append(xl[i] * (z[i] / f))
    y.append(yl[i] * (z[i] / f))

    print([x[i], y[i], z[i]])

# z = (f*b) / (xl - xr)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection='3d')

ax.scatter3D(x, y, z)

plt.show()

# for i in range(len(x)):
#     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]], zs=[VecStart_z[i],VecEnd_z[i]])


