import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from math import radians as d

# size x size cube, default 5x5
def mk_line(xflag, yflag, zflag, size=5):
    funcs = ["np.zeros((1, 100))", f"np.ones((1, 100))*{size}", f"np.linspace(0, {size}, num=100).reshape([1, 100])"]
    x = eval(funcs[xflag])
    y = eval(funcs[yflag])
    z = eval(funcs[zflag])
    return np.array([x, y, z, np.ones((1, 100))])

def get_pts(line, seen_pts, pts):
    p1 = tuple(map(tuple, (line[:3, :, 0])))
    if not p1 in seen_pts:
        seen_pts.add(p1)
        for i in range(3):
            if i == 4:
                pts[i].append(line[:3, :, 1][i+1])
            else:
                pts[i].append(line[:3, :, 0][i])
    p2 = tuple(map(tuple, (line[:3, :, 99])))
    if not p2 in seen_pts:
        seen_pts.add(p2)
        for i in range(3):
            pts[i].append(line[:3, :, 99][i])



def draw_cube(H, trans=True):
    pts = [0, 0, 0]

    cube = []
    cube.append(mk_line(2, 1, 1))
    cube.append(mk_line(2, 0, 1))
    cube.append(mk_line(2, 0, 0))
    cube.append(mk_line(2, 1, 0))

    cube.append(mk_line(1, 2, 1))
    cube.append(mk_line(0, 2, 1))
    cube.append(mk_line(0, 2, 0))
    cube.append(mk_line(1, 2, 0))

    cube.append(mk_line(1, 1, 2))
    cube.append(mk_line(0, 1, 2))
    cube.append(mk_line(0, 0, 2))
    cube.append(mk_line(1, 0, 2))

    if trans:
        seen = set()
        pts = [[], [], []]
        for j in range(len(cube)):
            for i in range(100):
                cube[j][:3, :, i] = np.matmul(H, cube[j][:, :, i])
                cube[j][:3, :, i] = cube[j][:3, :, i] / cube[j][2, :, i]
            get_pts(cube[j], seen, pts)
        
        print("PONTS", pts)

    return pts

draw_cube(None, trans=False)

f = 2715 # from Q2_a
dist = 1800 #cm
b = 15 #cm

D = np.zeros([9, 8])
X = np.zeros([9, 8])
Y = np.zeros([9, 8])

for j in range(10):
    H = np.array([[cos(d(46+j)), 0, sin(d(46+j)), b*j],
                  [0, 1, 0, 0], 
                  [0, 0, 0, f/dist]])
 
    if j > 0:
      xr, yr= xl, yl

    xl, yl, _ = draw_cube(H)

    x = []
    y = []
    z = []

    

    if j > 0:
        for i in range(len(xr)):
            X[j-1, i] = xl[i]
            Y[j-1, i] = yl[i]
            D[j-1, i] = (xl[i] - xr[i])[0]

# print(X.shape, X)
df = np.ones([9, 1]) * f*b
print(df, df.shape)
print(D, D.shape)

z = np.linalg.lstsq(D, df, rcond=None)[0]

for i in range(9):
    for j in range(8):
        X[i, j] *= z[j]
        Y[i, j] *= z[j]
x_3D = np.linalg.lstsq(X, f*np.ones([9, 1]), rcond=None)[0]
y_3D = np.linalg.lstsq(Y, f*np.ones([9, 1]), rcond=None)[0]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection='3d')

ax.scatter3D(x_3D, y_3D, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')



plt.show()

'''
To do the last part you would save the corners of the tracker every n (eg. 10) frames
these points would then become the u, v that you use to detect the depth. From these coordinates you could
use the difference in x between frames to calculate the disparity and then follow the above algorithm to solve 
for X, Y, and Z
'''