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
            pts[i].append(line[:3, :, 0][i])
    p2 = tuple(map(tuple, (line[:3, :, 99])))
    if not p2 in seen_pts:
        seen_pts.add(p2)
        for i in range(3):
            pts[i].append(line[:3, :, 99][i])



def draw_cube(H, trans=True):
    pts = [0, 0, 0]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Cube
    ax = fig.add_subplot(projection='3d')
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
            # vertex.add(tuple(map(tuple, (cube[j][:3, :, 0]))))
            # vertex.add(tuple(map(tuple, (cube[j][:3, :, 99]))))
            # print(cube[j][:3, :, 0])
            # print(cube[j][:3, :, 99])
        # pts = [[cube[j][:3, :, 99][0], cube[j][:3, :, 0][0]], [cube[j][:3, :, 99][1], cube[j][:3, :, 0][1]], [cube[j][:3, :, 99][2], cube[j][:3, :, 0][2]]]
        
        print("PONTS", pts)
        # print(vertex, len(vertex))
        # print(np.array(vertex[0]))
        ax = fig.add_subplot()
        # ax.set_xlim([-3.5, 2.5])
        # ax.set_ylim([-3.5, 2.5])
        for line in cube:
            ax.scatter(line[0], line[1], cmap='Greens')
    else:
        for line in cube:
            ax.scatter3D(line[0], line[1], line[2], cmap='Blues')
    plt.show()

    return pts

draw_cube(None, trans=False)

f = 2715 # from Q2_a
dist = 1800 #cm
b = 15 #cm

for j in range(10):
    H = np.array([[cos(d(46 + j)), 0, sin(d(46 + j)), (b +5)*j],
                  [0, 1, 0, 0], 
                  [0, 0, 0, f/dist]])
 
    if j > 0:
        xr, yr = xl, yl

    xl, yl, _ = draw_cube(H)
    print(xl, yl, "###################")

    x = []
    y = []
    z = []

    if j > 0:
        for i in range(len(xr)):
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
