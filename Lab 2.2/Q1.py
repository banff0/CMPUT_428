from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from math import degrees as d


def mk_line(xflag, yflag, zflag):
    funcs = ["np.zeros((1, 100))", "np.ones((1, 100))", "np.random.rand(1, 100)"]
    x = eval(funcs[xflag])
    y = eval(funcs[yflag])
    z = eval(funcs[zflag])
    return np.array([x, y, z, np.ones((1, 100))])

def draw_shapes(H, trans = True):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # line
    x = np.random.rand(1, 100)
    y = np.ones((1, 100))
    z = np.ones((1, 100))

    line = np.array([x, y, z, np.ones((1, 100))])
    if trans:
        for i in range(100):
            line[:3, :, i] = np.matmul(H, line[:, :, i])
            line[:3, :, i] = line[:3, :, i] / line[2, :, i]
        ax = fig.add_subplot(1, 3, 1)
        ax.set_xlim([-3.5, 2.5])
        ax.set_ylim([-3.5, 2.5])

        ax.scatter(line[0], line[1], cmap='Greens')

    else:
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter3D(x, y, z, cmap='Greens')

    # Circle
    r = np.random.rand(1, 100) * 3.14 * 2
    x = np.sin(r)
    y = np.cos(r)
    z = np.ones((1, 100))

    circle = np.array([x, y, z, np.ones((1, 100))])
    if trans:
        for i in range(100):
            circle[:3, :, i] = np.matmul(H, circle[:, :, i])
            circle[:3, :, i] = circle[:3, :, i] / circle[2, :, i]
        ax = fig.add_subplot(1, 3, 2)
        ax.set_xlim([-3.5, 2.5])
        ax.set_ylim([-3.5, 2.5])
        ax.scatter(circle[0], circle[1], cmap='Greens')
    else:
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter3D(x, y, z, cmap='Blues')

    # Cube
    ax = fig.add_subplot(1, 3, 3, projection='3d')
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
        for j in range(len(cube)):
            for i in range(100):
                cube[j][:3, :, i] = np.matmul(H, cube[j][:, :, i])
                cube[j][:3, :, i] = cube[j][:3, :, i] / cube[j][2, :, i]
        ax = fig.add_subplot(1, 3, 3)
        ax.set_xlim([-3.5, 2.5])
        ax.set_ylim([-3.5, 2.5])
        for line in cube:
            ax.scatter(line[0], line[1], cmap='Greens')
    else:
        for line in cube:
            ax.scatter3D(line[0], line[1], line[2], cmap='Blues')

    plt.show()



draw_shapes(None, trans=False)

H = np.array([[1, 0, 0 ,0],
              [0, 1, 0, 0], 
              [0, 0, 0, 1]])
draw_shapes(H)

H = np.array([[1, 0, 0 ,0],
              [0, 1, 0, 0], 
              [0, 0, 0, 1/2]])
draw_shapes(H)

H = np.array([[cos(d(45)), -sin(d(45)), 0 ,0],
              [sin(d(45)), cos(d(45)), 0, 0], 
              [0, 0, 0, 1/2]])
draw_shapes(H)



