import numpy as np
from numpy import sin, cos

pt = np.array([[10, 10, 10, 1]]).T

rot = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]])
trans = np.array([[0, 20, 0, 1]])

H = np.concatenate((rot, trans.T), axis=1)

pt = np.matmul(H, pt)
print(f"point after translate origin 20 along positive Y axis: \n{pt}")

theta = np.radians(30)
rot = np.array([[cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1],
                [0, 0, 0]])
trans = np.array([[0, 0, 0, 1]])

H = np.concatenate((rot, trans.T), axis=1)

pt = np.matmul(H, pt)
print(f"point after rotation around Z axis 30 deg: \n{pt}")

theta = np.radians(-10)
rot = np.array([[cos(theta), 0, sin(theta)],
                [0, 1, 0],
                [-sin(theta), 0, cos(theta)],
                [0, 0, 0]])
trans = np.array([[0, 0, 0, 1]])

H = np.concatenate((rot, trans.T), axis=1)

pt = np.matmul(H, pt)
print(f"point after rotation around Y axis -10 deg: \n{pt}")
