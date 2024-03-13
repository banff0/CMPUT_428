import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('stereo.mat')
print(mat.keys())

print()

W = np.array(mat["W"])


num_ims = 9
n = num_ims - 1
m = 16  
f = 100
b = 5 #cm


print(W.shape)

x = W[:num_ims]
y = W[num_ims:]

print(x.shape, y.shape)

D = np.zeros([n, m])
for j in range(n):
    for i in range(1, m):
        xl = x[j, i]
        xr = x[j, i-1]
        if i == 4:
            print(xl ,xr, (xl - xr), "------------------")
        # z.append((f*b) / (xl - xr))
        # x.append(xl[i] * (z[i] / f))
        # y.append(yl[i] * (z[i] / f))

        # print([x[i], y[i], z[i]])
        # print(D[j-1][i])
        D[j-1, i] = (xl - xr)


df = np.ones([n, 1]) * f*b
print(df, df.shape)
print(D, D.shape)

z = np.linalg.lstsq(D, df, rcond=None)[0]

print(z, len(z))
x_3D, y_3D = [], []
for i in range(len(z)):
    x_3D.append((b+x[j, i]) * (z[i] / f))
    y_3D.append(y[j, i] * (z[i] / f))


# print(x.shape, z.shape, x.dot(z).shape, "$$$$$$$$$$$$")
for i in range(0, 9):
    for j in range(16):
        x[i, j] *= z[j]
        y[i, j] *= z[j]
x_3D = np.linalg.lstsq(x, f*np.ones([9, 1]), rcond=None)[0]
y_3D = np.linalg.lstsq(y, f*np.ones([9, 1]), rcond=None)[0]
print(x.shape)
print(y)
print(z)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection='3d')

ax.scatter3D(x_3D, y_3D, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')



plt.show()

