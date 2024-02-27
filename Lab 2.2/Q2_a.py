from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from math import degrees as d

img = cv2.imread('/62f99534-6727-480b-8dfa-62d6fd8ea240.png')

x = 15 #cm
xl = ginput
z = 30 #cm

f = (xl / x) * z

print(z)