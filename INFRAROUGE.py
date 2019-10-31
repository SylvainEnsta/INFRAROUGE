import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def pics_information(img):

img = np.loadtxt("data/row_image.raw")
print(img.size)
plt.figure(1)
plt.imshow(img, vmin=6000, vmax=7000)
plt.colorbar()

# IMGSHOW
plt.waitforbuttonpress()

