import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def pics_information(img):
    return img

img = np.loadtxt("data/row_image.raw")
print(img.size)
plt.figure(1)
plt.imshow(img, vmin=4000, vmax=8000)
plt.colorbar()


# IMGSHOW
plt.waitforbuttonpress()
