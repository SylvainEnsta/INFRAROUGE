import cv2
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import curve_fit

img = np.loadtxt("data/row_image.raw")
plot.figure(1)
plot.imshow(img)
plot.colorbar()


