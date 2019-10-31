import cv2
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import curve_fit

img = np.loadtxt("raw_image.dat")
