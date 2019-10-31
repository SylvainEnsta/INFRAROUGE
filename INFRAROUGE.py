import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def pics_information(img):
    return img

# QUESTION 1
img_raw = np.loadtxt("data/row_image.dat")
print(img_raw.size)
plt.figure(1)
plt.subplot(331)
plt.title("RAW")
plt.imshow(img_raw, vmin=4000, vmax=8000)
plt.colorbar()

# QUESTION 2 - GAIN / OFFSET
# Corps Noir 18°C
img_temp1 = np.loadtxt("data/temp_bb_18C.dat")
plt.subplot(332)
plt.title("Ref : 18C")
plt.imshow(img_temp1, vmax=5000)
plt.colorbar()

# Corps Noir 40°C
img_temp2 = np.loadtxt("data/temp_bb_40C.dat")
plt.subplot(333)
plt.title("Ref: 40C")
plt.imshow(img_temp1, vmin=5000)
plt.colorbar()

# gain_matrice
alpha = (img_temp2.mean() - img_temp1.mean()) / (img_temp2 - img_temp1)
plt.subplot(334)
plt.title("Gain")
plt.imshow(alpha, vmin=0.75, vmax=1.25)
plt.colorbar()

# offset_matrice
beta = img_temp1.mean() - (alpha * img_temp1)
plt.subplot(335)
plt.title("Offset")
plt.imshow(beta, vmin=-200, vmax=200)
plt.colorbar()

# QUESTION 3 - NUC
img_nuc = alpha * img_raw + beta
plt.subplot(337)
plt.title("NUC")
plt.imshow(img_nuc, vmin=4000, vmax=8000)
plt.colorbar()



# IMGSHOW
plt.waitforbuttonpress()
