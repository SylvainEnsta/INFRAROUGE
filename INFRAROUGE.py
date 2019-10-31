import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def coordinates_planck(file_min, file_max, alpha, beta):
    mean_nuc = []
    temperature = []
    zeroabsolu = 273.15

    for nbfile in range(file_min, file_max, 2):
        img_raw = np.loadtxt("data/temp_bb_" + str(nbfile) + "C.dat")

        img_nuc = alpha * img_raw + beta
        mean_nuc.append(img_nuc.mean())
        temperature.append(nbfile + zeroabsolu)

    return mean_nuc, temperature


def planck_equation(mean_nuc, r, b, o):
    return b / np.log(r / (mean_nuc - o) + 1)


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
plt.title("T°1 : 18C")
plt.imshow(img_temp1, vmax=5000)
plt.colorbar()

# Corps Noir 40°C
img_temp2 = np.loadtxt("data/temp_bb_40C.dat")
plt.subplot(333)
plt.title("T°2: 40C")
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
plt.title("NUC - RAW")
plt.imshow(img_nuc, vmin=4000, vmax=8000)
plt.colorbar()

# img_temp1_nuc = alpha * img_temp1 + beta
# plt.subplot(337)
# plt.title("NUC - T°1")
# plt.imshow(img_temp1_nuc, vmin=4000, vmax=5000)
# plt.colorbar()
#
# img_temp2_nuc = alpha * img_temp2 + beta
# plt.subplot(338)
# plt.title("NUC - T°2")
# plt.imshow(img_temp2_nuc, vmin=7000,  vmax=8000)
# plt.colorbar()


# QUESTION 4 - ERRORS
errors = np.where((alpha < 0.75) | (alpha > 1.25) | (beta < -5000) | (beta > 5000))
print(errors)

alpha[errors] = 1
beta[errors] = 0

# QUESTION 5 - CORRECTION NON FINALISEE
img_nuc_correction = cv2.blur(img_nuc, (6, 6))
plt.subplot(338)
plt.title("CORRECTION")
plt.imshow(img_nuc_correction, vmin=4000, vmax=8000)
plt.colorbar()

# QUESTION 6 - PLANCK
xdata, ydata = coordinates_planck(16, 42, alpha, beta)
curve_1, curve_2 = scipy.optimize.curve_fit(planck_equation, xdata, ydata, [1e7, 3000, 1000])
r, b, o = curve_1
plt.subplot(339)
# plt.figure(2)
plt.title("Régle de Conversion Dl → T")
plt.plot(xdata, ydata, 'o', label="Value Data")
plt.plot(xdata, planck_equation(xdata, r, b, o), label="Theorical Planck Equation")
plt.legend()

# IMGSHOW
plt.waitforbuttonpress()


# QUESTION 4 - ERRORS

errors = np.where((alpha < 0.75) | (alpha > 1.25) | (beta < -5000) | (beta > 5000))
print(errors)

alpha[errors] = 1
beta[errors] = 0

img_nuc[errors] =0

plt.figure()
plt.title("Affichage des pixels défaillants en noir")
plt.imshow(img_nuc, vmin=4000, vmax=8000)
plt.colorbar()
plt.waitforbuttonpress()

img_nuc[errors] = img_nuc[errors[0]+1, errors[1]]

plt.figure()
plt.title("Image corrigée")
plt.imshow(img_nuc, vmin=4000, vmax=8000)
plt.colorbar()
# CORRECTION NON FINALISEE

# IMGSHOW
plt.waitforbuttonpress()
