import cv2
import pickle
import matplotlib.pyplot as plt


filename = "./test_images/test1.jpg"

# Load calibration data
with open("cal_data.p", "rb") as f:
    cal_data = pickle.load(f)

mtx, dist = cal_data["camera_mtx"], cal_data["dist_coeffs"]

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Correct image
dst = cv2.undistort(img, mtx, dist)

# Plots results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)

ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=50)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
