import numpy as np
import cv2
import glob
import pickle

# Termination criteria for the cv2.cornerSubPix function
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)

# Grab images for calibration
images = glob.glob(pathname="./camera_cal/*.jpg")

# Points in the 3D space
all_obj_points = []

# Corresponding points in image space
all_img_points = []

for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Search for patterns in the cartesian product [9, 8, 7, 6]x[6, 5, 4]
    for pattern in [(nx, ny) for ny in [6, 5, 4] for nx in [9, 8, 7, 6]]:
        # Try to find corners following this pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern)

        # Corners were found for this pattern!
        if ret:
            print("Matched pattern for {}: {}".format(image, pattern))
            cv2.drawChessboardCorners(img, pattern, corners, ret)
            cv2.imshow("Current image", img)
            cv2.waitKey(300)

            # Grid of object points in chessboard square units
            obj_points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            obj_points[:, :2] = np.mgrid[0:pattern[0],
                                         0:pattern[1]].T.reshape(-1, 2)

            all_obj_points.append(obj_points)

            # Adjust corners to sub pixel accuracy
            sub_pixel_corners = cv2.cornerSubPix(gray, corners, (11, 11),
                                                 (-1, -1), criteria)
            all_img_points.append(sub_pixel_corners)
            break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points,
                                                   all_img_points,
                                                   gray.shape[::-1],
                                                   None, None)

# Collect calibration data in a dictionary
cal_data = {"camera_mtx": mtx, "dist_coeffs": dist}

# Store calibration data for future use
with open("cal_data.p", "wb") as f:
    pickle.dump(cal_data, f)
