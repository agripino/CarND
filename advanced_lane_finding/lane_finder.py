import numpy as np
from collections import deque
import pickle
import cv2
from thresholds import dir_thresh
from color_spaces import hls_s_thresh
from image_warping import warp_image


class Line:
    def __init__(self, n_last=5):
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_x_fitted = deque(maxlen=n_last)

        # average x values of the fitted line over the last n iterations
        self.best_x = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.recent_fit = deque(maxlen=n_last)

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None


class LaneFinder:
    def __init__(self, cal_file="cal_data.p", n_windows=9, margin=100, min_pix=50,
                 n_last=5):
        # Camera matrix
        self.camera_mtx = None

        # Camera distortion coefficients
        self.dist_coeffs = None

        # Number of windows
        self.n_windows = n_windows

        # Margin: 2 * margin = search window width
        self.margin = margin

        # Minimum number of pixels in a window to update window center
        self.min_pix = min_pix

        # Lane lines
        self.left_line = Line(n_last)
        self.right_line = Line(n_last)

        # Read calibration data from pickle file
        self.load_cal_data(cal_file)

        # Current operational mode
        self.op_mode = "BLIND"

        # Op modes transition table
        self.transition = {("BLIND", "OK"): "TRACK",
                           ("BLIND", "NOT_OK"): "BLIND",
                           ("TRACK", "OK"): "TRACK",
                           ("TRACK", "NOT_OK"): "BLIND"}

        # Mapping from op mode to action
        self.action = {"BLIND": self.blind_search,
                       "TRACK": self.track_lines}

        # Current number of consecutive failed detections
        self.failed_detections = 0

        # Maximum number of failed detections allowed before doing blind search
        self.max_failed_detections = 10

    def __call__(self, img):
        """Applies the lane finding pipeline to a given image. img is a RGB image
        """
        # Removes estimated camera distortion from the RGB image
        undist = cv2.undistort(img, self.camera_mtx, self.dist_coeffs)

        # Apply thresholds and get a binary image
        binary = self.apply_thresholds(undist)

        # Warp binary image and get inverse transformation
        warped, M, M_inv = self.warp_binary(binary)

        # Do whatever the current states requires
        action = self.action[self.op_mode]

        action(warped)

        # Draw lines

        return img

    def blind_search(self, binary_warped):
        """Searches for lane lines without previous information"""
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :],
                           axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(binary_warped.shape[0] / self.n_windows)

        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_x_current = left_x_base
        right_x_current = right_x_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = left_x_current - self. margin
            win_xleft_high = left_x_current + self.margin
            win_xright_low = right_x_current - self.margin
            win_xright_high = right_x_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.min_pix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.min_pix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        # Fit a second order polynomial to each, but check before if the line
        # was detected
        if len(left_y) > 2:
            left_fit = np.polyfit(left_y, left_x, 2)
            self.left_line.detected = True
            self.left_line.recent_fit.append(left_fit)
            self.left_line.best_fit = np.mean(self.left_line.recent_fit, axis=0)
        else:
            self.left_line.detected = False
            left_fit = None

        if len(right_y) > 2:
            right_fit = np.polyfit(right_y, right_x, 2)
            self.right_line.detected = True
            self.right_line.recent_fit.append(right_fit)
            self.right_line.best_fit = np.mean(self.right_line.recent_fit, axis=0)
        else:
            self.right_line.detected = False
            right_fit = None

    def track_lines(self, binary_warped):
        # Get a pair of the row and column indices of nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_lane_inds = ((nonzero_x > (self.left_line.best_x - self.margin)) &
                          (nonzero_x < (self.left_line.best_x + self.margin)))

        right_lane_inds = ((nonzero_x > (self.right_line.best_x - self.margin)) &
                           (nonzero_x < (self.right_line.best_x + self.margin)))

        # Again, extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        return binary_warped

    def draw_lines(self, binary_warped):
        pass

    def load_cal_data(self, cal_file):
        with open(cal_file, "rb") as f:
            cal_data = pickle.load(f)
            self.camera_mtx = cal_data["camera_mtx"]
            self.dist_coeffs = cal_data["dist_coeffs"]

    @staticmethod
    def apply_thresholds(rgb_img):
        dir_binary = dir_thresh(rgb_img, thresh=(0.7, 1.3))
        hls_binary = hls_s_thresh(rgb_img, thresh=(130, 255))
        binary_output = np.zeros_like(dir_binary)
        binary_output[(dir_binary == 1) & (hls_binary == 1)] = 1
        return binary_output

    @staticmethod
    def warp_binary(bin_img):
        warped, M, M_inv = warp_image(bin_img)
        return warped, M, M_inv
