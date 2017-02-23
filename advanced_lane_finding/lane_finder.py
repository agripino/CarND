import numpy as np
import cv2


class LaneFinder:
    def __init__(self, n_windows=9, margin=100, min_pix=50):
        # Number of windows
        self.n_windows = n_windows

        # Margin: 2 * margin = search window width
        self.margin = margin

        # Minimum number of pixels in a window to update window center
        self.min_pix = min_pix

    def __call__(self, img):
        """Applies the lane finding pipeline to a given image"""
        pass

    def blind_search(self, warped_bin_img):
        """Searches for lane lines without previous information"""
        histogram = np.sum(warped_bin_img[warped_bin_img.shape[0] / 2:, :],
                           axis=0)

        midpoint = np.int(histogram.shape[1] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(warped_bin_img.shape[0] / self.n_windows)

        nonzero = warped_bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_bin_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_bin_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self. margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.min_pix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_pix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
