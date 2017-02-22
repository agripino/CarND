import numpy as np
import cv2
import matplotlib.pyplot as plt


def hls_s_thresh(img, thresh=(0, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[(hls[:, :, 2] >= thresh[0]) & (hls[:, :, 2] <= thresh[1])] = 1

    return binary_output


def rgb_r_thresh(img, thresh=(0, 255)):
    # Image is assumed in BGR color space
    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[(img[:, :, 2] >= thresh[0]) & (img[:, :, 2] <= thresh[1])] = 1

    return binary_output


def hsv_h_thresh(img, thresh=(0, 179)):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[(hsv[:, :, 0] >= thresh[0]) & (hsv[:, :, 0] <= thresh[1])] = 1

    return binary_output


def hsv_v_thresh(img, thresh=(0, 255)):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 2])
    binary_output[(hsv[:, :, 2] >= thresh[0]) & (hsv[:, :, 2] <= thresh[1])] = 1

    return binary_output


if __name__ == "__main__":
    image = cv2.imread("./test_images/test6.jpg", cv2.IMREAD_UNCHANGED)

    hls_thresh = hls_s_thresh(image, (120, 255))
    rgb_thresh = rgb_r_thresh(image, (200, 255))

    hsv_h = hsv_h_thresh(image, (20, 40))
    hsv_v = hsv_v_thresh(image, (210, 255))

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(hls_thresh, cmap="gray")
    ax1.set_title("HLS S Threshold")

    ax2.imshow(rgb_thresh, cmap="gray")
    ax2.set_title("RGB R Threshold")

    ax3.imshow(hls_thresh | rgb_thresh, cmap="gray")
    ax3.set_title("HLS S | RGB R")

    ax4.imshow(hsv_h, cmap="gray")
    ax4.set_title("HSV H Threshold")

    ax5.imshow(hsv_v, cmap="gray")
    ax5.set_title("HSV V Threshold")

    ax6.imshow(hsv_h & hsv_v, cmap="gray")
    ax6.set_title("HSV & & HSV V")
    plt.show()
