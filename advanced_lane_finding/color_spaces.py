import numpy as np
import cv2
import matplotlib.pyplot as plt


def hls_s_thresh(img, thresh=(0, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[(hls[:, :, 2] >= thresh[0]) & (hls[:, :, 2] < thresh[1])] = 1

    return binary_output


def rgb_r_thresh(img, thresh=(0, 255)):
    # Image is assumed in BGR color space
    # Create binary output mask
    binary_output = np.zeros_like(img[:, :, 0])
    binary_output[(img[:, :, 2] >= thresh[0]) & (img[:, :, 2] < thresh[1])] = 1

    return binary_output


if __name__ == "__main__":
    image = cv2.imread("./test_images/test6.jpg", cv2.IMREAD_UNCHANGED)

    hls_thresh = hls_s_thresh(image, (160, 255))
    rgb_thresh = rgb_r_thresh(image, (200, 255))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(hls_thresh, cmap="gray")
    ax2.imshow(rgb_thresh, cmap="gray")
    ax3.imshow(hls_thresh | rgb_thresh, cmap="gray")
    plt.show()
