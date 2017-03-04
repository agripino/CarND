import numpy as np
import cv2
import matplotlib.pyplot as plt
from color_spaces import hls_s_thresh


def warp_image(bin_image):
    # Width
    w = bin_image.shape[1]
    # Height
    h = bin_image.shape[0]

    src = np.float32([[(w / 2) - 60, h / 2 + 100],
                      [((w / 6) - 10), h],
                      [(w * 5 / 6) + 10, h],
                      [(w / 2 + 60), h / 2 + 100]])

    dst = np.float32([[(w / 4), 0],
                      [(w / 4), h],
                      [(w * 3 / 4), h],
                      [(w * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped_image = cv2.warpPerspective(bin_image, M, bin_image.shape[::-1])

    return warped_image, M, Minv

if __name__ == "__main__":
    img = cv2.imread("./test_images/test6.jpg", cv2.IMREAD_UNCHANGED)
    bin_img = hls_s_thresh(img, (120, 255))
    warped, M, Minv = warp_image(bin_img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(bin_img, cmap="gray")
    ax1.set_title("Binary image")

    ax2.imshow(warped, cmap="gray")
    ax2.set_title("Warped image")
    plt.show()
