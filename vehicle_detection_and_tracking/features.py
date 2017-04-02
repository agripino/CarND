import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


cs_dict = {
    'HSV': cv2.COLOR_RGB2HSV,
    'LUV': cv2.COLOR_RGB2LUV,
    'HLS': cv2.COLOR_RGB2HLS,
    'YUV': cv2.COLOR_RGB2YUV,
    'YCrCb': cv2.COLOR_RGB2YCrCb,
    'GRAY': cv2.COLOR_RGB2GRAY
}


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    if color_space != 'RGB':
        feature_image = cv2.cvtColor(img, cs_dict[color_space])
    else:
        feature_image = np.copy(img)

    return cv2.resize(feature_image, size).ravel()


def get_spatial_features():
    image = mpimg.imread('cutouts/cutout1.jpg')

    for i, cs in enumerate(cs_dict.keys(), 1):
        features = bin_spatial(image, cs)
        plt.subplot(2, 3, i)
        plt.title(cs)
        plt.plot(features)
    plt.show()


if __name__ == '__main__':
    get_spatial_features()
