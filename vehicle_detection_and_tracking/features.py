import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog


cs_dict = {
    'HSV': cv2.COLOR_RGB2HSV,
    'LUV': cv2.COLOR_RGB2LUV,
    'HLS': cv2.COLOR_RGB2HLS,
    'YUV': cv2.COLOR_RGB2YUV,
    'YCrCb': cv2.COLOR_RGB2YCrCb,
    'GRAY': cv2.COLOR_RGB2GRAY
}


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """Computes spatially binned features of an image.
    """
    if color_space != 'RGB':
        feature_image = cv2.cvtColor(img, cs_dict[color_space])
    else:
        feature_image = np.copy(img)

    return cv2.resize(feature_image, size).ravel()


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=False, transform_sqrt=True):
    """Computes HOG features of an image.
    
    Returns a visualization of the features if vis is set to True.
    """
    if vis:
        features, hog_image = hog(img, orient, (pix_per_cell, pix_per_cell),
                                  (cell_per_block, cell_per_block), True,
                                  transform_sqrt=transform_sqrt, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orient, (pix_per_cell, pix_per_cell),
                       (cell_per_block, cell_per_block), False,  # no hog image
                       transform_sqrt=transform_sqrt, feature_vector=feature_vec)
        return features


def extract_features(img):
    """img is represented in the RGB color space.
    """
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    ch_l_hog_features = get_hog_features(img_hls[:, :, 1], orient=9, pix_per_cell=8,
                                         cell_per_block=2, vis=False, feature_vec=True,
                                         transform_sqrt=True)
    ch_s_hog_features = get_hog_features(img_hls[:, :, 2], orient=9, pix_per_cell=8,
                                         cell_per_block=2, vis=False, feature_vec=True,
                                         transform_sqrt=True)
    spatial_features = bin_spatial(img, color_space='LUV', size=(24, 24))

    return np.hstack((spatial_features, ch_l_hog_features, ch_s_hog_features))


def get_spatial_features():
    image = mpimg.imread('cutouts/cutout1.jpg')

    for i, cs in enumerate(cs_dict.keys(), 1):
        features = bin_spatial(image, cs)
        plt.subplot(2, 3, i)
        plt.title(cs)
        plt.plot(features)
    plt.show()


def plot_hog_image():
    image = cv2.imread('cutouts/cutout1.jpg', cv2.IMREAD_GRAYSCALE)

    features, hog_image = get_hog_features(image, 9, 8, 2, vis=True)

    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Grayscale image')

    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG features visualization')
    plt.show()


if __name__ == '__main__':
    get_spatial_features()

    plot_hog_image()
