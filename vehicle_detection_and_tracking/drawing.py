import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6, line_type=cv2.LINE_AA):
    """Draws bounding boxes contained in bboxes on a copy of img and returns it
    """

    img_copy = np.copy(img)

    for box in bboxes:
        cv2.rectangle(img_copy, box[0], box[1], color=color, thickness=thick, lineType=line_type)

    return img_copy


def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plots a 3D scatter of pixels of an image
    """

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax


def drawing_boxes():
    image = mpimg.imread("cutouts/bbox-example-image.jpg")

    bboxes = [((270, 490), (395, 580)), ((475, 495), (565, 575)), ((849, 678), (1135, 512))]

    plt.imshow(draw_boxes(image, bboxes))
    plt.show()


def plotting_3d(filename):
    # Read a color image
    img = cv2.imread(filename)

    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_BGR2HLS)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
    img_small_YCrCB = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1] for plotting

    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()

    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()

    plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    plt.show()

    plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
    plt.show()

    plot3d(img_small_YCrCB, img_small_rgb, axis_labels=list("YCrCb"))
    plt.show()


if __name__ == "__main__":
    drawing_boxes()

    plotting_3d('cutouts/bbox-example-image.jpg')
