import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def color_hist(img, bins=32, bins_range=(0, 256)):
    # Compute the histogram of each channel
    red_hist = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
    green_hist = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
    blue_hist = np.histogram(img[:, :, 2], bins=bins, range=bins_range)

    # Compute bin centers
    bin_edges = red_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1]) / 2.0

    # Concatenate histograms into a single feature vector
    hist_features = np.concatenate((red_hist[0], green_hist[0], blue_hist[0]))

    return red_hist, green_hist, blue_hist, bin_centers, hist_features


def get_hist_features(img, bins=32, bins_range=(0, 256)):
    # Compute the histogram of each channel
    ch1_hist = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
    ch2_hist = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
    ch3_hist = np.histogram(img[:, :, 2], bins=bins, range=bins_range)

    # Concatenate histograms into a single feature vector
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

    return hist_features


if __name__ == "__main__":
    image = mpimg.imread("cutouts/cutout1.jpg")

    rhist, ghist, bhist, centers, features = color_hist(image)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Color histograms", fontsize=16)

    for ax, hist, title in zip((ax1, ax2, ax3), (rhist, ghist, bhist), ('Red', 'Green', 'Blue')):
        ax.bar(centers, hist[0])
        ax.grid(True)
        ax.set_xlim(0, 256)
        ax.set_title(title)

    plt.show()
