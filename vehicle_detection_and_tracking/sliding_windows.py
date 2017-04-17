import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from drawing import draw_boxes


def slide_window(img, x_start_stop=(0, None), y_start_stop=(0, None), xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x or y start/stop positions not defined, set to image size
    if None in x_start_stop:
        x_start_stop = [0, img.shape[1]]
    if None in y_start_stop:
        y_start_stop = [0, img.shape[0]]

    # Compute the span of the region to be searched
    span = (x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0])

    # Compute the number of pixels per step in x/y
    xpix_step = int((1 - xy_overlap[0]) * xy_window[0])
    ypix_step = int((1 - xy_overlap[1]) * xy_window[1])

    # Compute the number of windows in x/y
    x_windows = int(span[0] - int(xy_window[0]*xy_overlap[0])) // xpix_step
    y_windows = int(span[1] - int(xy_window[1]*xy_overlap[1])) // ypix_step

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(y_windows):
        for xs in range(x_windows):
            # Calculate window position
            start_x = xs*xpix_step + x_start_stop[0]
            end_x = start_x + xy_window[0]
            start_y = ys*ypix_step + y_start_stop[0]
            end_y = start_y + xy_window[1]

            # Append window position to list
            window_list.append(((start_x, start_y), (end_x, end_y)))
    return window_list


if __name__ == "__main__":
    image = mpimg.imread("cutouts/bbox-example-image.jpg")

    bboxes_1 = slide_window(image, x_start_stop=[40, 1280], y_start_stop=[480, 720], xy_window=[240, 240],
                            xy_overlap=[0.75, 0.75])
    bboxes_2 = slide_window(image, x_start_stop=[40, 1280], y_start_stop=[480, 720], xy_window=[192, 192],
                            xy_overlap=[0.75, 0.75])
    bboxes_3 = slide_window(image, x_start_stop=[40, 1280], y_start_stop=[480, 720], xy_window=[128, 128])
    bboxes_4 = slide_window(image, x_start_stop=[180, 1100], y_start_stop=[480, 580], xy_window=[64, 64])

    print("{} boxes in total".format(len(bboxes_1) + len(bboxes_2) + len(bboxes_3) + len(bboxes_4)))

    image_1 = draw_boxes(image, bboxes_1)
    image_2 = draw_boxes(image, bboxes_2)
    image_3 = draw_boxes(image, bboxes_3)
    image_4 = draw_boxes(image, bboxes_4)

    plt.subplot(221)
    plt.imshow(image_1)
    plt.subplot(222)
    plt.imshow(image_2)
    plt.subplot(223)
    plt.imshow(image_3)
    plt.subplot(224)
    plt.imshow(image_4)

    plt.show()
