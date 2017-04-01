import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6, line_type=cv2.LINE_AA):
    """Draws bounding boxes contained in bboxes on a copy of img and returns it
    """

    img_copy = np.copy(img)

    for box in bboxes:
        cv2.rectangle(img_copy, box[0], box[1], color=color, thickness=thick, lineType=line_type)

    return img_copy


if __name__ == "__main__":
    image = mpimg.imread("cutouts/bbox-example-image.jpg")

    bboxes = [((270, 490), (395, 580)), ((475, 495), (565, 575)),  ((849, 678), (1135, 512))]

    plt.imshow(draw_boxes(image, bboxes))
    plt.show()
