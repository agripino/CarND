import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from drawing import draw_boxes


class InvalidTemplateMatchingMethod(Exception):
    def __init__(self):
        super().__init__("Unknown template matching method provided.")


def find_matches(img, template_list, method=cv2.TM_SQDIFF_NORMED):
    if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
                      cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
                      cv2.TM_CCORR, cv2.TM_CCORR_NORMED]:
        raise InvalidTemplateMatchingMethod()

    img_copy = np.copy(img)

    bbox_list = []

    for template in template_list:
        img_temp = mpimg.imread(template)

        res = cv2.matchTemplate(img_copy, img_temp, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        w, h = img_temp.shape[1], img_temp.shape[0]

        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))

    return bbox_list


if __name__ == "__main__":
    import glob

    img_templates = sorted(glob.glob("cutouts/cutout*"))

    image = mpimg.imread("cutouts/bbox-example-image.jpg")

    bboxes = find_matches(image, img_templates, cv2.TM_SQDIFF_NORMED)

    plt.imshow(draw_boxes(image, bboxes))
    plt.show()
