import pickle
from collections import deque
import numpy as np
from moviepy.editor import VideoFileClip
from multiprocessing import Pool, cpu_count
from features import get_hog_features, get_hist_features
import cv2
from scipy.ndimage.measurements import label

# Accumulated detections over past frames
integrated_detections = deque(maxlen=10)


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color=(0, 200, 255), thickness=6)
    # Return the image
    return img


def find_cars(inputs):
    """inputs is (img, y_start, y_stop, orient, scale, pix_per_cell, cell_per_block, clf, scaler).
    """
    img, y_start, y_stop, orient, scale, pix_per_cell, cell_per_block, clf, scaler = inputs

    roi = np.copy(img[y_start:y_stop, :, :]).astype(np.float32) / 255

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    if scale != 1:
        img_shape = gray_roi.shape
        gray_roi = cv2.resize(gray_roi, (np.int(img_shape[1] / scale), np.int(img_shape[0] / scale)))

    # Define blocks and steps as above
    nx_blocks = (gray_roi.shape[1] // pix_per_cell) - 1
    ny_blocks = (gray_roi.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nx_blocks - nblocks_per_window) // cells_per_step
    nysteps = (ny_blocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(gray_roi, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # List of detections for this Region of Interest
    roi_detections = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            sub_img = cv2.resize(roi[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            hist_features = get_hist_features(cv2.cvtColor(sub_img, cv2.COLOR_RGB2HLS), bins=8)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                roi_detections.append(((xbox_left, ytop_draw + y_start),
                                       (xbox_left + win_draw, ytop_draw + win_draw + y_start)))

    return roi_detections


class VehicleDetector:
    def __init__(self, classifier, scaler, n_procs=cpu_count()):
        self.classifier = classifier
        self.scaler = scaler
        self.proc_pool = Pool(n_procs)

    def __call__(self, image):
        # Get a list of detections for each region of interest
        detections = self.proc_pool.map(find_cars, [  # (image, 450, 720, 9,  2, 8, 2, self.classifier, self.scaler),
                                                    (image, 360, 592, 9, 2, 8, 2, self.classifier, self.scaler),
                                                    ])

        # Accumulate lists of detections over past frames
        integrated_detections.append(detections)

        heatmap = np.zeros_like(image[:, :, 0])
        for frame_detections in integrated_detections:
            for bboxes in frame_detections:
                add_heat(heatmap, bboxes)

        # Apply threshold
        heatmap = apply_threshold(heatmap, 1)

        # Enumerate detected cars
        labels = label(heatmap)

        return draw_labeled_bboxes(image, labels)


if __name__ == "__main__":
    # Load classifier
    with open("clf.pkl", "rb") as file:
        clf = pickle.load(file)

    # Load scaler
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Create VehicleDetector object
    vehicle_detector = VehicleDetector(clf, scaler, n_procs=2)

    input_path = "./videos/project_video.mp4"
    output_path = "./videos/project_video_annotated.mp4"

    clip = VideoFileClip(input_path)
    output_video = clip.fl_image(vehicle_detector)

    # Write annotated video
    output_video.write_videofile(output_path, audio=False)
