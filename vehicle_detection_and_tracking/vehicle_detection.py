import pickle
from moviepy.editor import VideoFileClip
from multiprocessing import Pool
from drawing import draw_boxes


def find_cars(inputs):
    """inputs is (img, y_start, y_stop, scale, clf, scaler).
    """
    img, y_start, y_stop, scale, clf, scaler = inputs

    return [((0, 0), (100, 100)), ((200, 200), (400, 400))]  # dummy return value


class VehicleDetector:
    def __init__(self, classifier, scaler, n_procs=7):
        self.classifier = classifier
        self.scaler = scaler
        self.proc_pool = Pool(n_procs)

    def __call__(self, image):
        # Apply detection pipeline
        detections = self.proc_pool.map(find_cars, [(image, 500, 720, 8, self.classifier, self.scaler),
                                                    (image, 500, 720, 16, self.classifier, self.scaler),
                                                    (image, 500, 720, 24, self.classifier, self.scaler),
                                                    (image, 500, 720, 32, self.classifier, self.scaler)])
        for bboxes in detections:
            image = draw_boxes(image, bboxes)
        return image


if __name__ == "__main__":
    # Load classifier
    with open("clf.pkl", "rb") as file:
        clf = pickle.load(file)

    # Load scaler
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Create VehicleDetector object
    vehicle_detector = VehicleDetector(clf, scaler)

    input_path = "./videos/test_video.mp4"
    output_path = "./videos/test_video_annotated.mp4"

    clip = VideoFileClip(input_path)
    output_video = clip.fl_image(vehicle_detector)

    # Write annotated video
    output_video.write_videofile(output_path, audio=False)
