import glob
import os
import pickle
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from features import extract_features


def get_preprocessed_data():
    # Get data from images
    if not (os.path.exists("data.pkl") and os.path.exists("scaler.pkl")):
        print("Extracting data from images...")

        # Get vehicle and non vehicle image paths
        vehicle_paths = glob.glob("vehicles/**/*.png")
        non_vehicles_paths = glob.glob("non-vehicles/**/*.png")
        all_paths = vehicle_paths + non_vehicles_paths

        # Collect features...
        features = []
        for path in tqdm(all_paths, file=sys.stdout):
            img = mpimg.imread(path)
            feature_vector = extract_features(img)
            features.append(feature_vector)
        features = np.vstack(features)

        # ... and the corresponding labels.
        vehicle_labels = np.ones(len(vehicle_paths))
        non_vehicle_labels = np.zeros(len(non_vehicles_paths))
        labels = np.hstack((vehicle_labels, non_vehicle_labels))

        # standardize features and store standard scaler
        scaler = StandardScaler().fit(features)
        with open("scaler.pkl", "wb") as file:
            pickle.dump(scaler, file, protocol=4)

        # Transform features
        features = scaler.transform(features)

        # Shuffle samples
        features, labels = shuffle(features, labels)

        # Persist data for training/testing/validation
        data = {'features': features, 'labels': labels}
        with open("data.pkl", "wb") as file:
            pickle.dump(data, file, protocol=4)
    else:
        print("Loading pickled data...")
        # Get data previously saved
        with open("data.pkl", "rb") as file:
            data = pickle.load(file)
            features = data["features"]
            labels = data["labels"]

    return features, labels


def train_classifier():
    """Training a classifier with default SVC parameters.
    """
    features, labels = get_preprocessed_data()
    print("Number of samples: {}".format(features.shape[0]))
    print("Number of features: {}".format(features.shape[1]))

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.2,
                                                                                random_state=8)
    print("Training classifier...")
    clf = SVC()
    clf.fit(train_features, train_labels)

    # Store trained classifier
    with open("clf.pkl", "wb") as file:
        pickle.dump(clf, file, protocol=4)

    # Report score on test data
    score = clf.score(test_features, test_labels)
    print("Classifier score on test data: {:.4f}".format(score))

if __name__ == "__main__":
    train_classifier()
