import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Lambda, Flatten, Cropping2D, Input

# Dropout probability
DROP_PROB = 0.2

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Training/testing/validation data splitting
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_samples, test_samples = train_test_split(train_samples, test_size=0.2)


def generator(samples, batch_size=32):
    """Generates pairs of images and appropriate steering angles.

    Retrieves images from disk in a batch-wise, memory efficient way.

    :param samples: list of file names in disk
    :param batch_size: number of pairs to return each time the generator is called
    :return: batch of image/steering angle pairs
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # Lists of images and angles for this batch
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Correction to be applied to right and left cameras
                correction = 0.20

                # Change this directory string in your machine
                name = '/home/agripino/IMG/' + batch_sample[0].split('/')[-1]
                name_l = '/home/agripino/IMG/' + batch_sample[1].split('/')[-1]
                name_r = '/home/agripino/IMG/' + batch_sample[2].split('/')[-1]

                # Read images from the 3 cameras
                center_image = cv2.imread(name)
                left_image = cv2.imread(name_l)
                right_image = cv2.imread(name_r)

                # Flip the center image horizontally
                center_image_hf = cv2.flip(center_image, 1)

                # Compute steering angles
                center_angle = float(batch_sample[3])
                center_angle_hf = -center_angle
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # Append images
                images.append(center_image)
                images.append(center_image_hf)
                images.append(left_image)
                images.append(right_image)

                # Append corresponding angles
                angles.append(center_angle)
                angles.append(center_angle_hf)
                angles.append(left_angle)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            # Shuffle and return images and steering angles for this batch
            yield shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

model = Sequential()

# Select the mainly the road region
model.add(Cropping2D(cropping=((50, 20), (1, 1)), input_shape=(160, 320, 3)))

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
                 input_shape=(90, 318, 3),
                 output_shape=(90, 318, 3)))

# Architecture similar to, but lighter than, the one from the NVIDIA paper
model.add(Conv2D(nb_filter=18, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(nb_filter=48, nb_row=3, nb_col=3, activation="relu"))
model.add(Dropout(p=DROP_PROB))

model.add(Conv2D(nb_filter=48, nb_row=3, nb_col=3, activation="relu"))
model.add(Dropout(p=DROP_PROB))

model.add(Flatten())

model.add(Dense(output_dim=1064, activation="tanh"))
model.add(Dense(output_dim=100, activation="tanh"))
model.add(Dense(output_dim=50, activation="tanh"))
model.add(Dense(output_dim=10, activation="tanh"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=4*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=5)

# Print performance on unseen data
print(model.evaluate_generator(test_generator, val_samples=1000))

# Save the model
model.save("model.h5")
